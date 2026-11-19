#!/usr/bin/env python
"""Flow-GRPO (clipped GRPO on the SDE-sampled flow-matching chain) with a LoRA adapter on Stable Diffusion 3.5 Medium,
prototype area "image generation, SD3.5" (MULTIMODAL-SPEC section 6 row 2b; prototype, unpromotable).

A minimal port of yifan123/flow_grpo at commit 879042cf5707f8b90daa98d147d7deac2317c5da (MIT License, Copyright (c)
2025 Jie Liu) onto current diffusers + peft, configuration `general_ocr_sd3_1gpu`. Ported pieces, cited by file:line
of that commit:
  - `sde_step_with_logprob` ("sde" type): flow_grpo/diffusers_patch/sd3_sde_with_logprob.py:10-92;
  - the sampling loop with classifier-free guidance: flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:41-191;
  - prompt encoding (CLIP-L, CLIP-G penultimate hidden states + pooled projections, T5 with 128 tokens):
    flow_grpo/diffusers_patch/train_dreambooth_lora_sd3.py:17-144, scripts/train_sd3.py:124-131, :614-621;
  - `compute_log_prob`: scripts/train_sd3.py:181-215; the GRPO update with the KL term: scripts/train_sd3.py:831-962;
  - advantages: flow_grpo/stat_tracking.py:5-36 (`global_std`), cleared every epoch (train_sd3.py:768), the
    zero-advantage filter and its padding to a multiple of the batch count: scripts/train_sd3.py:783-823;
  - EMA of the LoRA parameters (optional here): flow_grpo/ema.py:8-53, train_sd3.py:444-445, :959-960, :323-332;
  - the OCR reward: flow_grpo/ocr.py:8-63 and rewards.py:128-140 (PaddleOCR, English, no angle classifier, CPU);
  - LoRA: r 32, alpha 64, gaussian init, attention projections of the joint blocks: scripts/train_sd3.py:417-440;
  - hyperparameters: config/base.py:5-127 and config/grpo.py:444-482 (`general_ocr_sd3_1gpu`).
Why a port and not the repository itself: README.md "Trainer choice".

Changes against flow_grpo (README.md "Changes against flow_grpo"): LoRA through diffusers `add_adapter` (peft) and the
frozen transformer in fp16 under autocast instead of fp32 weights under accelerate autocast (24 GB); the three text
encoders run once at start for every prompt of the run and are then freed (T5-XXL alone is about 9.5 GB); training
micro-batches of `train_batch` samples with the same gradient (mean over 32 samples x 9 timesteps per optimizer step);
gradient checkpointing; per-sample recorded noise seeds (CPU generator) instead of the global RNG; prompts drawn per
epoch from a registered training shard without replacement; EMA off by default; an observed consumption log; cuts
written atomically with a completion marker; no wandb, no in-training evaluation.

Outputs in --out (the run's only read-write mount):
  env-freeze.txt     `name==version` lines of the runtime, written before torch is imported (MM-33)
  env-runtime.json   interpreter, image CUDA version, lock and script SHA-256 (EXT-83), sampler code SHA-256
  run_args.json      every setting, seeds and their derivation, base model binding, data binding
  samples.jsonl      observed consumption log: one line per generated sample, written at sampling time
  epoch_plan.jsonl   per epoch, before its first optimizer step: kept and filtered (zero-advantage) sample ids
  train_steps.jsonl  one line per optimizer step: the sample ids whose gradients it applied, written before the step
  train_log.jsonl    per optimizer step: policy loss, KL, approx_kl, clipfrac, grad norm, scaler scale
  memory.jsonl       per epoch: seconds, peak device memory, peak RSS, reward statistics
  attempts.jsonl     one line per start and finish of this command (gpu-step.sh reruns after a preemption)
  checkpoint-<E>/    the cut after epoch E (1-based); `cut.json` written last

Usage (inside the container; README.md has the request shape):
  python train_flow_grpo.py --profile smoke|full --out /out [--seed S] [--epochs N]
  python train_flow_grpo.py --profile cpu-sanity --device cpu --out DIR      # host CPU check, tiny settings
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "models" / "stable-diffusion-3.5-medium"
DEFAULT_PROMPTS = HERE / "prompts"
OCR_MODELS = HERE / "ocr-models"
MODELS_JSON = HERE / "MODELS.json"
BASE_NAME = "stable-diffusion-3.5-medium"

# scripts/train_sd3.py:419-433
LORA = {"r": 32, "lora_alpha": 64, "init_lora_weights": "gaussian",
        "target_modules": ["attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
                           "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v"]}

COMMON = {
    # training sampler (config/grpo.py:444-455, config/base.py:60-63)
    "num_steps": 10,
    "guidance_scale": 4.5,
    "noise_level": 0.7,
    "resolution": 512,
    "max_sequence_length": 128,     # train_sd3.py:540, :619
    # deployment sampler (README "Deployment sampler"): flow_grpo evaluates with noise_level 0 (train_sd3.py:258) at
    # 40 steps (config/grpo.py:451); here the deployment sampler keeps the training step count (row 2b: 10 steps).
    "deploy_num_steps": 10,
    "deploy_noise_level": 0.0,
    # training (config/base.py:67-111, config/grpo.py:462-472)
    "learning_rate": 3.0e-4,
    "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_weight_decay": 1.0e-4, "adam_epsilon": 1.0e-8,
    "max_grad_norm": 1.0,
    "adv_clip_max": 5.0,
    "clip_range": 1.0e-4,
    "timestep_fraction": 0.99,      # int(10 * 0.99) = 9 trained steps, the last (sharpest) step is not trained
    "beta": 0.04,                   # KL coefficient against the adapter-disabled transformer
    "train_cfg": True,
    "num_inner_epochs": 1,
    "global_std": True,
    "ema": False,                   # flow_grpo: True (decay 0.9, every 8 steps); off here, README "Changes"
    "ema_decay": 0.9, "ema_update_step_interval": 8,
    # reward (config/grpo.py:477-479)
    "reward": "ocr",
    # memory knobs
    "precision": "fp16",            # frozen transformer and text encoders fp16, VAE fp32 (train_sd3.py:410), LoRA fp32
    "encoder_precision": "fp16",    # text encoders (cpu-sanity: bf16, so that T5 fits the host's memory)
    "gradient_checkpointing": True,
    "allow_tf32": True,
    "ocr_cpu_threads": 8,
}
PROFILES = {
    # epochs, prompts per epoch P, images per prompt G, sample batch (= one prompt group, train_sd3.py:482-489 with
    # k = 8, batch 8), train micro-batch, cut every N epochs. P x G = 64 samples per epoch = 8 sampling batches,
    # 2 optimizer steps of 32 samples (config/grpo.py:457-465 with gpu_number 1).
    "smoke": {"epochs": 2, "prompts_per_epoch": 8, "images_per_prompt": 8, "sample_batch": 8, "train_batch": 2,
              "save_every": 1},
    # full: set --epochs from the smoke's measured seconds per epoch (README "Profiles and cost"); default 10.
    "full": {"epochs": 10, "prompts_per_epoch": 8, "images_per_prompt": 8, "sample_batch": 8, "train_batch": 2,
             "save_every": 2},
    # host CPU sanity check only (README "CPU sanity check"): fp32, 128 px, 3 steps (2 trained), 2 prompts x 2 images.
    "cpu-sanity": {"epochs": 1, "prompts_per_epoch": 2, "images_per_prompt": 2, "sample_batch": 2, "train_batch": 1,
                   "save_every": 1, "num_steps": 3, "deploy_num_steps": 3, "resolution": 128, "precision": "fp32",
                   "encoder_precision": "fp16",
                   # host check only: 128 px images carry no readable text, so every OCR reward is 0, every
                   # advantage 0 and every sample filtered; a 0.01 x mean-brightness term exercises the update path.
                   "reward": "ocr+brightness_sanity",
                   "ocr_cpu_threads": 4},
}
CUT_MARKER = "cut.json"
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"
CUT_FILES = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json", "trainer_state.json")
DEPLOYABLE = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json")
ALGORITHM_REFERENCE = "yifan123/flow_grpo 879042cf5707f8b90daa98d147d7deac2317c5da"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def canonical(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_json(path: Path, doc) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# BEGIN SHARED SAMPLER (byte-identical in train_flow_grpo.py and eval_images.py; its SHA-256 is `sampler_code_sha256`)
# ---------------------------------------------------------------------------

def derive_seed(*parts) -> int:
    """63-bit seed from the SHA-256 of the parts joined by ':' (recorded derivation, MM-29)."""
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()[:8], "big") >> 1


def rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def encode_all_prompts(base_dir, prompts, device, dtype, max_sequence_length, out_dtype=None, batch_size=8):
    """Load the three text encoders, encode every prompt once, free the encoders. Port of flow_grpo
    train_dreambooth_lora_sd3.py:17-144 (`encode_prompt`): CLIP-L and CLIP-G give their penultimate hidden states
    (:83-84) and pooled projections (:83); the CLIP states are concatenated on the channel axis, zero-padded to
    T5's width and stacked in front of T5's 128 tokens (:136-142). Returns {prompt: (embeds, pooled)} on the CPU in
    `out_dtype` (default `dtype`). T5 alone is about 9.5 GB in fp16, so the encoders never share the device with the
    transformer."""
    import gc

    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    base_dir = str(base_dir)
    out_dtype = out_dtype or dtype
    kw = {"torch_dtype": dtype, "local_files_only": True}
    if str(device).startswith("cuda"):
        kw["device_map"] = str(device)       # load straight to the GPU: keeps the host RSS of the T5 load low
    toks = [CLIPTokenizer.from_pretrained(base_dir, subfolder="tokenizer", local_files_only=True),
            CLIPTokenizer.from_pretrained(base_dir, subfolder="tokenizer_2", local_files_only=True),
            T5TokenizerFast.from_pretrained(base_dir, subfolder="tokenizer_3", local_files_only=True)]
    encs = [CLIPTextModelWithProjection.from_pretrained(base_dir, subfolder="text_encoder", **kw).to(device),
            CLIPTextModelWithProjection.from_pretrained(base_dir, subfolder="text_encoder_2", **kw).to(device),
            T5EncoderModel.from_pretrained(base_dir, subfolder="text_encoder_3", **kw).to(device)]
    for e in encs:
        e.requires_grad_(False)
        e.eval()
    out = {}
    uniq = list(dict.fromkeys(prompts))
    with torch.no_grad():
        for b0 in range(0, len(uniq), batch_size):
            batch = uniq[b0:b0 + batch_size]
            clip_states, clip_pooled = [], []
            for tok, enc in zip(toks[:2], encs[:2]):
                ids = tok(batch, padding="max_length", max_length=77, truncation=True,
                          return_tensors="pt").input_ids.to(device)
                o = enc(ids, output_hidden_states=True)
                clip_pooled.append(o[0])
                clip_states.append(o.hidden_states[-2].to(dtype))
            ids = toks[2](batch, padding="max_length", max_length=max_sequence_length, truncation=True,
                          add_special_tokens=True, return_tensors="pt").input_ids.to(device)
            t5 = encs[2](ids)[0].to(dtype)
            clip = torch.cat(clip_states, dim=-1)
            clip = torch.nn.functional.pad(clip, (0, t5.shape[-1] - clip.shape[-1]))
            embeds = torch.cat([clip, t5], dim=-2)
            pooled = torch.cat(clip_pooled, dim=-1)
            for k, p in enumerate(batch):
                out[p] = (embeds[k].to("cpu", out_dtype).clone(), pooled[k].to("cpu", out_dtype).clone())
    del encs
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def load_transformer_vae(base_dir, device, dtype):
    """SD3.5 transformer (frozen, `dtype`), VAE in float32 (flow_grpo train_sd3.py:410), flow-matching scheduler."""
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel

    base_dir = str(base_dir)
    transformer = SD3Transformer2DModel.from_pretrained(base_dir, subfolder="transformer", torch_dtype=dtype,
                                                        local_files_only=True).to(device)
    vae = AutoencoderKL.from_pretrained(base_dir, subfolder="vae", torch_dtype=torch.float32,
                                        local_files_only=True).to(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_dir, subfolder="scheduler",
                                                                local_files_only=True)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    return transformer, vae, scheduler


def sde_step_with_logprob(sigmas, step_index, model_output, sample, noise_level, prev_sample=None, noise=None):
    """flow_grpo sd3_sde_with_logprob.py:10-92, sde_type 'sde'. `step_index` (B,) long: the index of the current
    timestep in the schedule (the port of `index_for_timestep`, :41-42). float32 throughout (:35-39). Returns
    (prev_sample, log_prob, prev_sample_mean, std_dev_t); log_prob is the mean over all but the batch dimension (:89)
    and None for noise_level 0 (the deterministic deployment sampler, an Euler ODE step)."""
    import torch

    model_output = model_output.float()
    sample = sample.float()
    shape = (-1,) + (1,) * (sample.ndim - 1)
    sigma = sigmas[step_index].to(sample.device, torch.float32).view(shape)                    # :43
    sigma_prev = sigmas[step_index + 1].to(sample.device, torch.float32).view(shape)           # :44
    sigma_max = float(sigmas[1])                                                                # :45
    dt = sigma_prev - sigma                                                                     # :46
    if noise_level == 0:
        return sample + model_output * dt, None, sample + model_output * dt, torch.zeros_like(sigma)
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, torch.full_like(sigma, sigma_max), sigma))) \
        * noise_level                                                                           # :49
    prev_sample_mean = (sample * (1 + std_dev_t ** 2 / (2 * sigma) * dt)
                        + model_output * (1 + std_dev_t ** 2 * (1 - sigma) / (2 * sigma)) * dt)  # :52
    if prev_sample is None:
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * noise                # :54-61
    prev_sample = prev_sample.float()
    log_prob = (-((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
                - torch.log(std_dev_t * torch.sqrt(-1 * dt))
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))                         # :63-67
    return prev_sample, log_prob.mean(dim=tuple(range(1, log_prob.ndim))), prev_sample_mean, std_dev_t


def predict_velocity(transformer, latents, timesteps, embeds, pooled, neg_embeds, neg_pooled, guidance_scale,
                     autocast_dtype):
    """Classifier-free guidance as sd3_pipeline_with_logprob.py:144-162 and train_sd3.py:181-197: one transformer
    call on [uncond, cond]; float32 out. Under autocast in fp16, as accelerate's mixed precision in flow_grpo."""
    import contextlib

    import torch

    ctx = (torch.autocast(latents.device.type, dtype=autocast_dtype) if autocast_dtype is not None
           else contextlib.nullcontext())
    with ctx:
        out = transformer(hidden_states=torch.cat([latents, latents]), timestep=torch.cat([timesteps, timesteps]),
                          encoder_hidden_states=torch.cat([neg_embeds, embeds]),
                          pooled_projections=torch.cat([neg_pooled, pooled]), return_dict=False)[0].float()
    uncond, cond = out.chunk(2)
    return uncond + guidance_scale * (cond - uncond)


def schedule(scheduler, num_steps):
    """retrieve_timesteps as the SD3 pipeline (sd3_pipeline_with_logprob.py:120-127): timesteps (N,) and sigmas
    (N + 1,) with the terminal 0, both float32 on the CPU."""
    import torch

    scheduler.set_timesteps(num_steps, device="cpu")
    return scheduler.timesteps.clone().float(), scheduler.sigmas.clone().float()


def sample_batch(transformer, vae, scheduler, embeds, pooled, neg_embeds, neg_pooled, seeds, num_steps, noise_level,
                 guidance_scale, resolution, device, autocast_dtype, keep_chain=True):
    """One batch of sd3_pipeline_with_logprob.py:41-191: initial latents and every step's noise from a CPU
    torch.Generator seeded with the sample's seed (device independent), the SDE step with `noise_level` (0: the
    Euler ODE), CFG. Returns images in [0, 1] (B, 3, H, W) float32 on the CPU and, with keep_chain, the latents
    (B, N + 1, 16, h, w), log-probs (B, N) and the timesteps (N,)."""
    import torch

    timesteps, sigmas = schedule(scheduler, num_steps)
    h = resolution // 8
    shape = (transformer.config.in_channels, h, h)
    gens = [torch.Generator("cpu").manual_seed(int(s)) for s in seeds]
    latents = torch.stack([torch.randn(shape, generator=g, dtype=torch.float32) for g in gens]).to(device)
    all_latents, all_logp = [latents], []
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            tb = torch.full((latents.shape[0],), float(t), dtype=torch.float32, device=device)
            v = predict_velocity(transformer, latents, tb, embeds, pooled, neg_embeds, neg_pooled, guidance_scale,
                                 autocast_dtype)
            noise = None
            if noise_level > 0:
                noise = torch.stack([torch.randn(shape, generator=g, dtype=torch.float32) for g in gens]).to(device)
            idx = torch.full((latents.shape[0],), i, dtype=torch.long)
            latents, logp, _, _ = sde_step_with_logprob(sigmas, idx, v, latents, noise_level, noise=noise)
            if keep_chain:
                all_latents.append(latents)
                if logp is not None:
                    all_logp.append(logp)
        x = (latents / vae.config.scaling_factor) + vae.config.shift_factor                   # :181
        img = vae.decode(x.to(vae.dtype), return_dict=False)[0].float()                       # :183
        img = (img / 2 + 0.5).clamp(0, 1).cpu()                                               # postprocess "pt"
    if not keep_chain:
        return img, None, None, timesteps
    logp = torch.stack(all_logp, dim=1) if all_logp else None
    return img, torch.stack(all_latents, dim=1), logp, timesteps


def to_uint8(img):
    """(3, H, W) float in [0, 1] -> (H, W, 3) uint8 RGB, as flow_grpo rewards.py:133-135."""
    import torch

    return (img * 255).round().clamp(0, 255).to(torch.uint8).numpy().transpose(1, 2, 0).copy()


def target_text(prompt: str) -> str:
    """flow_grpo ocr.py:32: the text between the first pair of double quotes."""
    return prompt.split('"')[1]


def ocr_accuracy(recognized: str, target: str) -> float:
    """flow_grpo ocr.py:46-60: spaces removed and lower case on both sides; 1 when the target is a substring of the
    recognized text, else 1 - min(Levenshtein distance, len(target)) / len(target)."""
    from rapidfuzz.distance.Levenshtein import distance

    recognized = recognized.replace(" ", "").lower()
    target = target.replace(" ", "").lower()
    dist = 0 if target in recognized else distance(recognized, target)
    dist = min(dist, len(target))
    return 1 - dist / len(target)


class PaddleOcrA:
    """OCR engine A, the training reward: flow_grpo ocr.py:8-63 (`PaddleOCR(use_angle_cls=False, lang="en",
    use_gpu=False, show_log=False)`, `.ocr(img, cls=False)`, text = concatenation of the recognized lines with a
    positive score). Built from paddleocr 2.10.0's own `TextSystem` (the class `PaddleOCR` wraps, paddleocr.py:
    628-694) so that the local, offline model directories are used: `PaddleOCR.__init__` downloads the default
    PP-OCR tar files from paddleocr.bj.bcebos.com when `inference.pdmodel` is absent (ppocr/utils/network.py:
    106-128). `PaddleOCR.ocr` for one ndarray is reproduced (paddleocr.py:744-772). As in flow_grpo, the RGB array is
    passed as is (PaddleOCR treats a 3-channel ndarray as BGR). Paddle 3.0's inference predictor rejects
    `enable_memory_optim()` for PIR (`.json`) models on the CPU ("Not find predictor_id 0 and pass_name
    memory_optimize_pass"); that call, a memory-planning switch with no effect on results, is disabled."""

    def __init__(self, model_root, cpu_threads=8):
        import paddle.inference as pinf
        from paddleocr import paddleocr as po
        from paddleocr.tools.infer.predict_system import TextSystem

        pinf.Config.enable_memory_optim = lambda self, *a, **k: None
        params = po.parse_args(mMain=False)
        params.__dict__.update(use_angle_cls=False, lang="en", use_gpu=False, show_log=False,
                               det_model_dir=str(Path(model_root) / "paddleocr" / "PP-OCRv4_mobile_det"),
                               rec_model_dir=str(Path(model_root) / "paddleocr" / "en_PP-OCRv4_mobile_rec"),
                               rec_char_dict_path=str(Path(po.__file__).parent / "ppocr" / "utils" / "en_dict.txt"),
                               rec_image_shape="3, 48, 320", cpu_threads=cpu_threads)
        self.po = po
        self.system = TextSystem(params)

    def text(self, arr) -> str:
        img, _, _ = self.po.check_img(arr, (255, 255, 255))
        img = self.po.alpha_to_color(img, (255, 255, 255))
        boxes, rec, _ = self.system(img, False, {})
        if not boxes and not rec:
            return ""
        return "".join(r[0] if r[1] > 0 else "" for r in rec)                                 # ocr.py:43

    def __call__(self, arrays, prompts):
        """Rewards as flow_grpo ocr.py:23-63; an OCR exception gives the maximum penalty (:55-58)."""
        out = []
        for arr, prompt in zip(arrays, prompts):
            target = target_text(prompt)
            try:
                text = self.text(arr)
                reward = ocr_accuracy(text, target)
                err = None
            except Exception as e:                                                             # noqa: BLE001
                text, reward, err = None, 0.0, f"{type(e).__name__}: {e}"[:300]
            out.append({"text": text, "reward": reward, "error": err})
        return out


def sampler_doc(num_steps, noise_level, guidance_scale, resolution, max_sequence_length, scheduler_config,
                sigmas, transformer_dtype, encoder_dtype) -> dict:
    """The deployment sampler (08 section 3.6 'scheduler config'), stored in every cut as sampler.json; the weight
    precisions are part of it because they change the images."""
    scheduler_config = {k: (sorted(v) if k == "_use_default_values" else v) for k, v in scheduler_config.items()}
    return {"scheduler": "FlowMatchEulerDiscreteScheduler", "scheduler_config": scheduler_config,
            "num_steps": num_steps, "noise_level": noise_level,
            "step": "flow_grpo sde_step_with_logprob (sde); noise_level 0 = Euler ODE step",
            "sigmas": [round(float(s), 8) for s in sigmas], "guidance_scale": guidance_scale,
            "resolution": resolution, "negative_prompt": "", "max_sequence_length": max_sequence_length,
            "noise": "per sample: torch.Generator('cpu').manual_seed(seed); initial latents, then one draw per step "
                     "when noise_level > 0", "latents_dtype": "float32", "vae_dtype": "float32",
            "transformer_dtype": transformer_dtype, "text_encoder_dtype": encoder_dtype,
            "autocast": "float16" if transformer_dtype == "fp16" else None}

# ---------------------------------------------------------------------------
# END SHARED SAMPLER
# ---------------------------------------------------------------------------


def sampler_code_sha256() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    start = text.index("# BEGIN SHARED SAMPLER")
    end = text.index("# END SHARED SAMPLER")
    return sha256_bytes(text[start:end].encode("utf-8"))


class PerPromptStatTracker:
    """flow_grpo stat_tracking.py:5-36, type 'grpo', cleared after every epoch (train_sd3.py:768)."""

    def __init__(self, global_std: bool):
        self.global_std = global_std

    def update(self, prompts, rewards):
        import numpy as np

        prompts, rewards = np.array(prompts), np.array(rewards, dtype=np.float64)
        adv = np.zeros_like(rewards)
        for p in np.unique(prompts):
            pr = rewards[prompts == p]
            mean = np.mean(pr)
            std = (np.std(rewards) if self.global_std else np.std(pr)) + 1e-4
            adv[prompts == p] = (pr - mean) / std
        return adv


class Ema:
    """flow_grpo ema.py:8-53 (EMAModuleWrapper): decay min((1 + s) / (10 + s), decay), updated only when
    (s + 1) % interval == 0."""

    def __init__(self, params, decay, interval):
        self.ema = [p.detach().clone() for p in params]
        self.decay, self.interval = decay, interval

    def step(self, params, s):
        import torch

        with torch.no_grad():
            if (s + 1) % self.interval == 0:
                k = 1 - min((1 + s) / (10 + s), self.decay)
                for e, p in zip(self.ema, params):
                    e.add_(k * (p.detach() - e))

    def swap(self, params):
        for e, p in zip(self.ema, params):
            tmp = p.data.clone()
            p.data.copy_(e)
            e.copy_(tmp)


# ---------------------------------------------------------------------------
# Pre-step (MM-33)
# ---------------------------------------------------------------------------

def write_env_freeze(out: Path) -> dict:
    pkgs = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            pkgs[name.lower().replace("_", "-")] = dist.version
    lines = [f"{k}=={pkgs[k]}" for k in sorted(pkgs)]
    lines.append(f"# flow_grpo algorithm reference {ALGORITHM_REFERENCE} (ported)")
    (out / "env-freeze.txt").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    lock = HERE / "uv.lock"
    runtime = {
        "python": platform.python_version(), "executable": sys.executable,
        "image_cuda_version": os.environ.get("CUDA_VERSION"),
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
        "script_sha256": sha256_file(Path(__file__).resolve()), "sampler_code_sha256": sampler_code_sha256(),
        "uid": os.getuid(), "gid": os.getgid(), "cpu_count": os.cpu_count(),
        "env_freeze_sha256": sha256_file(out / "env-freeze.txt"),
    }
    write_json(out / "env-runtime.json", runtime)
    return runtime


def prepare_out(out: Path) -> int:
    """As area 3: a non-empty --out is a rerun after a preemption; earlier contents move to `preempted-attempt-<k>/`
    (never deleted). Returns the attempt number."""
    out.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out.iterdir() if not p.name.startswith("preempted-attempt-")]
    previous = sorted(p for p in out.iterdir() if p.name.startswith("preempted-attempt-"))
    attempt = len(previous) + 1
    if existing:
        attempt = len(previous) + 2
        dest = out / f"preempted-attempt-{attempt - 1}"
        dest.mkdir()
        for p in existing:
            p.rename(dest / p.name)
        if (dest / "attempts.jsonl").exists():
            shutil.copyfile(dest / "attempts.jsonl", out / "attempts.jsonl")
    append_jsonl(out / "attempts.jsonl", {"attempt": attempt, "started_at": utc_now(), "argv": sys.argv})
    return attempt


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------

def base_binding(base: Path, verify: bool) -> dict:
    """Base model reference (MM-18, MM-21): repo, 40-hex revision and every file's SHA-256 from MODELS.json (the
    user-downloaded directory, `.cache/` excluded); with `verify`, every file is re-hashed (about 27 GB)."""
    record = json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"][BASE_NAME]
    files = record["files"]
    on_disk = []
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(base).as_posix()
        if rel == ".cache" or rel.startswith(".cache/"):
            continue
        if p.is_symlink():
            raise SystemExit(f"{p} is a symbolic link; the base directory must be plain (MM-20)")
        if p.is_file():
            on_disk.append(rel)
    if on_disk != sorted(files):
        raise SystemExit(f"{base} does not hold exactly the files of {MODELS_JSON} [{BASE_NAME}]")
    if verify:
        for rel, want in files.items():
            if sha256_file(base / rel) != want["sha256"]:
                raise SystemExit(f"{base / rel} differs from its SHA-256 in {MODELS_JSON}")
    rev = record["revision"]
    if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
        raise SystemExit(f"base revision {rev!r} is not a 40-hex commit (MM-18)")
    digests = {k: v["sha256"] for k, v in files.items()}
    return {"repo": record["repo"], "revision": rev, "local_dir": str(base), "files": digests,
            "files_sha256": sha256_bytes(canonical(digests)), "verified": verify}


def read_prompts(prompts_dir: Path) -> tuple[list[dict], Path]:
    path = prompts_dir / "train.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids = [r["prompt_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: prompt ids are not unique")
    return rows, path


# ---------------------------------------------------------------------------
# Cuts
# ---------------------------------------------------------------------------

def canonicalize_safetensors_header(path: Path) -> None:
    """As area 3: rewrite the safetensors header as sorted, compact JSON padded to 8 bytes (tensor bytes and offsets
    unchanged) and sort the LoraConfig's `target_modules`, so the file hash depends on the weights and metadata only."""
    import struct

    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    meta = header.get("__metadata__", {})
    if "lora_adapter_metadata" in meta:
        cfg = json.loads(meta["lora_adapter_metadata"])
        if isinstance(cfg.get("target_modules"), list):
            cfg["target_modules"] = sorted(cfg["target_modules"])
        meta["lora_adapter_metadata"] = json.dumps(cfg, indent=2, sort_keys=True)
    text = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    text += b" " * (-len(text) % 8)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + raw[8 + n:])
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_cut(transformer, out: Path, epoch: int, global_step: int, samples_consumed: int, samples_filtered: int,
             base_ref: dict, sampler: dict, lora_cfg: dict, extra: dict) -> Path:
    """checkpoint-<epoch>/ via .checkpoint-<epoch>.partial/ and an atomic rename; cut.json last (as area 3)."""
    cut = out / f"checkpoint-{epoch}"
    if cut.exists():
        return cut
    tmp = out / f".checkpoint-{epoch}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    transformer.save_lora_adapter(str(tmp), adapter_name="default", safe_serialization=True, weight_name=LORA_WEIGHTS)
    canonicalize_safetensors_header(tmp / LORA_WEIGHTS)
    os.chmod(tmp / LORA_WEIGHTS, 0o644)
    write_json(tmp / "adapter_config.json", {**lora_cfg, "base_model_name_or_path": base_ref["repo"],
                                             "base_model_revision": base_ref["revision"], "format": "peft",
                                             "loader": "SD3Transformer2DModel.load_lora_adapter(dir, prefix=None)",
                                             "weights": "ema" if extra.get("ema") else "raw"})
    write_json(tmp / "sampler.json", sampler)
    write_json(tmp / "base_ref.json", {k: base_ref[k] for k in ("repo", "revision", "files", "files_sha256")})
    write_json(tmp / "trainer_state.json", {
        "epoch": epoch, "global_step": global_step, "samples_consumed": samples_consumed,
        "samples_filtered": samples_filtered,
        "note": "resume-only information; the run does not resume from a cut (a preempted run restarts)"})
    files = {name: sha256_file(tmp / name) for name in CUT_FILES}
    write_json(tmp / CUT_MARKER, {
        "schema": "ptc_mm.flow_grpo_sd35_cut", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "epoch": epoch, "global_step": global_step, "samples_consumed": samples_consumed,
        "samples_filtered": samples_filtered, "files": files, "deployable": list(DEPLOYABLE),
        "created_at": utc_now(), **extra,
    })
    os.replace(tmp, cut)
    return cut


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--profile", choices=sorted(PROFILES), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, help="override the profile's epoch count")
    p.add_argument("--no-verify-base", action="store_true", help="skip re-hashing the base files (host checks)")
    p.add_argument("--train-batch", type=int, help="memory knob: override the profile's train micro-batch")
    p.add_argument("--no-gradient-checkpointing", action="store_true", help="memory knob: faster, more memory")
    p.add_argument("--beta", type=float, help="budget knob: KL coefficient (flow_grpo 0.04; 0 skips the reference "
                                              "forward)")
    p.add_argument("--ema", action="store_true", help="flow_grpo's EMA of the LoRA weights; cuts hold the EMA")
    return p.parse_args(argv)


def resolve_settings(args) -> dict:
    s = {**COMMON, **PROFILES[args.profile]}
    if args.epochs is not None:
        s["epochs"] = args.epochs
    if args.train_batch is not None:
        s["train_batch"] = args.train_batch
    if args.no_gradient_checkpointing:
        s["gradient_checkpointing"] = False
    if args.beta is not None:
        s["beta"] = args.beta
    if args.ema:
        s["ema"] = True
    s["seed"] = args.seed
    s["samples_per_epoch"] = s["prompts_per_epoch"] * s["images_per_prompt"]
    if s["sample_batch"] != s["images_per_prompt"]:
        raise SystemExit("a sampling batch is one prompt's group (k = batch, train_sd3.py:482-489)")
    # flow_grpo: num_batches_per_epoch sampling batches; training splits the kept samples into the same number of
    # micro-batches and accumulates over half of them (config/grpo.py:457-465), so two optimizer steps per epoch.
    s["num_batches_per_epoch"] = s["samples_per_epoch"] // s["sample_batch"]
    if s["num_batches_per_epoch"] % 2:
        raise SystemExit("num_batches_per_epoch must be even (config/grpo.py:458)")
    s["gradient_accumulation_steps"] = s["num_batches_per_epoch"] // 2
    s["num_train_timesteps"] = int(s["num_steps"] * s["timestep_fraction"])                  # train_sd3.py:345
    return s


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.out.resolve()
    attempt = prepare_out(out)
    runtime = write_env_freeze(out)                 # before torch is imported (MM-33)
    s = resolve_settings(args)
    t_start = time.time()
    base_ref = base_binding(args.base.resolve(), verify=not args.no_verify_base)
    prompts, prompts_file = read_prompts(args.prompts)

    import concurrent.futures

    import numpy as np
    import torch
    from peft import LoraConfig

    device = torch.device(args.device)
    dtype = torch.float16 if s["precision"] == "fp16" else torch.float32
    autocast_dtype = torch.float16 if s["precision"] == "fp16" else None
    if s["allow_tf32"] and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    P, G, B, E = s["prompts_per_epoch"], s["images_per_prompt"], s["sample_batch"], s["epochs"]
    plans = {}
    for epoch in range(1, E + 1):
        rng = np.random.Generator(np.random.PCG64(derive_seed("prompts", s["seed"], epoch)))
        plans[epoch] = [prompts[int(c)] for c in rng.choice(len(prompts), size=P, replace=False)]
    mem_log = {"rss_mb_start": rss_mb()}

    # --- text encoders: every prompt of the run once, then freed (T5-XXL ~9.5 GB in fp16) ---
    t0 = time.time()
    run_prompts = [""] + [r["prompt"] for e in plans for r in plans[e]]
    enc_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[s["encoder_precision"]]
    emb = encode_all_prompts(args.base, run_prompts, device, enc_dtype, s["max_sequence_length"], out_dtype=dtype)
    mem_log.update(seconds_encode=round(time.time() - t0, 1), rss_mb_after_encode=rss_mb())
    if device.type == "cuda":
        mem_log["peak_allocated_gb_encode"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
        torch.cuda.reset_peak_memory_stats(device)
    neg_e, neg_p = (x.to(device) for x in emb[""])

    transformer, vae, scheduler = load_transformer_vae(args.base, device, dtype)
    torch.manual_seed(derive_seed("torch", s["seed"]))   # peft gaussian init draws from the global RNG
    lora_config = LoraConfig(**LORA)
    transformer.add_adapter(lora_config)
    for prm in transformer.parameters():            # LoRA parameters in fp32 (diffusers cast_training_params)
        if prm.requires_grad:
            prm.data = prm.data.float()
    if s["gradient_checkpointing"]:
        transformer.enable_gradient_checkpointing()
    params = [prm for prm in transformer.parameters() if prm.requires_grad]
    lora_cfg = json.loads(json.dumps(lora_config.to_dict(), default=lambda o: sorted(o) if isinstance(o, set)
                                     else str(o)))
    optimizer = torch.optim.AdamW(params, lr=s["learning_rate"], betas=(s["adam_beta1"], s["adam_beta2"]),
                                  weight_decay=s["adam_weight_decay"], eps=s["adam_epsilon"])
    scaler = torch.amp.GradScaler(device.type, enabled=(s["precision"] == "fp16"))
    ema = Ema(params, s["ema_decay"], s["ema_update_step_interval"]) if s["ema"] else None
    _, deploy_sigmas = schedule(scheduler, s["deploy_num_steps"])
    sampler = sampler_doc(s["deploy_num_steps"], s["deploy_noise_level"], s["guidance_scale"], s["resolution"],
                          s["max_sequence_length"], dict(scheduler.config), deploy_sigmas, s["precision"],
                          s["encoder_precision"])
    train_timesteps, train_sigmas = schedule(scheduler, s["num_steps"])
    train_sampler = sampler_doc(s["num_steps"], s["noise_level"], s["guidance_scale"], s["resolution"],
                                s["max_sequence_length"], dict(scheduler.config), train_sigmas, s["precision"],
                                s["encoder_precision"])
    reward_a = PaddleOcrA(OCR_MODELS, cpu_threads=s["ocr_cpu_threads"])
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)   # rewards overlap sampling (train_sd3.py:562)
    mem_log["rss_mb_after_load"] = rss_mb()

    run_args = {
        "schema": "ptc_mm.flow_grpo_sd35_run_args", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "profile": args.profile, "attempt": attempt, "settings": s, "lora": lora_cfg, "seed": s["seed"],
        "seed_derivation": ("derive_seed(*parts) = first 8 bytes of SHA-256(':'.join(parts)) >> 1. Epoch e prompts: "
                            "numpy Generator(PCG64(derive_seed('prompts', seed, e))).choice(n_train, P, "
                            "replace=False); sample i of epoch e: noise seed derive_seed('sample', seed, e, i) for a "
                            "CPU torch.Generator (initial latents, then one draw per denoising step); zero-advantage "
                            "padding: torch.Generator('cpu').manual_seed(derive_seed('mask', seed, e)); training "
                            "order: torch.Generator('cpu').manual_seed(derive_seed('perm', seed, e, inner)); LoRA "
                            "init: torch global RNG seeded with derive_seed('torch', seed)"),
        "base_model": base_ref["local_dir"],
        "base_ref": {k: base_ref[k] for k in ("repo", "revision", "files_sha256", "verified")},
        "train_file": str(prompts_file), "train_file_sha256": sha256_file(prompts_file),
        "n_train_prompts": len(prompts),
        "sampler": sampler, "sampler_sha256": sha256_bytes(canonical(sampler)), "training_sampler": train_sampler,
        "reward": {"name": "ocr", "engine": "PaddleOCR (paddleocr 2.10.0 TextSystem, paddlepaddle CPU)",
                   "models": json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"]
                   .get("paddleocr/en_PP-OCRv4_mobile_rec", {}).get("revision"),
                   "formula": "flow_grpo ocr.py:46-60 on the first quoted span of the prompt"},
        "consumption_log": "samples.jsonl", "epoch_plan": "epoch_plan.jsonl", "step_log": "train_steps.jsonl",
        "consumption_semantics": ("samples.jsonl: one line per generated sample at sampling time; every sample of "
                                  "epoch e is either consumed by exactly one optimizer step of epoch e per inner "
                                  "epoch (train_steps.jsonl) or filtered (epoch_plan.jsonl `filtered`: all its "
                                  "advantages are 0, train_sd3.py:783-823)"),
        "device": str(device), "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "sampler_code_sha256": runtime["sampler_code_sha256"], "script_sha256": runtime["script_sha256"],
        "memory_at_start": mem_log, "created_at": utc_now(),
    }
    write_json(out / "run_args.json", run_args)

    global_step = 0
    samples_consumed = 0
    samples_filtered = 0
    tracker = PerPromptStatTracker(s["global_std"])
    T = s["num_train_timesteps"]
    nb = s["num_batches_per_epoch"]
    for epoch in range(1, E + 1):
        t_ep = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # ---------------- sampling (train_sd3.py:608-704) ----------------
        transformer.eval()
        plan = [(row, g) for row in plans[epoch] for g in range(G)]
        rec = {"latents": [], "next_latents": [], "log_probs": []}
        futures, sample_ids, prompt_ids, seeds_all, rows_all, bright = [], [], [], [], [], []
        for b0 in range(0, len(plan), B):
            batch = plan[b0:b0 + B]
            idx = list(range(b0, b0 + len(batch)))
            seeds = [derive_seed("sample", s["seed"], epoch, i) for i in idx]
            e = torch.stack([emb[r["prompt"]][0] for r, _ in batch]).to(device)
            pe = torch.stack([emb[r["prompt"]][1] for r, _ in batch]).to(device)
            imgs, lat, logp, _ = sample_batch(transformer, vae, scheduler, e, pe, neg_e.expand(len(batch), -1, -1),
                                              neg_p.expand(len(batch), -1), seeds, s["num_steps"], s["noise_level"],
                                              s["guidance_scale"], s["resolution"], device, autocast_dtype)
            rec["latents"].append(lat[:, :-1])
            rec["next_latents"].append(lat[:, 1:])
            rec["log_probs"].append(logp)
            arrays = [to_uint8(imgs[k]) for k in range(len(batch))]
            bright.extend(float(a.mean()) / 255 for a in arrays)
            futures.append(pool.submit(reward_a, arrays, [r["prompt"] for r, _ in batch]))
            for k, (row, g) in enumerate(batch):
                sample_ids.append(sha256_bytes(canonical(["flow-grpo-sd35", s["seed"], attempt, epoch, idx[k],
                                                          row["prompt_id"], seeds[k], g])))
                prompt_ids.append(row["prompt_id"])
                seeds_all.append(seeds[k])
                rows_all.append((row, g, idx[k]))
        results = [r for f in futures for r in f.result()]
        rewards = [r["reward"] for r in results]
        if s["reward"] == "ocr+brightness_sanity":
            rewards = [r + 0.01 * b for r, b in zip(rewards, bright)]
        elif s["reward"] != "ocr":
            raise SystemExit(f"unknown reward {s['reward']}")
        t_sample = time.time() - t_ep
        first_step = global_step + 1
        for k, (row, g, i) in enumerate(rows_all):
            append_jsonl(out / "samples.jsonl", {
                "epoch": epoch, "sample_index": i, "sample_id": sample_ids[k], "prompt_id": row["prompt_id"],
                "prompt": row["prompt"], "target": target_text(row["prompt"]), "member": g, "seed": seeds_all[k],
                "ocr_text": results[k]["text"], "ocr_error": results[k]["error"], "ocr_acc": results[k]["reward"],
                "reward": rewards[k],
                "attempt": attempt, "utc": utc_now()})
        # ---------------- advantages and the zero-advantage filter (train_sd3.py:748-823) ----------------
        adv = torch.as_tensor(tracker.update(prompt_ids, rewards), dtype=torch.float32)
        n = len(plan)
        mask = adv != 0                                  # (advantages over timesteps all equal per sample)
        true_count = int(mask.sum())
        reincluded = []
        if true_count % nb != 0:
            false_idx = torch.where(~mask)[0]
            need = nb - (true_count % nb)
            if len(false_idx) >= need:
                gm = torch.Generator("cpu").manual_seed(derive_seed("mask", s["seed"], epoch))
                pick = false_idx[torch.randperm(len(false_idx), generator=gm)[:need]]
                mask[pick] = True
                reincluded = sorted(int(x) for x in pick)
        kept = torch.where(mask)[0]
        n_kept = len(kept)
        if n_kept % nb:                                  # cannot happen: the epoch's sample count is a multiple of nb
            raise SystemExit(f"epoch {epoch}: {n_kept} kept samples are not a multiple of {nb}")
        filtered_ids = [sample_ids[i] for i in range(n) if not bool(mask[i])]
        n_steps_epoch = 2 * s["num_inner_epochs"] if n_kept else 0
        append_jsonl(out / "epoch_plan.jsonl", {
            "epoch": epoch, "kept": [sample_ids[i] for i in kept.tolist()], "filtered": filtered_ids,
            "reincluded": [sample_ids[i] for i in reincluded], "filter_reason": "zero_advantage",
            "steps": [first_step, global_step + n_steps_epoch] if n_steps_epoch else [], "utc": utc_now()})
        lat_all = torch.cat(rec["latents"])
        next_all = torch.cat(rec["next_latents"])
        logp_all = torch.cat(rec["log_probs"])
        del rec
        # ---------------- training (train_sd3.py:831-962) ----------------
        transformer.train()
        t_tr = time.time()
        _, sigmas = schedule(scheduler, s["num_steps"])
        if n_kept and n_kept % nb == 0:
            m = n_kept // nb                                # flow_grpo micro-batch size (train_sd3.py:838-841)
            for inner in range(s["num_inner_epochs"]):
                gp = torch.Generator("cpu").manual_seed(derive_seed("perm", s["seed"], epoch, inner))
                order = kept[torch.randperm(n_kept, generator=gp)].tolist()
                virtual = [order[v * m:(v + 1) * m] for v in range(nb)]
                info, consumed = {"policy_loss": [], "kl_loss": [], "approx_kl": [], "clipfrac": []}, []
                optimizer.zero_grad(set_to_none=True)
                for v, vidx in enumerate(virtual):
                    for c0 in range(0, m, s["train_batch"]):
                        bidx = vidx[c0:c0 + s["train_batch"]]
                        bt = torch.tensor(bidx)
                        e = torch.stack([emb[plan[i][0]["prompt"]][0] for i in bidx]).to(device)
                        pe = torch.stack([emb[plan[i][0]["prompt"]][1] for i in bidx]).to(device)
                        ne, npool = neg_e.expand(len(bidx), -1, -1), neg_p.expand(len(bidx), -1)
                        a = torch.clamp(adv[bt], -s["adv_clip_max"], s["adv_clip_max"]).to(device)
                        for j in range(T):
                            lat_j = lat_all[bt, j].to(device)
                            nxt_j = next_all[bt, j].to(device)
                            old_lp = logp_all[bt, j].to(device)
                            tj = torch.full((len(bidx),), float(train_timesteps[j]), device=device)
                            sj = torch.full((len(bidx),), j, dtype=torch.long)
                            vel = predict_velocity(transformer, lat_j, tj, e, pe, ne, npool, s["guidance_scale"],
                                                   autocast_dtype)
                            _, lp, mean, std = sde_step_with_logprob(sigmas, sj, vel, lat_j, s["noise_level"],
                                                                     prev_sample=nxt_j)
                            ratio = torch.exp(lp - old_lp)
                            per = torch.maximum(-a * ratio,
                                                -a * torch.clamp(ratio, 1.0 - s["clip_range"], 1.0 + s["clip_range"]))
                            kl = torch.zeros_like(per)
                            if s["beta"] > 0:
                                with torch.no_grad():
                                    transformer.disable_adapters()
                                    vref = predict_velocity(transformer, lat_j, tj, e, pe, ne, npool,
                                                            s["guidance_scale"], autocast_dtype)
                                    transformer.enable_adapters()
                                    _, _, mean_ref, _ = sde_step_with_logprob(sigmas, sj, vref, lat_j,
                                                                              s["noise_level"], prev_sample=nxt_j)
                                kl = ((mean - mean_ref) ** 2).mean(dim=(1, 2, 3)) / (2 * std.view(-1) ** 2)
                            # flow_grpo: loss = mean over the micro-batch of m samples, averaged over the
                            # accumulation window of (gradient_accumulation_steps x T) backward calls.
                            loss = (per + s["beta"] * kl).sum() / (m * s["gradient_accumulation_steps"] * T)
                            scaler.scale(loss).backward()
                            info["policy_loss"].append(float(per.mean()))
                            info["kl_loss"].append(float(kl.mean()))
                            info["approx_kl"].append(float(0.5 * torch.mean((lp - old_lp) ** 2)))
                            info["clipfrac"].append(float(torch.mean((torch.abs(ratio - 1.0) > s["clip_range"])
                                                                     .float())))
                        consumed.extend(sample_ids[i] for i in bidx)
                    if (v + 1) % s["gradient_accumulation_steps"] == 0:
                        global_step += 1
                        append_jsonl(out / "train_steps.jsonl", {"step": global_step, "epoch": epoch, "inner": inner,
                                                                 "sample_ids": consumed, "utc": utc_now()})
                        scaler.unscale_(optimizer)
                        gnorm = float(torch.nn.utils.clip_grad_norm_(params, s["max_grad_norm"]))
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        append_jsonl(out / "train_log.jsonl", {
                            "step": global_step, "epoch": epoch, "grad_norm": gnorm, "scale": float(scaler.get_scale()),
                            **{k: float(np.mean(x)) for k, x in info.items()}})
                        info = {k: [] for k in info}
                        consumed = []
                    if ema is not None:                       # after every flow_grpo micro-batch (train_sd3.py:959)
                        ema.step(params, global_step)
            samples_consumed += n_kept
        samples_filtered += n - n_kept
        t_train = time.time() - t_tr
        r = np.array(rewards)
        mem = {"epoch": epoch, "seconds_sampling": round(t_sample, 2), "seconds_training": round(t_train, 2),
               "seconds_epoch": round(time.time() - t_ep, 2), "global_step": global_step,
               "reward_mean": float(r.mean()), "reward_std": float(r.std()), "kept": n_kept,
               "zero_std_groups": int(sum(np.std(r[[k for k in range(n) if prompt_ids[k] == p]]) == 0
                                          for p in dict.fromkeys(prompt_ids))),
               "max_rss_mb": rss_mb(), "utc": utc_now()}
        if device.type == "cuda":
            mem["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
            mem["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
        append_jsonl(out / "memory.jsonl", mem)
        print(json.dumps(mem, sort_keys=True), flush=True)
        if epoch % s["save_every"] == 0 or epoch == E:
            if ema is not None:
                ema.swap(params)
            save_cut(transformer, out, epoch, global_step, samples_consumed, samples_filtered, base_ref, sampler,
                     lora_cfg, {"script_sha256": runtime["script_sha256"],
                                "sampler_code_sha256": runtime["sampler_code_sha256"], "seed": s["seed"],
                                "profile": args.profile, "ema": bool(ema)})
            if ema is not None:
                ema.swap(params)
    pool.shutdown()
    append_jsonl(out / "attempts.jsonl", {"attempt": attempt, "finished_at": utc_now(), "epochs": E,
                                          "global_step": global_step, "samples_consumed": samples_consumed,
                                          "samples_filtered": samples_filtered,
                                          "wall_seconds": round(time.time() - t_start, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
