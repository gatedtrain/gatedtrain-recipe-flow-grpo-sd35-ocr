#!/usr/bin/env python
"""Per-item OCR evaluation of an SD3.5 Medium + LoRA cut (or the base alone) for prototype area "image generation,
SD3.5" (MULTIMODAL-SPEC section 6 row 2b; prototype, unpromotable).

An item is one registered prompt (design 05 section 3.3) from flow_grpo's OCR test split (dataset/ocr/test.txt at
commit 879042cf5707f8b90daa98d147d7deac2317c5da), each asking for one quoted text to be rendered. Per item and
repeat r: one image from the deployment sampler (the cut's sampler.json: flow-matching Euler ODE, 10 steps, CFG 4.5,
512 px) with noise seed derive_seed('eval', eval_seeds[r], item), the same for every arm (common random numbers).
Every image is kept as PNG with its SHA-256; every metric is computed from the saved uint8 image.

Metrics (README.md "Reward and deciding metric"; tiers are proposals):
  ocr_b_acc   deciding (proposed T3, construct-aligned, R15-1), higher is better, [0, 1]: approximate-substring OCR
              accuracy (`ocr_substring_accuracy`: spaces removed, lower case; 1 - min over substrings s of the
              recognized text of lev(s, target) / len(target)) with OCR engine B = docTR (db_resnet50 detector,
              crnn_vgg16_bn recognizer), a detector and recognizer other than the reward's. Formula chosen after the
              smoke baseline, before registration (README.md "Decisions taken", smoke finding of
              2026-11-14): docTR also reads background texture, and the whole-text edit distance clipped most items to 0
  ocr_a_acc   supporting, T0 (the training reward: flow_grpo's formula, ocr.py:46-60, with OCR engine A =
              PaddleOCR PP-OCRv4), [0, 1]
Descriptive sub-metrics only (never deciding), finite numbers only (MM-1): both engines' scores (an engine that
raised is left out), nonfinite. The target and the recognized texts are in scores.jsonl.

Ports from yifan123/flow_grpo at commit 879042cf5707f8b90daa98d147d7deac2317c5da (MIT License, Copyright (c) 2025
Jie Liu), cited by file:line of that commit where they are used: the shared sampler block (the SDE step with
log-probability, the sampling loop with classifier-free guidance, the prompt encoding and the OCR reward,
byte-identical in train_flow_grpo.py), the OCR accuracy formula (ocr.py) and the OCR test split. The edit distance
comes from rapidfuzz (MIT); flow_grpo and the published 2026-11-14 run used python-Levenshtein, which returns the same
values (RESULT.md).

Two modes:
  python eval_images.py --make-items [--n-items 400] [--split-seed S]
      writes items/items.jsonl (registered items) and prompts/train.jsonl (training shard, disjoint by prompt and by
      target text) with their meta files, from data-source/ocr/{train,test}.txt
  python eval_images.py (--checkpoint CUT | --base-only) --out DIR --eval-seeds 11 [--items DIR] [--limit N]
                        [--model-hash H] [--base-model-hash H] [--item-set-hash H] [--device cuda]
      writes DIR/images/<item>/r<repeat>.png, DIR/scores.jsonl, and one MM-1/MM-2 arm per metric:
      DIR/arm-<metric>/{items.jsonl, eval_meta.json, images/ (hard links)}
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "models" / "stable-diffusion-3.5-medium"
OCR_MODELS = HERE / "ocr-models"
MODELS_JSON = HERE / "MODELS.json"
DATA_SOURCE = HERE / "data-source" / "ocr"
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"
ADAPTER = "flow_grpo_sd35"
SCORER_VERSION = "eval_images@0.1"
ITEM_NAMESPACE = 0x5D35A5        # split permutation: SeedSequence([ITEM_NAMESPACE, split_seed])
SAMPLER_DEFAULT = {"num_steps": 10, "noise_level": 0.0, "guidance_scale": 4.5, "resolution": 512,
                   "max_sequence_length": 128}
DOCTR = {"det_arch": "db_resnet50", "reco_arch": "crnn_vgg16_bn", "assume_straight_pages": True,
         "preserve_aspect_ratio": True, "symmetric_pad": True, "join": "words in reading order, no separator"}
METRICS = {
    # metric -> evaluator id, direction, range, tier (proposal), role
    "ocr_b_acc": ("t2i.sd35_ocr_doctr_items@0.1", "higher_is_better", [0, 1], "T3 (proposed, construct-aligned)",
                  "deciding"),
    "ocr_a_acc": ("t2i.sd35_ocr_paddle_items@0.1", "higher_is_better", [0, 1], "T0", "supporting"),
}
ERROR_CLASSES = ("scorer_error",)


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
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


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


def substring_edit_distance(text: str, pattern: str) -> int:
    """min over substrings s of `text` (the empty one included) of the Levenshtein distance lev(s, `pattern`): Sellers'
    approximate-substring dynamic programme, O(len(text) x len(pattern)) time, O(len(text)) memory. Row i is the
    pattern prefix of length i; row 0 is all zeros (a match may start anywhere in `text`) and the answer is the
    minimum of the last row (it may end anywhere)."""
    prev = [0] * (len(text) + 1)
    for i, pc in enumerate(pattern, start=1):
        cur = [i] + [0] * len(text)
        for j, tc in enumerate(text, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (pc != tc))
        prev = cur
    return min(prev)


def ocr_substring_accuracy(recognized: str, target: str) -> float:
    """The deciding metric `ocr_b_acc` (README.md "Decisions taken", smoke finding of
    2026-11-14; chosen after the smoke baseline, before registration): spaces removed and lower case on both sides,
    then 1 - min over substrings s of the recognized text of lev(s, target) / len(target), in [0, 1]; an empty
    recognized text scores 0. Unlike flow_grpo's formula (`ocr_accuracy`, the reward and `ocr_a_acc`), text read
    from the background around the target does not lower the score. An empty normalized target raises (an error,
    never a score)."""
    recognized = recognized.replace(" ", "").lower()
    target = target.replace(" ", "").lower()
    if not target:
        raise ValueError("the target text is empty after normalization")
    if not recognized:
        return 0.0
    return 1 - substring_edit_distance(recognized, target) / len(target)


# ---------------------------------------------------------------------------
# Items and the training shard
# ---------------------------------------------------------------------------

def norm_target(prompt: str) -> str:
    return target_text(prompt).replace(" ", "").lower()


def make_items(n_items: int, split_seed: int) -> int:
    import numpy as np

    for t in (HERE / "prompts", HERE / "items"):
        if t.exists() and any(t.iterdir()):
            raise SystemExit(f"{t} exists and is not empty")
        t.mkdir(parents=True, exist_ok=True)
    train_lines = (DATA_SOURCE / "train.txt").read_text(encoding="utf-8").splitlines()
    test_lines = (DATA_SOURCE / "test.txt").read_text(encoding="utf-8").splitlines()
    train_first = {}
    for i, ln in enumerate(train_lines):
        if ln.strip():
            train_first.setdefault(ln.strip(), i)
    test = [(i, ln.strip()) for i, ln in enumerate(test_lines) if ln.strip()]
    candidates = [(i, p) for i, p in test if p not in train_first]          # drop exact train/test overlap (1)
    rng = np.random.default_rng(np.random.SeedSequence([ITEM_NAMESPACE, split_seed]))
    perm = [int(x) for x in rng.permutation(len(candidates))]
    chosen = sorted(candidates[k] for k in perm[:n_items])
    items = [{"item": f"ocr-test-{i:04d}", "prompt": p, "target": target_text(p), "source_line": i}
             for i, p in chosen]
    item_prompts = {r["prompt"] for r in items}
    item_targets = {norm_target(r["prompt"]) for r in items}
    shard, dropped_target = [], 0
    for p, i in sorted(train_first.items(), key=lambda kv: kv[1]):
        if p in item_prompts:
            continue
        if norm_target(p) in item_targets:                                    # disjoint by target text too
            dropped_target += 1
            continue
        shard.append({"prompt_id": f"ocr-train-{i:05d}", "prompt": p, "target": target_text(p), "source_line": i})
    if {r["prompt"] for r in shard} & item_prompts or {norm_target(r["prompt"]) for r in shard} & item_targets:
        raise SystemExit("items and training shard overlap")
    write_jsonl(HERE / "items" / "items.jsonl", items)
    write_jsonl(HERE / "prompts" / "train.jsonl", shard)
    source = {"files": {f"data-source/ocr/{n}": sha256_file(DATA_SOURCE / n) for n in ("train.txt", "test.txt")},
              "origin": "yifan123/flow_grpo@879042cf5707f8b90daa98d147d7deac2317c5da dataset/ocr/{train,test}.txt "
                        "(MIT)",
              "target_rule": "flow_grpo ocr.py:32: the first double-quoted span of the prompt"}
    split = {"n_test": len(test), "n_test_after_exact_overlap": len(candidates), "n_items": len(items),
             "n_train_lines": len([ln for ln in train_lines if ln.strip()]), "n_train_unique": len(train_first),
             "n_train_dropped_same_target": dropped_target, "n_train_shard": len(shard), "split_seed": split_seed,
             "generator": f"numpy default_rng(SeedSequence([{ITEM_NAMESPACE:#x}, split_seed])).permutation over the "
                          "test prompts that do not occur in train.txt; the first n_items are the items. Training "
                          "shard: the unique train.txt prompts (first occurrence) without any item prompt and without "
                          "any prompt whose target (spaces removed, lower case) equals an item's target",
             "numpy": np.__version__, "exact_overlap": 0, "target_overlap": 0}
    write_json(HERE / "items" / "item_set_meta.json", {
        "schema": "ptc_mm.sd35_ocr_item_set", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "source": source, "split": split, "items_sha256": sha256_file(HERE / "items" / "items.jsonl"),
        "train_sha256": sha256_file(HERE / "prompts" / "train.jsonl")})
    write_json(HERE / "prompts" / "prompts_meta.json", {
        "schema": "ptc_mm.sd35_ocr_train_prompts", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "source": source, "split": split, "train_sha256": sha256_file(HERE / "prompts" / "train.jsonl")})
    print(json.dumps(split, indent=2))
    return 0


def read_items(items_dir: Path) -> list[dict]:
    rows = [json.loads(ln) for ln in (items_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    ids = [r["item"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("item ids are not unique")
    return rows


# ---------------------------------------------------------------------------
# OCR engine B
# ---------------------------------------------------------------------------

class DoctrB:
    """OCR engine B: docTR (python-doctr 1.1.0) `ocr_predictor` with db_resnet50 + crnn_vgg16_bn, weights from the
    mindee/doctr GitHub release assets converted to safetensors (fetch_models.py); RGB uint8 input."""

    def __init__(self, device):
        import torch
        from doctr.models import crnn_vgg16_bn, db_resnet50, ocr_predictor
        from safetensors.torch import load_file

        det = db_resnet50(pretrained=False, pretrained_backbone=False)
        det.load_state_dict(load_file(str(OCR_MODELS / "doctr" / "db_resnet50.safetensors")))
        rec = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
        rec.load_state_dict(load_file(str(OCR_MODELS / "doctr" / "crnn_vgg16_bn.safetensors")))
        self.torch = torch
        self.predictor = ocr_predictor(det_arch=det, reco_arch=rec, pretrained=False,
                                       assume_straight_pages=DOCTR["assume_straight_pages"],
                                       preserve_aspect_ratio=DOCTR["preserve_aspect_ratio"],
                                       symmetric_pad=DOCTR["symmetric_pad"]).to(device).eval()

    def texts(self, arrays):
        with self.torch.no_grad():
            doc = self.predictor(list(arrays))
        return ["".join(w.value for b in page.blocks for ln in b.lines for w in ln.words) for page in doc.pages]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def base_record() -> dict:
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"]["stable-diffusion-3.5-medium"]


def check_cut(cut: Path) -> dict:
    marker = json.loads((cut / "cut.json").read_text(encoding="utf-8"))
    for name, want in marker["files"].items():
        if sha256_file(cut / name) != want:
            raise SystemExit(f"{cut / name} differs from its SHA-256 in cut.json")
    if marker.get("sampler_code_sha256") != sampler_code_sha256():
        raise SystemExit("the cut was trained with another sampler code than this evaluator's")
    base_ref = json.loads((cut / "base_ref.json").read_text(encoding="utf-8"))
    record = base_record()
    if base_ref["revision"] != record["revision"] or base_ref["files"] != {k: v["sha256"] for k, v in
                                                                           record["files"].items()}:
        raise SystemExit("the cut names another base model than models/stable-diffusion-3.5-medium")
    return marker


def resolve_sampler(args, cut: Path | None, scheduler) -> dict:
    want = {k: getattr(args, k) for k in SAMPLER_DEFAULT if getattr(args, k) is not None}
    if cut is not None:
        doc = json.loads((cut / "sampler.json").read_text(encoding="utf-8"))
        for k, v in want.items():
            if doc[k] != v:
                raise SystemExit(f"--{k.replace('_', '-')} {v} differs from the cut's sampler.json ({doc[k]})")
        _, sig = schedule(scheduler, doc["num_steps"])
        if doc != sampler_doc(doc["num_steps"], doc["noise_level"], doc["guidance_scale"], doc["resolution"],
                              doc["max_sequence_length"], dict(scheduler.config), sig, args.dtype,
                              args.encoder_dtype or args.dtype):
            raise SystemExit("the cut's sampler.json is not this evaluator's sampler")
        return doc
    s = {**SAMPLER_DEFAULT, **want}
    _, sig = schedule(scheduler, s["num_steps"])
    return sampler_doc(s["num_steps"], s["noise_level"], s["guidance_scale"], s["resolution"],
                       s["max_sequence_length"], dict(scheduler.config), sig, args.dtype, args.encoder_dtype or args.dtype)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def evaluate(args) -> int:
    import numpy as np
    import torch
    from PIL import Image

    t0 = time.time()
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    enc_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.encoder_dtype or
                                                                                         args.dtype]
    autocast_dtype = torch.float16 if args.dtype == "fp16" else None
    cut = args.checkpoint.resolve() if args.checkpoint else None
    marker = check_cut(cut) if cut else None
    items = read_items(args.items)
    if args.limit:
        items = items[: args.limit]
    seeds = [int(x) for x in args.eval_seeds.split(",")]
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    rss = {"start": rss_mb()}

    max_len = args.max_sequence_length or SAMPLER_DEFAULT["max_sequence_length"]
    emb = encode_all_prompts(args.base, [""] + [it["prompt"] for it in items], device, enc_dtype, max_len,
                             out_dtype=dtype)
    rss["after_encode"] = rss_mb()
    transformer, vae, scheduler = load_transformer_vae(args.base, device, dtype)
    if cut is not None:
        transformer.load_lora_adapter(str(cut), prefix=None, weight_name=LORA_WEIGHTS, adapter_name="flow_grpo",
                                      local_files_only=True)
    transformer.eval()
    sampler = resolve_sampler(args, cut, scheduler)
    engine_a = PaddleOcrA(OCR_MODELS, cpu_threads=args.ocr_cpu_threads)
    engine_b = DoctrB(device)
    rss["after_load"] = rss_mb()
    neg_e, neg_p = (x.to(device) for x in emb[""])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    jobs = [(r, it) for r in range(len(seeds)) for it in items]
    scores = []
    t_gen = t_a = t_b = 0.0
    for b0 in range(0, len(jobs), args.batch_size):
        batch = jobs[b0:b0 + args.batch_size]
        noise_seeds = [derive_seed("eval", seeds[r], it["item"]) for r, it in batch]
        e = torch.stack([emb[it["prompt"]][0] for _, it in batch]).to(device)
        pe = torch.stack([emb[it["prompt"]][1] for _, it in batch]).to(device)
        tg = time.time()
        imgs, _, _, _ = sample_batch(transformer, vae, scheduler, e, pe, neg_e.expand(len(batch), -1, -1),
                                     neg_p.expand(len(batch), -1), noise_seeds, sampler["num_steps"],
                                     sampler["noise_level"], sampler["guidance_scale"], sampler["resolution"],
                                     device, autocast_dtype, keep_chain=False)
        t_gen += time.time() - tg
        rows, saved = [], []
        for k, (r, it) in enumerate(batch):
            img = imgs[k]
            nonfinite = not bool(torch.isfinite(img).all())
            if nonfinite:
                img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            rel = f"images/{it['item']}/r{r}.png"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(to_uint8(img)).save(out / rel, format="PNG")
            arr = np.asarray(Image.open(out / rel).convert("RGB"))
            saved.append(arr)
            rows.append({"item": it["item"], "repeat": r, "seed": seeds[r], "noise_seed": noise_seeds[k],
                         "target": it["target"], "output_ref": {"path": rel, "sha256": sha256_file(out / rel)},
                         "nonfinite": nonfinite, "errors": {}})
        ta = time.time()
        for k, row in enumerate(rows):
            try:
                row["ocr_a_text"] = engine_a.text(saved[k])
                row["ocr_a_acc"] = ocr_accuracy(row["ocr_a_text"], row["target"])
            except Exception as ex:                                  # scorer failure: an error record (MM-4)
                row["ocr_a_text"] = row["ocr_a_acc"] = None
                row["errors"]["ocr_a_acc"] = {"class": "scorer_error", "message": f"{type(ex).__name__}: {ex}"[:300]}
        t_a += time.time() - ta
        tb = time.time()
        try:
            texts_b = engine_b.texts(saved)
        except Exception as ex:
            texts_b = None
            for row in rows:
                row["errors"]["ocr_b_acc"] = {"class": "scorer_error", "message": f"{type(ex).__name__}: {ex}"[:300]}
        for k, row in enumerate(rows):
            row["ocr_b_text"] = texts_b[k] if texts_b else None
            row["ocr_b_acc"] = None
            if texts_b:
                try:
                    row["ocr_b_acc"] = ocr_substring_accuracy(texts_b[k], row["target"])
                except Exception as ex:                              # scorer failure: an error record (MM-4)
                    row["errors"]["ocr_b_acc"] = {"class": "scorer_error",
                                                  "message": f"{type(ex).__name__}: {ex}"[:300]}
        t_b += time.time() - tb
        scores.extend(rows)
        print(f"{len(scores)}/{len(jobs)} images, {time.time() - t0:.0f} s", flush=True)
    wall = time.time() - t0
    write_jsonl(out / "scores.jsonl", scores)

    code_sha = sha256_file(Path(__file__).resolve())
    sampler_hash = sha256_bytes(canonical(sampler))
    models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"]
    versions = {k: importlib.metadata.version(k) for k in ("paddlepaddle", "paddleocr", "python-doctr", "rapidfuzz",
                                                           "pillow")}
    scorers = {
        "ocr_b_acc": {"name": "ocr_accuracy_doctr",
                      "version": f"{SCORER_VERSION} engine docTR db_resnet50+crnn_vgg16_bn python-doctr {versions['python-doctr']} "
                                 f"{json.dumps(DOCTR, sort_keys=True)} approximate-substring edit distance (Sellers DP), "
                                 "spaces removed, lower case",
                      "weights_sha256": sha256_bytes(canonical({k: v["sha256"] for k, v in
                                                                models["doctr"]["files"].items()})),
                      "code_sha256": code_sha},
        "ocr_a_acc": {"name": "ocr_accuracy_paddleocr",
                      "version": f"{SCORER_VERSION} engine PaddleOCR PP-OCRv4 paddleocr {versions['paddleocr']} paddlepaddle "
                                 f"{versions['paddlepaddle']} PP-OCRv4_mobile_det@"
                                 f"{models['paddleocr/PP-OCRv4_mobile_det']['revision']} en_PP-OCRv4_mobile_rec@"
                                 f"{models['paddleocr/en_PP-OCRv4_mobile_rec']['revision']}",
                      "weights_sha256": sha256_bytes(canonical({**{f"det/{k}": v["sha256"] for k, v in models[
                          "paddleocr/PP-OCRv4_mobile_det"]["files"].items()}, **{f"rec/{k}": v["sha256"] for k, v in
                          models["paddleocr/en_PP-OCRv4_mobile_rec"]["files"].items()}})),
                      "code_sha256": code_sha},
    }
    for metric, (evaluator_id, direction, rng, tier, role) in METRICS.items():
        arm = out / f"arm-{metric}"
        arm.mkdir()
        lines = []
        for row in scores:
            link_or_copy(out / row["output_ref"]["path"], arm / row["output_ref"]["path"])
            err = row["errors"].get(metric)
            # MM-1: sub_metrics hold finite numbers only. The target and both recognized texts stay in scores.jsonl;
            # an engine that raised has no score, so its key is left out (its error is the arm line's `error` in that
            # engine's own arm), never written as null.
            sub = {k: row[k] for k in ("ocr_a_acc", "ocr_b_acc") if row[k] is not None}
            sub["nonfinite"] = int(row["nonfinite"])
            lines.append({"item": row["item"], "repeat": row["repeat"], "seed": row["seed"],
                          "score": None if err else row[metric], "error": err, "sub_metrics": sub,
                          "output_ref": row["output_ref"]})
        write_jsonl(arm / "items.jsonl", lines)
        write_json(arm / "eval_meta.json", {
            "schema": "gatedtrain.item_eval", "schema_version": "1.0",
            "evaluator_id": evaluator_id, "adapter": ADAPTER, "metric": metric, "direction": direction,
            "score_range": rng,
            "checkpoint": {"model_hash": args.model_hash, "base_model_hash": args.base_model_hash},
            "normalizer_sha256": None, "env_version_sha256": None,
            "item_set_hash": args.item_set_hash, "eval_seeds": seeds,
            "sampler": sampler, "sampler_hash": sampler_hash, "scorer": scorers[metric],
            "framework": {"name": "diffusers", "version": importlib.metadata.version("diffusers")},
            "created_at": utc_now()})

    def stats(vals):
        vals = [v for v in vals if v is not None]
        n = len(vals)
        if not n:
            return {"n": 0}
        mean = sum(vals) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None
        return {"n": n, "mean": mean, "sd": sd, "min": min(vals), "max": max(vals),
                "share_at_0": sum(v == 0 for v in vals) / n, "share_at_1": sum(v == 1 for v in vals) / n}

    summary = {"checkpoint": str(cut) if cut else None, "base_only": cut is None,
               "epoch": marker["epoch"] if marker else 0, "items": len(items), "repeats": len(seeds),
               "images": len(scores), "errors": sum(1 for r in scores if r["errors"]),
               "nonfinite": sum(int(r["nonfinite"]) for r in scores),
               "metrics": {m: stats([r[m] for r in scores]) for m in ("ocr_b_acc", "ocr_a_acc")},
               "agreement_a_b": {"both_1": sum(r["ocr_a_acc"] == 1 and r["ocr_b_acc"] == 1 for r in scores),
                                 "pearson": None},
               "seconds": {"wall": round(wall, 1), "generation": round(t_gen, 1), "ocr_a": round(t_a, 1),
                           "ocr_b": round(t_b, 1)},
               "seconds_per_image": round(wall / max(1, len(scores)), 3), "rss_mb": {**rss, "peak": rss_mb()},
               "sampler_hash": sampler_hash, "device": str(device), "dtype": str(dtype),
               "arms": {m: {"evaluator_id": v[0], "tier_proposal": v[3], "role": v[4],
                            "construct_aligned": m == "ocr_b_acc"} for m, v in METRICS.items()},
               "missing_hashes": [k for k, v in (("model_hash", args.model_hash),
                                                  ("item_set_hash", args.item_set_hash)) if v is None]}
    pairs = [(r["ocr_a_acc"], r["ocr_b_acc"]) for r in scores if r["ocr_a_acc"] is not None
             and r["ocr_b_acc"] is not None]
    if len(pairs) > 2:
        a, b = np.array(pairs).T
        if a.std() > 0 and b.std() > 0:
            summary["agreement_a_b"]["pearson"] = float(np.corrcoef(a, b)[0, 1])
    if device.type == "cuda":
        summary["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
        summary["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
    lock = HERE / "uv.lock"
    write_json(out / "env-runtime.json", {
        "python": sys.version.split()[0], "executable": sys.executable,
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None, "script_sha256": code_sha,
        "sampler_code_sha256": sampler_code_sha256(), "versions": versions, "torch": torch.__version__,
        "checkpoint": str(cut) if cut else None, "items": str(args.items.resolve()),
        "items_sha256": sha256_file(args.items / "items.jsonl"), "uid": os.getuid(), "gid": os.getgid()})
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--make-items", action="store_true")
    p.add_argument("--n-items", type=int, default=400)
    p.add_argument("--split-seed", type=int, default=20261113)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--items", type=Path, default=HERE / "items")
    p.add_argument("--out", type=Path)
    p.add_argument("--eval-seeds", default="11")
    p.add_argument("--model-hash")
    p.add_argument("--base-model-hash")
    p.add_argument("--item-set-hash")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--encoder-dtype", choices=("fp16", "bf16", "fp32"), help="text encoders (default --dtype)")
    p.add_argument("--ocr-cpu-threads", type=int, default=8)
    for k, v in SAMPLER_DEFAULT.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None)
    args = p.parse_args(argv)
    if not args.make_items:
        if args.out is None or (args.checkpoint is None) == (not args.base_only):
            p.error("evaluation needs --out and exactly one of --checkpoint / --base-only")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.make_items:
        return make_items(args.n_items, args.split_seed)
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
