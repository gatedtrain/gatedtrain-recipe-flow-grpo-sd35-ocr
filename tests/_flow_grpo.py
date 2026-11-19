"""Area "image generation, SD3.5" test helpers (verifier-owned; tests/README.md section 11.22, EXT-160 to EXT-178).

Written from the runtime (`runtime/`: README.md "Outputs of
`train_flow_grpo.py`", "Evaluator outputs", "What gatedtrain needs", "Launch request facts"; `train_flow_grpo.py`
`save_cut`, `base_binding`, `resolve_settings`, `main`; `eval_images.py` `evaluate`; read 2026-11-13) and its CPU
sanity outputs of 2026-11-13, not from gatedtrain. Small real pieces are copied into `fixtures/flow_grpo/`: the first
40 registered items and the first 64 training-shard rows (byte slices of `runtime/items/items.jsonl` and
`runtime/prompts/train.jsonl`), the cpu-sanity run's `env-freeze.txt` and its cut's `adapter_config.json`, the cpu-sanity
logs and cut marker under `cpu-sanity/` (with the two shard rows that run sampled), and the arms' `eval_meta.json` and
one arm's `items.jsonl` of the 2026-11-14 CPU evaluation of that cut with the patched `eval_images.py`. `sampler.json` is the deployment sampler of the smoke and full profiles (10 steps, 512 px, fp16
transformer and text encoders, autocast float16; the eleven sigmas of a 512-px base evaluation): the cpu-sanity cut's
`sampler.json` with those five values changed, since no GPU run exists yet. Weights are byte stand-ins (minimal
safetensors files): nothing in gatedtrain loads them (MM-12).

Layout of `FlowGrpoScenario(root)`:
- `<root>/area5-sd35/` the runtime directory stand-in (mounted read-only at `/inputs/a5`, EXT-112): `train_flow_grpo.py`,
  `eval_images.py`, `.venv/bin/python` (stand-ins, never executed), `prompts/{train.jsonl, prompts_meta.json}`,
  `items/{items.jsonl, item_set_meta.json}`, `models/stable-diffusion-3.5-medium/` (the 37 files of the real base with
  stand-in contents, the `*.fp16.*` duplicates byte-identical to their default files, plus a `.cache/huggingface/`
  download-metadata tree that the base hash excludes) and `MODELS.json`;
- `<root>/out/area5-sd35/full-1/`: one training run (seed 5000, 3 epochs of P = 4 prompts x G = 4 images, 4 sampling
  batches, 2 optimizer steps per epoch when samples are kept, cuts at epochs 2 and 3) with run_args.json,
  env-freeze.txt, env-runtime.json, samples.jsonl, epoch_plan.jsonl, train_steps.jsonl, train_log.jsonl,
  memory.jsonl, attempts.jsonl. Epoch 1 keeps every sample; epoch 2 filters one never-rendered group (4 samples);
  epoch 3 filters one always-rendered group and re-includes 2 zero-advantage samples of another group (the padding
  rule) so that 12 samples are kept;
- `<root>/eval/{base,cand}/`: two evaluation runs as `eval_images.py` writes them, each with `images/`, `scores.jsonl`
  and the arms `arm-ocr_b_acc/` and `arm-ocr_a_acc/` (hard-linked images; numeric `sub_metrics` only, as the
  2026-11-14 `eval_images.py` writes them); by default the gatedtrain hashes are already filled (as `ext-arm-fill` leaves
  them);
- spec, registration (gate on `ocr_b_acc`, higher is better, superiority, margin 0, sigma 0.3 fixed, planning effect
  0.15, alpha 0.025, N 40, K 1), trainer.yaml, a GPU launch record (gpu_est_h 1.2), a smoke record and the host
  work's GPU log.
"""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import random
import shutil
import struct
import unittest
from pathlib import Path
from typing import Any, Callable

import yaml

EXT = Path(__file__).resolve().parent / "fixtures"     # the recipe's fixtures (gatedtrain: tests/fixtures/ext)
from gatedtrain.testing.ext import (ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_INTERPRETER, ENV_INTERPRETER_TREES, ENV_OUTPUT_ROOT,
                             IMAGE_ID, gpu_step_script, host_config_sha256, launch_label, safetensors_bytes, sha_of,
                             write_canonical)
from gatedtrain.ext.arm import META_SCHEMA
from gatedtrain.ext.launch import RECORD_SCHEMA
from gatedtrain.ext.smoke import SMOKE_SCHEMA
from gatedtrain.testing.support import canonical_bytes, file_sha256, manifest_entries, manifest_hash, run_cli

FG = EXT / "flow_grpo"
CPU = FG / "cpu-sanity"
ADAPTER = "flow_grpo_sd35"
TRAINER_ID = "rl.flow_grpo@0.1"
EVALUATOR_ID = "t2i.sd35_ocr_doctr_items@0.1"
SUPPORT_ID = "t2i.sd35_ocr_paddle_items@0.1"
METRIC = "ocr_b_acc"
SUPPORT_METRIC = "ocr_a_acc"
AREA5 = Path(__import__("gatedtrain_recipe_flow_grpo_sd35_ocr.rows", fromlist=["RUNTIME"]).RUNTIME)   # the runtime directory
EVAL_IMAGES = AREA5 / "eval_images.py"
TRAIN_SCRIPT = AREA5 / "train_flow_grpo.py"
EVAL_IMAGES_SHA256 = "sha256:2872dc8b38b2811191e1dcb237c8478acbbe2a043e5c1d64d1f2d2810c16e8a7"
# The eval_images.py that wrote the fixture arms (eval_meta.*.cpu-sanity.json) and the 2026-11-14 result; the
# published file differs from it in its docstrings, the schema id it writes (gatedtrain.item_eval) and its
# edit-distance import (RESULT.md).
EVAL_IMAGES_SHA256_RUN = "sha256:fdad8eb9b8de45a0c002a62469fb043fdeaf1c38b9449d41fd5a430e9adee8ad"
# The SHARED SAMPLER block of both scripts (byte-identical), README "Measured hashes"; the cpu-sanity fixtures and the
# 2026-11-14 run recorded the block before its edit-distance import moved to rapidfuzz (`SAMPLER_CODE_SHA_RUN`).
SAMPLER_CODE_SHA = "sha256:bba7ecd54c681151f4f7514cde9844431b5706623ad2233bfd8034a6483cbb6f"
SAMPLER_CODE_SHA_RUN = "sha256:83133e246f4b73daa8cf3576b2e7152960a98fb7f19b5e92ad36cf3b5bbff589"
# The scorers as eval_images.py wrote them in the 2026-11-13 cpu-sanity arms (fixtures eval_meta.*.json), with the
# code hash of the published eval_images.py (the registry rows' pin).
SCORER = {**json.loads((FG / "eval_meta.ocr_b_acc.cpu-sanity.json").read_text(encoding="utf-8"))["scorer"],
          "code_sha256": EVAL_IMAGES_SHA256}
SUPPORT_SCORER = {**json.loads((FG / "eval_meta.ocr_a_acc.cpu-sanity.json").read_text(encoding="utf-8"))["scorer"],
                  "code_sha256": EVAL_IMAGES_SHA256}
SAMPLER = json.loads((FG / "sampler.json").read_text(encoding="utf-8"))
SAMPLER_KEYS = ("autocast", "guidance_scale", "latents_dtype", "max_sequence_length", "negative_prompt", "noise",
                "noise_level", "num_steps", "resolution", "scheduler", "scheduler_config", "sigmas", "step",
                "text_encoder_dtype", "transformer_dtype", "vae_dtype")
SEEDS = [11]
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"
CUT_FILES = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json", "trainer_state.json")
DEPLOYABLE = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json")
# The 37 files of models/stable-diffusion-3.5-medium outside `.cache/` (MODELS.json; the cpu-sanity cut's
# base_ref.json). The `*.fp16.*` duplicates stay (decision "Area 5 decisions", not published).
BASE_FILES = tuple(sorted(json.loads((CPU / "checkpoint-1" / "base_ref.json").read_text(encoding="utf-8"))["files"]))
DUPLICATES = {"text_encoder/model.fp16.safetensors": "text_encoder/model.safetensors",
              "text_encoder_2/model.fp16.safetensors": "text_encoder_2/model.safetensors",
              "text_encoder_3/model.fp16-00001-of-00002.safetensors": "text_encoder_3/model-00001-of-00002.safetensors",
              "text_encoder_3/model.fp16-00002-of-00002.safetensors": "text_encoder_3/model-00002-of-00002.safetensors"}
CACHE_FILES = (".cache/huggingface/.gitignore", ".cache/huggingface/CACHEDIR.TAG",
               ".cache/huggingface/download/model_index.json.lock",
               ".cache/huggingface/download/model_index.json.metadata",
               ".cache/huggingface/download/text_encoder_3/model-00001-of-00002.safetensors.metadata")
BASE_NAME = "stable-diffusion-3.5-medium"
REPO = "stabilityai/stable-diffusion-3.5-medium"
REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
RUNTIME_CONTAINER = "/inputs/a5"
SEED = 5000
SCRIPT_TEXT = "# fixture stand-in for runtime/train_flow_grpo.py; never executed\n"
EVAL_TEXT = "# fixture stand-in for runtime/eval_images.py; never executed\n"
INTERPRETER_TEXT = "# fixture stand-in for the area-5 venv interpreter; never executed\n"
LABEL = "mm-a5-sd35-full"
AREA = 2                                              # decision scope row 2b; the directory is named after the run order
ENV_NAMES = ["DOCTR_CACHE_DIR", "HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
             "TOKENIZERS_PARALLELISM", "XDG_CACHE_HOME"]
# The fixture profile: train_flow_grpo.py COMMON plus a small profile (P = 4, G = 4: 16 samples, 4 sampling batches,
# 2 optimizer steps of 2 accumulated micro-batches per epoch).
PROFILE = {"epochs": 3, "prompts_per_epoch": 4, "images_per_prompt": 4, "sample_batch": 4, "train_batch": 2,
           "save_every": 2}
# Rewards of one prompt group (OCR accuracy, exact binary fractions): `varied` has no member at the group mean;
# `flat0` / `flat1` never / always render the text (all advantages 0); `mean` has two members at the group mean.
PATTERNS = {"varied": [0.0, 0.25, 0.3125, 1.0], "flat0": [0.0, 0.0, 0.0, 0.0], "flat1": [1.0, 1.0, 1.0, 1.0],
            "mean": [0.0, 0.5, 1.0, 0.5]}
DEFAULT_PLAN = {1: ["varied"] * 4, 2: ["varied", "flat0", "varied", "varied"], 3: ["flat1", "mean", "varied", "varied"]}


def run_user() -> str:
    """The container's `<uid>:<gid>` in fixture requests and records: the uid running the tests and its primary group,
    so the output directories the tests create are owned by the container's uid (EXT-63) whoever runs them. The launch
    rules refuse root, so a test that needs it skips under uid 0."""
    uid = os.getuid()
    if uid == 0:
        raise unittest.SkipTest("the launch rules refuse a container running as root (uid 0); run the tests as a "
                                "non-root user")
    return f"{uid}:{pwd.getpwuid(uid).pw_gid}"


def sha256_text(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def derive_seed(*parts) -> int:
    """train_flow_grpo.py / eval_images.py `derive_seed` (SHARED SAMPLER block)."""
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()[:8], "big") >> 1


def compact_hash(obj: Any) -> str:
    """The runtime's own hash encoding (`canonical` in both scripts): compact sorted JSON, not RFC 8785."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def write_json(path: Path, doc: Any) -> Path:
    """train_flow_grpo.py / eval_images.py `write_json`: indent 2, sorted keys, final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, docs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def target_text(prompt: str) -> str:
    """train_flow_grpo.py `target_text` (flow_grpo ocr.py:32)."""
    return prompt.split('"')[1]


# ---------------------------------------------------------------------------
# Base model, cuts and hashes (README "What gatedtrain needs" items 1-3; MM-17, MM-18, MM-20)
# ---------------------------------------------------------------------------

def build_base(d: Path, tag: str = "", *, cache: bool = True) -> Path:
    """The SD3.5 Medium directory with the real base's 37 file names (stand-in contents; the `*.fp16.*` files are
    byte copies of their default files, as on disk) and, with `cache`, a `.cache/huggingface/` tree."""
    for rel in BASE_FILES:
        if rel in DUPLICATES:
            continue
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".safetensors"):
            p.write_bytes(safetensors_bytes(f"{rel} {tag}"))
        elif rel.endswith(".json"):
            p.write_text(json.dumps({"_fixture": rel, "_class_name": rel.split("/")[0], "tag": tag}, indent=2) + "\n",
                         encoding="utf-8")
        else:
            p.write_text(f"fixture stand-in {rel} {tag}\n", encoding="utf-8")
    for dup, src in DUPLICATES.items():
        shutil.copyfile(d / src, d / dup)
    if cache:
        for rel in CACHE_FILES:
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text(f"download metadata stand-in {rel}\n", encoding="utf-8")
    return d


def base_files(base: Path) -> dict[str, str]:
    return {rel: file_sha256(base / rel) for rel in BASE_FILES}


def base_manifest_hash(base: Path) -> str:
    """The base's model hash: the manifest of every file outside `.cache/` (decision "Area 5 decisions", not
    published)."""
    entries = [e for e in manifest_entries(base) if not e["path"].startswith(".cache/")]
    return "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()


def models_json(base: Path) -> dict:
    """MODELS.json as fetch_models.py writes it (the base entry only)."""
    return {"schema": "ptc_mm.models", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
            "models": {BASE_NAME: {"repo": REPO, "revision": REVISION,
                                   "files": {rel: {"sha256": file_sha256(base / rel), "size": (base / rel).stat().st_size}
                                             for rel in BASE_FILES}}}}


def base_ref_doc(base: Path, *, revision: str = REVISION) -> dict:
    """train_flow_grpo.py `base_binding` reduced to the cut's base_ref.json keys."""
    files = base_files(base)
    return {"repo": REPO, "revision": revision, "files": files, "files_sha256": compact_hash(files)}


def lora_bytes(tag: str) -> bytes:
    """`save_lora_adapter` + `canonicalize_safetensors_header`: sorted compact header, space padding to 8 bytes,
    the LoraConfig in `__metadata__.lora_adapter_metadata`."""
    meta = json.dumps({"lora_alpha": 64, "r": 32, "target_modules": ["attn.add_k_proj", "attn.to_q"]}, indent=2,
                      sort_keys=True)
    header = {"__metadata__": {"format": "pt", "lora_adapter_metadata": meta},
              "transformer_blocks.0.attn.to_q.lora_A.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
              "transformer_blocks.0.attn.to_q.lora_B.weight": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]}}
    text = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text + hashlib.sha256(tag.encode()).digest()[:16]


def build_cut(out: Path, epoch: int, *, global_step: int, samples_consumed: int, samples_filtered: int, base: Path,
              script_sha256: str, profile: str = "full", seed: int = SEED, tag: str = "", ema: bool = False,
              sampler: dict | None = None) -> Path:
    """`checkpoint-<epoch>/` as train_flow_grpo.py save_cut writes it, `cut.json` last."""
    d = out / f"checkpoint-{epoch}"
    d.mkdir(parents=True)
    (d / LORA_WEIGHTS).write_bytes(lora_bytes(f"lora epoch {epoch} {tag}"))
    cfg = json.loads((FG / "adapter_config.json").read_text(encoding="utf-8"))
    cfg["weights"] = "ema" if ema else "raw"
    write_json(d / "adapter_config.json", cfg)
    write_json(d / "sampler.json", sampler if sampler is not None else SAMPLER)
    write_json(d / "base_ref.json", base_ref_doc(base))
    write_json(d / "trainer_state.json", {"epoch": epoch, "global_step": global_step,
                                          "note": "resume-only information; the run does not resume from a cut "
                                                  "(a preempted run restarts)",
                                          "samples_consumed": samples_consumed, "samples_filtered": samples_filtered})
    files = {n: file_sha256(d / n) for n in CUT_FILES}
    write_json(d / "cut.json", {
        "schema": "ptc_mm.flow_grpo_sd35_cut", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "epoch": epoch, "global_step": global_step, "samples_consumed": samples_consumed,
        "samples_filtered": samples_filtered, "files": files, "deployable": list(DEPLOYABLE),
        "created_at": "2026-11-14T20:00:00Z", "script_sha256": script_sha256, "sampler_code_sha256": SAMPLER_CODE_SHA,
        "seed": seed, "profile": profile, "ema": ema})
    return d


def adapter_hash(cut: Path) -> str:
    """Manifest hash (T0-09) of the deployable components of a cut (cut.json `deployable`)."""
    entries = [e for e in manifest_entries(cut) if e["path"] in DEPLOYABLE]
    return "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()


def lora_model_hash(cut: Path, base: Path) -> str:
    """MM-18: sha256(canonical_json({base: base_hash, adapter: adapter_hash})), the base hash without `.cache/`."""
    return sha_of({"base": base_manifest_hash(base), "adapter": adapter_hash(cut)})


# ---------------------------------------------------------------------------
# A training run (train_flow_grpo.py main: samples.jsonl at sampling time, epoch_plan.jsonl before the epoch's first
# optimizer step, train_steps.jsonl before each step)
# ---------------------------------------------------------------------------

def settings(profile: dict | None = None, seed: int = SEED) -> dict:
    """train_flow_grpo.py `resolve_settings` on COMMON (the cpu-sanity run's settings without its profile keys)."""
    s = json.loads((CPU / "run_args.json").read_text(encoding="utf-8"))["settings"]
    s.update({"num_steps": 10, "deploy_num_steps": 10, "resolution": 512, "precision": "fp16",
              "encoder_precision": "fp16", "reward": "ocr", "ocr_cpu_threads": 8})
    s.update(profile or PROFILE)
    s["seed"] = seed
    s["samples_per_epoch"] = s["prompts_per_epoch"] * s["images_per_prompt"]
    s["num_batches_per_epoch"] = s["samples_per_epoch"] // s["sample_batch"]
    s["gradient_accumulation_steps"] = s["num_batches_per_epoch"] // 2
    s["num_train_timesteps"] = int(s["num_steps"] * s["timestep_fraction"])
    return s


def simulate(prompts: list[dict], s: dict, *, attempt: int = 1, plan: dict[int, list[str]] | None = None
             ) -> tuple[list[dict], list[dict], list[dict], dict[int, tuple]]:
    """(samples.jsonl, epoch_plan.jsonl, train_steps.jsonl lines, {epoch: (global_step, consumed, filtered)}).
    The zero-advantage filter and its padding as train_flow_grpo.py :480-503 (a sample whose reward equals its
    group's mean has advantage 0; padding picks from the zero-advantage samples with a seeded generator); the training
    order as :512-564 (one permutation per inner epoch, `num_batches_per_epoch` virtual batches, one optimizer step
    per `gradient_accumulation_steps` of them). Random draws use `random.Random` (the runtime uses numpy and torch
    generators; the adapter does not re-derive these draws)."""
    plan = plan or DEFAULT_PLAN
    seed, P, G, E = s["seed"], s["prompts_per_epoch"], s["images_per_prompt"], s["epochs"]
    nb, gas, inner_n = s["num_batches_per_epoch"], s["gradient_accumulation_steps"], s["num_inner_epochs"]
    samples, plans, steps, state = [], [], [], {}
    gs = consumed = filtered = 0
    for e in range(1, E + 1):
        chosen = random.Random(derive_seed("prompts", seed, e)).sample(range(len(prompts)), P)
        rows = [(prompts[c], g) for c in chosen for g in range(G)]
        pats = plan.get(e, ["varied"] * P)
        ids, rewards = [], []
        for i, (row, g) in enumerate(rows):
            noise = derive_seed("sample", seed, e, i)
            sid = sha256_text(json.dumps(["flow-grpo-sd35", seed, attempt, e, i, row["prompt_id"], noise, g],
                                         sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
            r = PATTERNS[pats[i // G]][g]
            target = target_text(row["prompt"])
            ids.append(sid)
            rewards.append(r)
            samples.append({"attempt": attempt, "epoch": e, "member": g, "ocr_acc": r, "ocr_error": None,
                            "ocr_text": target if r == 1.0 else "garbled text", "prompt": row["prompt"],
                            "prompt_id": row["prompt_id"], "reward": r, "sample_id": sid, "sample_index": i,
                            "seed": noise, "target": target, "utc": "2026-11-14T19:00:00Z"})
        n = len(rows)
        means = [sum(rewards[k * G:(k + 1) * G]) / G for k in range(P)]
        mask = [rewards[i] != means[i // G] for i in range(n)]
        reincluded = []
        true_count = sum(mask)
        if true_count % nb:
            false_idx = [i for i in range(n) if not mask[i]]
            need = nb - true_count % nb
            if len(false_idx) >= need:
                reincluded = sorted(random.Random(derive_seed("mask", seed, e)).sample(false_idx, need))
                for i in reincluded:
                    mask[i] = True
        kept = [i for i in range(n) if mask[i]]
        n_steps = (nb // gas) * inner_n if kept else 0
        plans.append({"epoch": e, "filter_reason": "zero_advantage", "filtered": [ids[i] for i in range(n) if not mask[i]],
                      "kept": [ids[i] for i in kept], "reincluded": [ids[i] for i in reincluded],
                      "steps": [gs + 1, gs + n_steps] if n_steps else [], "utc": "2026-11-14T19:00:01Z"})
        if kept:
            m = len(kept) // nb
            for inner in range(inner_n):
                order = list(kept)
                random.Random(derive_seed("perm", seed, e, inner)).shuffle(order)
                virtual = [order[v * m:(v + 1) * m] for v in range(nb)]
                acc = []
                for v, vidx in enumerate(virtual):
                    acc += [ids[i] for i in vidx]
                    if (v + 1) % gas == 0:
                        gs += 1
                        steps.append({"epoch": e, "inner": inner, "sample_ids": acc, "step": gs,
                                      "utc": "2026-11-14T19:00:05Z"})
                        acc = []
            consumed += len(kept)
        filtered += n - len(kept)
        state[e] = (gs, consumed, filtered)
    return samples, plans, steps, state


def run_args(*, s: dict, profile: str, attempt: int, base_container: str, train_container: str, train_sha256: str,
             n_train: int, files_sha256: str, script_sha256: str) -> dict:
    """train_flow_grpo.py run_args.json: the cpu-sanity run's document with this run's values."""
    doc = json.loads((CPU / "run_args.json").read_text(encoding="utf-8"))
    training = dict(SAMPLER, noise_level=s["noise_level"])
    doc.update({"profile": profile, "attempt": attempt, "settings": s, "seed": s["seed"], "base_model": base_container,
                "base_ref": {"repo": REPO, "revision": REVISION, "files_sha256": files_sha256, "verified": True},
                "train_file": train_container, "train_file_sha256": train_sha256, "n_train_prompts": n_train,
                "sampler": SAMPLER, "sampler_sha256": compact_hash(SAMPLER), "training_sampler": training,
                "device": "cuda", "dtype": "torch.float16", "gpu": "NVIDIA GeForce RTX 4090",
                "sampler_code_sha256": SAMPLER_CODE_SHA, "script_sha256": script_sha256,
                "created_at": "2026-11-14T19:00:00Z"})
    return doc


def build_run(out: Path, *, runtime: Path, base: Path, script_sha256: str, profile: str = "full",
              s: dict | None = None, plan: dict[int, list[str]] | None = None) -> tuple[Path, dict[int, tuple]]:
    s = s or settings()
    out.mkdir(parents=True)
    prompts = read_jsonl(runtime / "prompts" / "train.jsonl")
    samples, plans, steps, state = simulate(prompts, s, plan=plan)
    shutil.copyfile(FG / "env-freeze.txt", out / "env-freeze.txt")
    rt = json.loads((CPU / "env-runtime.json").read_text(encoding="utf-8"))
    rt.update({"executable": f"{RUNTIME_CONTAINER}/.venv/bin/python", "image_cuda_version": "13.0.3",
               "script_sha256": script_sha256, "env_freeze_sha256": file_sha256(out / "env-freeze.txt"),
               "sampler_code_sha256": SAMPLER_CODE_SHA})             # this code's block, not the cpu-sanity run's
    write_json(out / "env-runtime.json", rt)
    write_json(out / "run_args.json", run_args(
        s=s, profile=profile, attempt=1, base_container=f"{RUNTIME_CONTAINER}/models/{BASE_NAME}",
        train_container=f"{RUNTIME_CONTAINER}/prompts/train.jsonl",
        train_sha256=file_sha256(runtime / "prompts" / "train.jsonl"), n_train=len(prompts),
        files_sha256=base_ref_doc(base)["files_sha256"], script_sha256=script_sha256))
    write_jsonl(out / "samples.jsonl", samples)
    write_jsonl(out / "epoch_plan.jsonl", plans)
    write_jsonl(out / "train_steps.jsonl", steps)
    write_jsonl(out / "train_log.jsonl", [{"step": st["step"], "epoch": st["epoch"], "policy_loss": 0.0}
                                         for st in steps])
    write_jsonl(out / "memory.jsonl", [{"epoch": e, "seconds_epoch": 420.0, "kept": None} for e in state])
    write_jsonl(out / "attempts.jsonl", [{"attempt": 1, "started_at": "2026-11-14T19:00:00Z"}])
    for e, (gs, consumed, filtered) in state.items():
        if e % s["save_every"] == 0 or e == s["epochs"]:
            build_cut(out, e, global_step=gs, samples_consumed=consumed, samples_filtered=filtered, base=base,
                      script_sha256=script_sha256, profile=profile, seed=s["seed"])
    return out, state


# ---------------------------------------------------------------------------
# Evaluation runs (eval_images.py evaluate: images, scores.jsonl, one arm per metric)
# ---------------------------------------------------------------------------

def baseline_score(i: int, r: int = 0) -> float:
    return ((i * 7) % 9) / 8


def candidate_score(i: int, r: int = 0) -> float:
    return min(1.0, baseline_score(i, r) + 0.125 + 0.0625 * (i % 3))


def build_eval(out: Path, items: list[dict], score_fn: Callable[[int, int], float], *, tag: str,
               model: str | None, base_model: str | None, item_set: str | None, filled: bool = True,
               metrics: tuple[str, ...] = (METRIC, SUPPORT_METRIC)) -> Path:
    """`<out>/images/<item>/r<r>.png`, `<out>/scores.jsonl` and `<out>/arm-<metric>/` per metric with hard-linked
    images, as eval_images.py writes them. `filled`: the gatedtrain hashes are set (model, item set, RFC 8785 sampler
    hash), as after ext-arm-fill; otherwise the evaluator's own output (nulls, compact-JSON sampler hash)."""
    out.mkdir(parents=True)
    rows = []
    for r, seed in enumerate(SEEDS):
        for i, it in enumerate(items):
            rel = f"images/{it['item']}/r{r}.png"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_bytes(b"\x89PNG\r\n\x1a\n" + f"stand-in {tag} {it['item']} {r}".encode())
            b = score_fn(i, r)
            rows.append({"item": it["item"], "repeat": r, "seed": seed, "noise_seed": derive_seed("eval", seed, it["item"]),
                         "target": it["target"], "output_ref": {"path": rel, "sha256": file_sha256(out / rel)},
                         "nonfinite": False, "errors": {}, "ocr_a_text": "text a", "ocr_a_acc": b,
                         "ocr_b_text": "textb", "ocr_b_acc": b})
    write_jsonl(out / "scores.jsonl", rows)
    scorers = {METRIC: (EVALUATOR_ID, SCORER), SUPPORT_METRIC: (SUPPORT_ID, SUPPORT_SCORER)}
    for metric in metrics:
        arm = out / f"arm-{metric}"
        arm.mkdir()
        lines = []
        for row in rows:
            dst = arm / row["output_ref"]["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(out / row["output_ref"]["path"], dst)
            # eval_images.py (sha256 fdad8eb9...): finite numbers only (MM-1); the target and the texts stay in
            # scores.jsonl; an engine that raised is left out.
            sub = {k: row[k] for k in ("ocr_a_acc", "ocr_b_acc") if row[k] is not None}
            sub["nonfinite"] = int(row["nonfinite"])
            lines.append({"item": row["item"], "repeat": row["repeat"], "seed": row["seed"], "score": row[metric],
                          "error": None, "sub_metrics": sub, "output_ref": row["output_ref"]})
        write_jsonl(arm / "items.jsonl", lines)
        evaluator, scorer = scorers[metric]
        write_json(arm / "eval_meta.json", {
            "schema": META_SCHEMA, "schema_version": "1.0", "evaluator_id": evaluator, "adapter": ADAPTER,
            "metric": metric, "direction": "higher_is_better", "score_range": [0, 1],
            "checkpoint": {"model_hash": model if filled else None, "base_model_hash": base_model if filled else None},
            "normalizer_sha256": None, "env_version_sha256": None,
            "item_set_hash": item_set if filled else None, "eval_seeds": list(SEEDS),
            "sampler": SAMPLER, "sampler_hash": sha_of(SAMPLER) if filled else compact_hash(SAMPLER),
            "scorer": dict(scorer), "framework": {"name": "diffusers", "version": "0.42.1"},
            "created_at": "2026-11-14T21:00:00Z"})
    return out


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def smoke_record(*, base_model: str, item_set: str, **overrides) -> dict[str, Any]:
    """A PASS smoke.json (MM-38) on the deciding metric: 40 baseline ocr_b_acc scores in [0, 1], no value shared by
    half of them, mean inside the 5 % margins."""
    rec = {"schema": SMOKE_SCHEMA, "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
           "adapter": ADAPTER, "area": AREA, "base_model_hash": base_model, "item_set_hash": item_set,
           "peak_gpu_mem_gb": 13.0, "host_ram_peak_gb": 16.0, "swap_peak_gb": 0.0, "seconds_per_step": 225.0,
           "projected_gpu_h": 1.2, "gpu_log_total_h": 6.0,
           "baseline_scores": [baseline_score(i) for i in range(40)], "error_count": 0, "verdict": "PASS",
           "reasons": []}
    rec.update(overrides)
    return rec


def guard_log(host_work: Path, label: str, *, lines: list[str] | None = None) -> Path:
    """gpu-step.sh's log: one finished attempt of 3600 s."""
    host_work.mkdir(parents=True, exist_ok=True)
    lines = lines if lines is not None else [f"2026-11-11T09:00:00Z START {label} attempt=1",
                                             f"2026-11-11T10:00:00Z END {label} attempt=1 rc=0"]
    p = host_work / "gpu-guard.log"
    p.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    return p


class FlowGrpoScenario:
    """A whole area-5 run collected against the SD3.5 Medium base (module docstring)."""

    def __init__(self, root: Path, *, n_items: int = 40, reg_mutate: Callable[[dict], None] | None = None,
                 overlap: int = 0, filled: bool = True, inner_epochs: int = 1,
                 plan: dict[int, list[str]] | None = None, cache: bool = True):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.runtime = root / "area5-sd35"
        self.data = self.runtime / "prompts"
        self.items_dir = self.runtime / "items"
        self.data.mkdir(parents=True)
        self.items_dir.mkdir(parents=True)
        all_items = read_jsonl(FG / "items.jsonl")[:n_items]
        write_jsonl(self.items_dir / "items.jsonl", all_items)
        write_json(self.items_dir / "item_set_meta.json", {"schema": "ptc_mm.sd35_ocr_item_set", "n_items": len(all_items)})
        prompts = read_jsonl(FG / "train.jsonl")
        for row, item in zip(prompts[:overlap], all_items):          # exact overlap: a shard prompt equals an item's
            row["prompt"] = item["prompt"]
        if overlap:
            write_jsonl(self.data / "train.jsonl", prompts)
        else:
            shutil.copyfile(FG / "train.jsonl", self.data / "train.jsonl")
        write_json(self.data / "prompts_meta.json", {"schema": "ptc_mm.sd35_ocr_train_prompts", "n_train": len(prompts)})
        self.train_rows = prompts
        self.item_rows = all_items
        self.items = [r["item"] for r in all_items]
        self.base = build_base(self.runtime / "models" / BASE_NAME, cache=cache)
        write_json(self.runtime / "MODELS.json", models_json(self.base))
        self.script = self.runtime / "train_flow_grpo.py"
        self.script.write_text(SCRIPT_TEXT, encoding="utf-8")
        (self.runtime / "eval_images.py").write_text(EVAL_TEXT, encoding="utf-8")
        (self.runtime / "uv.lock").write_text("# lock stand-in\n", encoding="utf-8")
        self.interpreter = self.runtime / ".venv" / "bin" / "python"
        self.interpreter.parent.mkdir(parents=True)
        self.interpreter.write_text(INTERPRETER_TEXT, encoding="utf-8")
        (root / "trainer.yaml").write_text("# area 5 trainer config (fixture)\nbackend: flow_grpo_sd35\nprofile: full\n",
                                           encoding="utf-8")
        self.host_work = root / "host-work"
        self.output_root = root / "out"
        gpu_step_script(self.host_work)
        (self.host_work / "gpu-log.md").write_bytes((EXT / "gpu-log.fixture.md").read_bytes())
        self.stage_out, self.state = build_run(self.output_root / "area5-sd35" / "full-1", runtime=self.runtime,
                                               base=self.base, script_sha256=file_sha256(self.script),
                                               s={**settings(), "num_inner_epochs": inner_epochs}, plan=plan)
        self.cuts = sorted(self.stage_out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        self.final_cut = self.cuts[-1]
        self.base_model = base_manifest_hash(self.base)
        self.cand_model = lora_model_hash(self.final_cut, self.base)
        self.item_set_hash = manifest_hash(self.items_dir)
        self.reg_mutate = reg_mutate
        self.render()
        self.base_eval = build_eval(root / "eval" / "base", self.item_rows, baseline_score, tag="base",
                                    model=self.base_model, base_model=self.base_model, item_set=self.item_set_hash,
                                    filled=filled)
        self.cand_eval = build_eval(root / "eval" / "cand", self.item_rows, candidate_score, tag="cand",
                                    model=self.cand_model, base_model=self.base_model, item_set=self.item_set_hash,
                                    filled=filled)
        self.baseline = self.base_eval / f"arm-{METRIC}"
        self.candidate = self.cand_eval / f"arm-{METRIC}"
        self.launch = write_canonical(root / "launch.json", self.launch_record())
        self.label = self.launch_record()["label"]
        self.smoke = write_canonical(root / "smoke.json", smoke_record(base_model=self.base_model,
                                                                         item_set=self.item_set_hash))
        guard_log(self.host_work, self.label)

    def env(self) -> dict[str, str]:
        return {ENV_HOST_WORK: str(self.host_work), ENV_OUTPUT_ROOT: str(self.output_root),
                ENV_INPUT_ROOTS: str(self.root), ENV_INTERPRETER: f"{RUNTIME_CONTAINER}/.venv/bin/python",
                ENV_INTERPRETER_TREES: str(self.interpreter.parent)}

    @property
    def paths(self) -> dict[str, Path]:
        return {"spec": self.root / "spec.yaml", "registration": self.root / "registration.yaml"}

    def render(self, reg_mutate: Callable[[dict], None] | None = None,
               spec_mutate: Callable[[dict], None] | None = None) -> None:
        text = (FG / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", self.item_set_hash).replace("{sampler}", json.dumps(SAMPLER))
        text = text.replace("{evaluator_hash}", sha_of({"evaluator": EVALUATOR_ID, "metric": METRIC,
                                                         "direction": "higher_is_better", "scorer": SCORER}))
        mutate = reg_mutate or self.reg_mutate
        if mutate is not None:
            reg = yaml.safe_load(text)
            mutate(reg)
            text = yaml.safe_dump(reg, sort_keys=False)
        (self.root / "registration.yaml").write_text(text, encoding="utf-8")
        spec = (FG / "spec.template.yaml").read_text(encoding="utf-8")
        spec = spec.replace("{registration_sha256}", file_sha256(self.root / "registration.yaml")[len("sha256:"):])
        spec = spec.replace("{base}", str(self.base)).replace("{data}", str(self.data))
        spec = spec.replace("{items}", str(self.items_dir))
        if spec_mutate is not None:
            doc = yaml.safe_load(spec)
            spec_mutate(doc)
            spec = yaml.safe_dump(doc, sort_keys=False)
        (self.root / "spec.yaml").write_text(spec, encoding="utf-8")

    def launch_record(self, **overrides) -> dict[str, Any]:
        """ext-launch's record of the full run (README "Launch request facts", full; gpus true)."""
        rec = {"schema": RECORD_SCHEMA, "schema_version": "2.0", "tags": ["prototype", "unpromotable"],
               "adapter": ADAPTER, "area": AREA, "profile": "full", "label": LABEL, "host_work": str(self.host_work),
               "runtime": {"kind": "venv_in_container", "image": IMAGE_ID,
                           "venv_lock_sha256": file_sha256(self.runtime / "uv.lock")},
               "framework": {"name": "diffusers", "version": "0.42.1", "commit": None},
               "training_seed": SEED, "gpus": True,
               "argv": [f"{RUNTIME_CONTAINER}/.venv/bin/python", f"{RUNTIME_CONTAINER}/train_flow_grpo.py",
                        "--profile", "full", "--out", "/out", "--seed", str(SEED), "--epochs", "3"],
               "mounts": [{"host": str(self.runtime), "container": RUNTIME_CONTAINER, "mode": "ro"},
                          {"host": str(self.stage_out), "container": "/out", "mode": "rw"}],
               "tmpfs": ["/tmp"], "network": "none", "network_reason": None, "user": run_user(),
               "env_names": list(ENV_NAMES), "gpu_est_h": 1.2, "gpu_used_h": 6.0,
               "output_dir": str(self.stage_out), "output_root": str(self.output_root),
               "gpu_step_sha256": file_sha256(self.host_work / "bin" / "gpu-step.sh"), "allow_network": False,
               "interpreter_path": f"{RUNTIME_CONTAINER}/.venv/bin/python",
               "interpreter_sha256": file_sha256(self.interpreter),
               "script_path": f"{RUNTIME_CONTAINER}/train_flow_grpo.py", "script_sha256": file_sha256(self.script),
               "host_config_sha256": host_config_sha256(self.host_work, self.output_root, [self.root],
                                                         interpreter=f"{RUNTIME_CONTAINER}/.venv/bin/python",
                                                         interpreter_trees=[self.interpreter.parent])}
        rec.update(overrides)
        rec["label"] = launch_label(rec)
        return rec

    def edit_json(self, path: Path, fn: Callable[[dict], None]) -> None:
        doc = json.loads(path.read_text(encoding="utf-8"))
        fn(doc)
        write_json(path, doc)

    def logs(self) -> tuple[list[dict], list[dict], list[dict]]:
        return (read_jsonl(self.stage_out / "samples.jsonl"), read_jsonl(self.stage_out / "epoch_plan.jsonl"),
                read_jsonl(self.stage_out / "train_steps.jsonl"))

    def edit_logs(self, fn: Callable[[list[dict], list[dict], list[dict]], None]) -> None:
        samples, plans, steps = self.logs()
        fn(samples, plans, steps)
        write_jsonl(self.stage_out / "samples.jsonl", samples)
        write_jsonl(self.stage_out / "epoch_plan.jsonl", plans)
        write_jsonl(self.stage_out / "train_steps.jsonl", steps)

    def reseal(self, cut: Path) -> None:
        """Rewrite cut.json's file hashes after a cut file was edited (the cut stays complete)."""
        self.edit_json(cut / "cut.json", lambda d: d.update(files={n: file_sha256(cut / n) for n in CUT_FILES}))

    def collect(self, runs_root: Path, *, baseline: Path | None = None, candidate: Path | None = None,
                launch: Path | None = None, smoke: Path | None = None, timeout: int = 300):
        args = ["ext-collect", str(self.paths["spec"]), "--adapter", ADAPTER, "--stage-out", str(self.stage_out),
                "--launch-record", str(launch or self.launch), "--smoke-record", str(smoke or self.smoke),
                "--eval-baseline", str(baseline or self.baseline), "--eval-candidate", str(candidate or self.candidate),
                "--runs-root", str(runs_root)]
        return run_cli(args, cwd=self.root, timeout=timeout, env=self.env())

    def fill(self, arm: Path, checkpoint: Path, *, base: Path | None = None):
        args = ["ext-arm-fill", "--adapter", ADAPTER, "--checkpoint", str(checkpoint), "--items", str(self.items_dir)]
        if base is not None:
            args += ["--base", str(base)]
        return run_cli(args + [str(arm)], cwd=self.root, env=self.env())

    def request(self, out: Path, **overrides) -> dict[str, Any]:
        """An ext-launch request into `out` (README "Launch request facts"; argv of the full training run)."""
        rec = self.launch_record()
        req = {"area": AREA, "label": LABEL, "host_work": rec["host_work"],
               "runtime": {"kind": "venv_in_container", "image": IMAGE_ID, "venv_lock": str(self.runtime / "uv.lock")},
               "framework": rec["framework"], "training_seed": SEED, "gpus": True, "network": "none",
               "network_reason": None, "user": run_user(),
               "env": {"HF_HUB_OFFLINE": "1", "HOME": "/tmp/home", "XDG_CACHE_HOME": "/tmp/cache",
                       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                       "DOCTR_CACHE_DIR": "/tmp/cache/doctr"},
               "mounts": [rec["mounts"][0], {"host": str(out), "container": "/out", "mode": "rw"}],
               "tmpfs": ["/tmp"], "output_dir": str(out), "gpu_est_h": 0.3, "argv": list(rec["argv"])}
        req.update(overrides)
        return req

    def launch_request(self, req: dict, name: str, *, profile: str = "smoke", smoke: Path | None = None,
                       extra: list[str] = ()):
        path = self.root / f"{name}.request.yaml"
        path.write_text(yaml.safe_dump(req, sort_keys=False), encoding="utf-8")
        out = self.root / f"{name}.launch.json"
        args = ["ext-launch", "--adapter", ADAPTER, "--profile", profile, "--request", str(path), "--gpu-log",
                str(self.host_work / "gpu-log.md"), "--out-record", str(out)]
        if smoke is not None:
            args += ["--smoke-record", str(smoke)]
        return run_cli(args + list(extra), cwd=self.root, env=self.env()), out


def shared_sampler_sha256(script: Path) -> str:
    """train_flow_grpo.py / eval_images.py `sampler_code_sha256`: SHA-256 of the text from `# BEGIN SHARED SAMPLER` up
    to `# END SHARED SAMPLER`."""
    text = script.read_text(encoding="utf-8")
    return sha256_text(text[text.index("# BEGIN SHARED SAMPLER"):text.index("# END SHARED SAMPLER")].encode("utf-8"))
