"""Registry rows of adapter `flow_grpo_sd35`: Flow-GRPO + LoRA on SD3.5 Medium (area 5).

The training row, the evaluator rows and the adapter class, loaded by `gatedtrain.ext.registry` through the
`gatedtrain.adapters` entry point (MULTIMODAL-SPEC section 6)."""

from __future__ import annotations

from typing import Any

from gatedtrain.ext.rowkit import CONTAINER_PATHS, runtime_dir, venv_runtime_rows

NAME = "flow_grpo_sd35"
# The runtime directory: this repository's runtime/ (scripts, its .venv, models, OCR models, prompts, items; mounted
# read-only at its own host path), or the override; earlier runs lie under the host's output root as
# `{output_root}/<OUT_NAME>`.
RUNTIME_ENV = "GATEDTRAIN_RECIPE_FLOW_GRPO_SD35_OCR_RUNTIME"
OUT_NAME = "flow-grpo-sd35-ocr"
RUNTIME = runtime_dir(RUNTIME_ENV, __file__)
HOST_ROWS = venv_runtime_rows(RUNTIME, OUT_NAME)

TRAINER_ROWS: dict[str, dict[str, Any]] = {}
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {}

# Area "image generation, SD3.5" (MULTIMODAL-SPEC section 6 row 2b and its deviations; runtime/README.md
# "Decisions taken"; runtime/README.md "What gatedtrain needs" items 1-9 and "Launch request facts", read 2026-11-13).
# The trainer is the Flow-GRPO port train_flow_grpo.py on diffusers + peft (EMA off), LoRA cuts on SD3.5 Medium; its
# epoch plan logs zero-advantage `filtered` samples (MM-28).
TRAINER_ROWS["flow_grpo_sd35"] = {
    "uses": ("rl.flow_grpo@0.1",),
    # from env-freeze.txt (MM-33): the framework and the reward engine (README item 9); python-doctr is the evaluator's.
    "pins": ("diffusers", "peft", "torch", "transformers", "paddleocr", "paddlepaddle"),
    "outputs": ("checkpoint-<epoch>/ LoRA cuts with cut.json written last", "run_args.json", "samples.jsonl",
                "epoch_plan.jsonl", "train_steps.jsonl"),
    # EXT-162: train_flow_grpo.py DEPLOYABLE (cut.json `deployable`), as area 3.
    "components": ("pytorch_lora_weights.safetensors", "adapter_config.json", "sampler.json", "base_ref.json"),
    # EXT-165: observed per-sample log with filtered outcomes; a log that does not match the run refuses (no fallback,
    # as ddpo_sd15; the README proposed `stage`).
    "consumption": {"target": "row", "fallback": None},
    "gpu_h_estimate": 1.9,                           # estimate (README "Profiles and cost": area total)
    "evaluators": ("t2i.sd35_ocr_doctr_items@0.1", "t2i.sd35_ocr_paddle_items@0.1"),   # deciding first (MM-38)
    "framework": "diffusers",
    # EXT-173: the interpreter, the runtime directory (scripts, venv, models, OCR models, prompts, items) and the
    # training runs' outputs (an evaluation mounts a cut read-only at /inputs/cut).
    "input_roots": HOST_ROWS["input_roots"],
    "spec_binding": False,                           # the smoke launch precedes the registration (sigma from the smoke)
    # README "Launch request facts": every name, none a credential (HF_TOKEN is never allowed); PYTORCH_CUDA_ALLOC_CONF
    # is optional.
    "env_names": ("DOCTR_CACHE_DIR", "HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
                  "PYTORCH_CUDA_ALLOC_CONF", "TOKENIZERS_PARALLELISM", "XDG_CACHE_HOME"),
    "scan_skip": HOST_ROWS["scan_skip"],
    "interpreter_trees": HOST_ROWS["interpreter_trees"],
    "interpreter": HOST_ROWS["interpreter"],
    "container_paths": CONTAINER_PATHS,
    "env_paths": ("DOCTR_CACHE_DIR", "HOME", "XDG_CACHE_HOME"),
}

# Area "image generation, SD3.5" evaluators (runtime/README.md "Decisions taken": deciding `ocr_b_acc` T3, construct-aligned
# under R15-1, blocking; `ocr_a_acc` supporting). Scorer: eval_images.py on the first quoted text of the prompt:
# `ocr_a_acc` with flow_grpo's OCR accuracy formula (ocr.py:46-60, the reward's), `ocr_b_acc` with the approximate-
# substring edit distance ("Decisions taken", smoke finding of 2026-11-14: chosen after the smoke
# baseline, before registration); code_sha256 its SHA-256; the versions and weight hashes are the strings it
# writes (its 2026-11-14 CPU evaluation of the cpu-sanity cut). An item is one registered prompt of flow_grpo's OCR test split; its `prompt`
# is the exact-overlap key (EXT-168); `source_line` is an integer and not a payload field.
# The published eval_images.py differs from the file that produced the 2026-11-14 result
# (sha256:fdad8eb9b8de45a0c002a62469fb043fdeaf1c38b9449d41fd5a430e9adee8ad) in its docstrings (a cleanup, then the MIT
# attribution of its ported code), in the schema id it writes (`gatedtrain.item_eval` since the rename to gatedtrain
# 0.3.0) and in its edit-distance import (rapidfuzz instead of python-Levenshtein, the same values); only this pin is
# accepted (RESULT.md).
EVAL_IMAGES_SD35_SHA256 = "sha256:2872dc8b38b2811191e1dcb237c8478acbbe2a043e5c1d64d1f2d2810c16e8a7"
_SD35 = {
    "direction": "higher_is_better",
    "score_range": (0, 1),
    "binary": False,
    "error_classes": ("scorer_error",),                  # MM-4: an OCR engine raised (eval_images.py)
    "score_outcomes": ("nonfinite_image",),              # MM-4: a non-finite image is scored on the NaN-zeroed image
    "payload_text_field": "prompt",
    "item_payload": ("prompt", "target"),
    "normalizer": False,
    "env_version": False,
    # EXT-171: the deployment sampler of the cuts (both scripts' `sampler_doc`), exactly these keys.
    "sampler_keys": ("autocast", "guidance_scale", "latents_dtype", "max_sequence_length", "negative_prompt", "noise",
                     "noise_level", "num_steps", "resolution", "scheduler", "scheduler_config", "sigmas", "step",
                     "text_encoder_dtype", "transformer_dtype", "vae_dtype"),
    "framework": "diffusers",
    "samples": "400, 1",
}
EVALUATOR_ROWS["t2i.sd35_ocr_doctr_items@0.1"] = {
    **_SD35, "metric": "ocr_b_acc",
    # 05 section 6.1, R15-1: another OCR engine (docTR: other detector, recognizer, framework and vendor) than the
    # reward's (PaddleOCR PP-OCRv4) and another formula (best-matching substring instead of the whole recognized
    # text), so T3; the same construct (legible rendering of the quoted text), so construct-aligned, stated in the
    # registration and the report.
    "tier": "T3",
    "construct_aligned": {
        "reward": "ocr_accuracy_paddleocr",
        "statement": ("ocr_b_acc measures the construct of the training reward ocr_accuracy_paddleocr (the prompt's "
                      "quoted text is legibly rendered) with another OCR engine (docTR db_resnet50 + crnn_vgg16_bn, not "
                      "PaddleOCR PP-OCRv4) and another formula (edit distance of the best-matching substring of the "
                      "recognized text, not of the whole text; chosen after the smoke baseline, before registration): a "
                      "pass shows that the reward's gain transfers to an independent OCR engine, not that the images are "
                      "better otherwise (05 section 6.1, R15-1)")},
    "scorer": {"name": "ocr_accuracy_doctr",
               "version": ('eval_images@0.1 engine docTR db_resnet50+crnn_vgg16_bn python-doctr 1.1.0 '
                           '{"assume_straight_pages": true, "det_arch": "db_resnet50", "join": "words in reading '
                           'order, no separator", "preserve_aspect_ratio": true, "reco_arch": "crnn_vgg16_bn", '
                           '"symmetric_pad": true} approximate-substring edit distance (Sellers DP), spaces removed, '
                           'lower case'),
               "weights_sha256": "sha256:7380ad4e2df54bd2b610b1e19e7cde2b4a35d06c9fa00eda70f7f6b23947004c",
               "code_sha256": EVAL_IMAGES_SD35_SHA256}}
EVALUATOR_ROWS["t2i.sd35_ocr_paddle_items@0.1"] = {
    **_SD35, "metric": "ocr_a_acc", "tier": "T0",       # the training reward itself: supporting, never deciding
    "scorer": {"name": "ocr_accuracy_paddleocr",
               "version": ("eval_images@0.1 engine PaddleOCR PP-OCRv4 paddleocr 2.10.0 paddlepaddle 3.0.0 "
                           "PP-OCRv4_mobile_det@3cc09f3a5b424e8e010abc7a4271aea12999c2f7 "
                           "en_PP-OCRv4_mobile_rec@f97b62fdc0eb71c689393a19a4b21baa1795c9ab"),
               "weights_sha256": "sha256:7783ef1516fc76bcd590b6fac074b4942f15a9111c9d55506256a8067d13f07a",
               "code_sha256": EVAL_IMAGES_SD35_SHA256}}

PLUGIN = {"adapters": {"flow_grpo_sd35": "gatedtrain_recipe_flow_grpo_sd35_ocr.adapter:FlowGrpoSd35Adapter"},
          "trainer_rows": TRAINER_ROWS,
          "evaluator_rows": EVALUATOR_ROWS}
