"""Flow-GRPO SD3.5 Medium + LoRA adapter (area "image generation, SD3.5": the Flow-GRPO port `train_flow_grpo.py` on
diffusers 0.42.1 + peft 0.21.2, OCR text rendering; MULTIMODAL-SPEC.md section 6 row 2b and runtime/README.md
"Decisions taken"; tests/README.md section 11.22, EXT-160 to EXT-178).

MM-12: a pure-Python class that imports no framework, torch or pickle loader. It reads JSON, JSONL, the text
`env-freeze.txt` and safetensors headers (standard library `struct` and `json`), and hashes everything else.

Run layout, read from the runtime's code `runtime/train_flow_grpo.py` (and its
README.md "Outputs of `train_flow_grpo.py`", "Seeds", "What gatedtrain needs"; read 2026-11-13; where the two differ the
code is followed, docs/findings.md). The layout is area 3's (`ddpo_sd15`, whose adapter this one extends):

- `save_cut` (train_flow_grpo.py:602-633) writes `checkpoint-<E>/` (E the 1-based epoch) through
  `.checkpoint-<E>.partial/`: the LoRA weights (header canonicalized), `adapter_config.json` (`weights` raw or ema),
  `sampler.json` (the deployment sampler: Euler ODE, 10 steps, CFG 4.5, 512 px, weight precisions), `base_ref.json`,
  `trainer_state.json`, then `cut.json` last with area 3's keys plus `samples_filtered` and `ema`;
- `samples.jsonl` (:473-479): one line per generated sample after the epoch's rewards, `{epoch, sample_index,
  sample_id, prompt_id, prompt, target, member, seed, ocr_text, ocr_error, ocr_acc, reward, attempt, utc}`;
- `epoch_plan.jsonl` (:480-503): one line per epoch before its first optimizer step, `{epoch, kept, filtered,
  reincluded, filter_reason: "zero_advantage", steps: [first, last] or []}`. flow_grpo's zero-advantage filter
  (train_sd3.py:783-823) drops every sample whose advantage is 0 (a prompt group whose OCR rewards are all equal) and
  pads the kept count to a multiple of `num_batches_per_epoch` with randomly re-included zero-advantage samples;
- `train_steps.jsonl` (:563-564): one line per optimizer step, before the step, `{step, epoch, inner, sample_ids,
  utc}`. Each inner epoch applies each kept sample once; a filtered sample is in no step; an epoch without kept
  samples has no step;
- `run_args.json` (:397-427): area 3's keys plus `n_train_prompts`, `sampler_sha256` (compact JSON),
  `training_sampler`, `reward`; `env-freeze.txt` lines plus `# flow_grpo algorithm reference yifan123/flow_grpo
  <40 hex> (ported)`.

Consumption (MM-24, MM-28; 05 section 12.3): kept samples give `consumed` records (one per applying step), filtered
samples one `filtered` record with reason `zero_advantage` and no step. Identity (MM-17, MM-18) as area 3, except that
the base's manifest leaves out the base directory's own `.cache/` (Hugging Face download metadata; decision "Area 5
decisions", not published: the model directory stays untouched, its byte-identical `*.fp16.*` duplicates are hashed).
"""

from __future__ import annotations

import hashlib
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from gatedtrain.canonical import canonical_json, file_sha256, manifest_entries, manifest_hash, sha256_bytes, text_lines
from gatedtrain.errors import Problem
from gatedtrain.ext.registry import TRAINER_ROWS
from gatedtrain.ext.sdk import (compact_sha256, derive_seed, is_int as _int, is_num as _num, plain_file as _plain_file,
                         read_cut_marker, read_json as _read_json, read_jsonl, read_run_args, runtime_argv_problems,
                         safetensors_problem, single_shard as _shard)
from gatedtrain.families.diffusers_lora import CHECKPOINT_DIR_RE, CUT_MARKER, DiffusersLoraAdapter
from gatedtrain.ext.sdk import MAX_SEED
from gatedtrain.families.flow_grpo import CUT_COMPLETENESS, FlowGrpoAdapter, _str_list, target_text

NAME = "flow_grpo_sd35"
LIBRARY = "the Flow-GRPO port train_flow_grpo.py on diffusers + peft (cuts by its save_cut, cut.json written last)"
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"
CUT_FILES = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json", "trainer_state.json")
COMPONENTS = tuple(TRAINER_ROWS[NAME]["components"])          # train_flow_grpo.py DEPLOYABLE (EXT-162)
assert COMPONENTS == FlowGrpoAdapter.COMPONENTS, "the registry row and the flow_grpo family disagree on DEPLOYABLE"
FRAMEWORK_PACKAGE = "diffusers"
PINS = tuple(TRAINER_ROWS[NAME]["pins"])
REFERENCE_LINE_RE = re.compile(r"^# flow_grpo algorithm reference yifan123/flow_grpo [0-9a-f]{40} \(ported\)$")
FREEZE_FILE = "env-freeze.txt"
RUNTIME_FILE = "env-runtime.json"
SAMPLES = "samples.jsonl"
PLAN = "epoch_plan.jsonl"
STEPS = "train_steps.jsonl"
FILTER_REASON = "zero_advantage"                              # 05 section 12.3; train_flow_grpo.py :502
# The SD3.5 Medium directory (README "What gatedtrain needs" item 2).
BASE_REQUIRED = ("model_index.json", "scheduler/scheduler_config.json", "transformer/config.json", "vae/config.json",
                 "text_encoder/config.json", "text_encoder_2/config.json", "text_encoder_3/config.json",
                 "tokenizer/tokenizer_config.json", "tokenizer_2/tokenizer_config.json",
                 "tokenizer_3/tokenizer_config.json")
BASE_WEIGHT_DIRS = ("transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3")
# runtime/README.md "Decisions taken" 4: the base hash covers the model directory without its own `.cache/`
# (download metadata and lock files of the Hugging Face download); nothing else is left out.
BASE_EXCLUDES = (".cache",)
# The SHARED SAMPLER block of train_flow_grpo.py and eval_images.py (byte-identical; README "Measured hashes"): the
# evaluator's sampler code, which every cut's `sampler_code_sha256` must name (README item 5; EXT-169). The
# published 2026-11-14 run recorded the block before its edit-distance import moved from python-Levenshtein to
# rapidfuzz (sha256:83133e24...; RESULT.md); only this pin is accepted.
SAMPLER_CODE_SHA256 = "sha256:bba7ecd54c681151f4f7514cde9844431b5706623ad2233bfd8034a6483cbb6f"
TRAIN_SCRIPT = "train_flow_grpo.py"
EVAL_SCRIPT = "eval_images.py"
DEFAULT_BASE_REL = "models/stable-diffusion-3.5-medium"       # both scripts: DEFAULT_BASE = HERE / this
PROFILES = ("smoke", "full")
# argparse options of the two scripts (parse_args; train_flow_grpo.py :640-655, eval_images.py :708-734): option ->
# takes a value. argparse accepts a unique prefix of an option (allow_abbrev), so a token is resolved the same way.
TRAIN_OPTIONS = {"--profile": True, "--out": True, "--base": True, "--prompts": True, "--seed": True, "--device": True,
                 "--epochs": True, "--no-verify-base": False, "--train-batch": True,
                 "--no-gradient-checkpointing": False, "--beta": True, "--ema": False}
EVAL_OPTIONS = {"--make-items": False, "--n-items": True, "--split-seed": True, "--checkpoint": True,
                "--base-only": False, "--base": True, "--items": True, "--out": True, "--eval-seeds": True,
                "--model-hash": True, "--base-model-hash": True, "--item-set-hash": True, "--limit": True,
                "--batch-size": True, "--device": True, "--dtype": True, "--encoder-dtype": True,
                "--ocr-cpu-threads": True, "--num-steps": True, "--noise-level": True, "--guidance-scale": True,
                "--resolution": True, "--max-sequence-length": True}

CHECKPOINT_COMPLETENESS = {
    "library": LIBRARY,
    **CUT_COMPLETENESS,
    "sources": ["runtime/train_flow_grpo.py:602-633 save_cut (cut.json last, "
                "os.replace)",
                "runtime/train_flow_grpo.py:125-128 CUT_FILES, DEPLOYABLE",
                "runtime/README.md \"Outputs of train_flow_grpo.py\", cut layout"],
    "verified": True,                                # the layout was read from the CPU sanity run's cut (README)
}
SEED_DERIVATION = ("train_flow_grpo.py derive_seed(*parts) = first 8 bytes of SHA-256(':'.join(parts)) >> 1: epoch e's "
                   "prompts from numpy PCG64(derive_seed('prompts', seed, e)) without replacement, sample i's noise "
                   "from a CPU torch.Generator seeded with derive_seed('sample', seed, e, i) (recorded per sample in "
                   "samples.jsonl and in the ledger's seed_inputs), the zero-advantage padding from "
                   "derive_seed('mask', seed, e), the training order from derive_seed('perm', seed, e, inner), the LoRA "
                   "init from derive_seed('torch', seed). The training seed is run_args.json `seed`, which must equal "
                   "the launch record's training_seed and every cut's cut.json `seed`; 05 section 14 H(run_seed, step, "
                   "data_ref, sample_index) is not claimed.")


class FlowGrpoSd35Adapter(FlowGrpoAdapter):
    """The MM-12 interface for Flow-GRPO + LoRA runs on SD3.5 Medium (EXT-160 to EXT-178). Cuts, identity, exact
    overlap and the deployment-sampler check follow area 3's layout (`DiffusersLoraAdapter`); the base directory, its
    `.cache/` exclusion, the epoch plan with `filtered` outcomes, the pins and the run checks are this area's."""

    DEFAULT_BASE_REL = DEFAULT_BASE_REL
    EVAL_OPTIONS = EVAL_OPTIONS
    EVAL_SCRIPT = EVAL_SCRIPT
    PLAN = PLAN
    PROFILES = PROFILES
    RUNTIME_FILE = RUNTIME_FILE
    SAMPLES = SAMPLES
    TRAIN_OPTIONS = TRAIN_OPTIONS
    TRAIN_SCRIPT = TRAIN_SCRIPT

    name = NAME
    trainer_id = TRAINER_ROWS[NAME]["uses"][0]
    evaluator_ids = tuple(TRAINER_ROWS[NAME]["evaluators"])
    checkpoint_completeness = CHECKPOINT_COMPLETENESS
    seed_derivation = SEED_DERIVATION
    binds_spec = bool(TRAINER_ROWS[NAME]["spec_binding"])
    # Area facts a sibling adapter on the same run layout overrides (`flow_grpo_wan`, EXT-228): the base directory, the
    # freeze pins, the pinned SHARED SAMPLER block and where it lives, the `sample_id` domain of samples.jsonl.
    base_required = BASE_REQUIRED
    base_weight_dirs = BASE_WEIGHT_DIRS
    base_kind = "an SD3.5 Medium directory"
    pins = PINS
    sampler_code_sha256 = SAMPLER_CODE_SHA256
    sampler_code_source = "eval_images.py SHARED SAMPLER; README item 5"
    sample_id_domain = "flow-grpo-sd35"

    # --- the base directory and cuts (MM-17, MM-18, MM-20; EXT-161, EXT-162) ---


    # --- consumption (MM-24, MM-27, MM-28; EXT-165, EXT-166, EXT-167) ---


    # Hooks of `_ledger_plan` (EXT-228: a sibling adapter on the same layout overrides them; this area's rules here).

    def _settings_problems(self, s: dict) -> list[str]:
        """Further run_args.json `settings` the consumption rules need (none for this area)."""
        return []

    def _reward_field_problems(self, x: dict, n: int, s: dict, ocr_reward: bool) -> list[str]:
        """A samples.jsonl line's reward fields: reward = the OCR accuracy exactly when the run's reward is `ocr`
        (train_flow_grpo.py :466-470; json writes both floats with repr, which round-trips); the error path scores 0
        with no text (:398-401)."""
        reward, acc = x.get("reward"), x.get("ocr_acc")
        out = []
        if ocr_reward and reward != acc:
            out.append(f"{self.SAMPLES} line {n}: reward {reward!r} is not ocr_acc {acc!r} (settings.reward ocr)")
        if x.get("ocr_error") is not None and (acc != 0 or x.get("ocr_text") is not None):
            out.append(f"{self.SAMPLES} line {n}: an OCR error scores ocr_acc 0 with no ocr_text (train_flow_grpo.py "
                       ":398-401)")
        return out

    def _epoch_prompt_problems(self, e: int, got: list[dict], spe: int, group: int, s: dict,
                               rows: list[dict]) -> list[str]:
        """Sample i is member i % G of prompt group i // G (train_flow_grpo.py :441); an epoch draws its prompts
        without replacement (:354-355), so the groups' prompts differ."""
        out, pids = [], []
        for g0 in range(0, spe, group):
            members = {x["prompt_id"] for x in got[g0:g0 + group]}
            if len(members) != 1:
                out.append(f"{self.SAMPLES} epoch {e} prompt group {g0 // group} (sample_index {g0} .. {g0 + group - 1}) "
                           f"has prompts {sorted(members)[:3]}, not one")
            pids.append(sorted(members)[0])
        if len(set(pids)) != len(pids):
            out.append(f"{self.SAMPLES} epoch {e} draws a prompt for two groups (prompts are drawn without replacement)")
        return out


    # --- target-text overlap (MM-11; README item 7; EXT-176) ---


    # --- pins and checks (MM-21, MM-29, MM-33; EXT-163, EXT-164, EXT-169) ---


    BASE_FIX = ("train from the spec's SD3.5 Medium directory inside a read-only mount (train_flow_grpo.py --base, "
                "default <area5-sd35>/models/stable-diffusion-3.5-medium, re-hashed at start), and record that mount in "
                "the launch record")
    DATA_FIX = ("train on the spec's prompt shard inside a read-only mount (train_flow_grpo.py --prompts, recorded in "
                "run_args.json train_file, train_file_sha256 and n_train_prompts)")

    def argv_problems(self, argv: Any, *, profile: str | None = None) -> list[str]:
        """EXT-173 (as EXT-133, EXT-150, EXT-156), ext-launch and --recheck. argv[1] is train_flow_grpo.py or
        eval_images.py of the runtime directory (the pinned interpreter's). eval_images.py loads `--base` and `--items`
        without hashing them, so an evaluation may use only the defaults inside the read-only runtime mount; `--limit`
        is refused under `--profile full`. train_flow_grpo.py: `--no-verify-base` is refused (collect refuses a run
        that did not re-hash its base), `--ema` is refused (decision "Area 5 decisions", not published: EMA off, cuts
        hold the raw LoRA weights) and `--profile` is smoke or full (cpu-sanity is a host check, never collected). Options are
        resolved as argparse resolves them (a unique prefix counts)."""
        out, script, parsed = runtime_argv_problems(argv, profile=profile,
                                                    options_of={self.TRAIN_SCRIPT: self.TRAIN_OPTIONS, self.EVAL_SCRIPT: self.EVAL_OPTIONS},
                                                    eval_script=self.EVAL_SCRIPT, default_base_rel=self.DEFAULT_BASE_REL,
                                                    rule="EXT-173")
        if script == self.TRAIN_SCRIPT:
            for opt, value, tok in parsed:
                if opt == "--no-verify-base":
                    out.append(f"{self.TRAIN_SCRIPT} {tok} skips re-hashing the base, and ext-collect refuses such a run "
                               "(run_args.json base_ref.verified false, EXT-163)")
                if opt == "--ema":
                    out.append(f"{self.TRAIN_SCRIPT} {tok} saves EMA weights in the cuts; the area runs with EMA off (decision "
                               "\"Area 5 decisions\", not published: a cut holds the raw LoRA weights; EXT-173)")
                if opt == "--profile" and value not in self.PROFILES:
                    out.append(f"{self.TRAIN_SCRIPT} --profile {value!r} is not smoke or full: cpu-sanity is a host check "
                               "and is never collected (EXT-173)")
        return out


    def _reward_setting_problems(self, s: dict) -> list[Problem]:
        """run_args.json settings.reward is the area's collected reward (EXT-STAGE)."""
        if s.get("reward") != "ocr":
            return [Problem("EXT-STAGE", "--stage-out", f"run_args.json settings.reward is {s.get('reward')!r}, not "
                            "'ocr' (the OCR accuracy of engine A, decision \"Area 5 decisions\", not published; the "
                            "cpu-sanity brightness term is never collected)", "collect a smoke or full run")]
        return []
