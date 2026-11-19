# Area "image generation, SD3.5": Flow-GRPO + LoRA on Stable Diffusion 3.5 Medium, OCR text rendering (prototype, unpromotable)

Multimodal prototype, gatedtrain's `MULTIMODAL-SPEC.md` §6 row 2b ("Image generation | flow_grpo | SD3.5 Medium + LoRA,
512 px, 10 steps"). Runtime
kind `venv_in_container` (MM-32), as areas 1, 3 and 4: a pinned CUDA base image plus this directory's host venv,
built from the committed `uv.lock` and mounted read-only. Nothing here starts a container or uses the GPU; the
commands below run through gatedtrain's `ext-launch` and its `gpu-step.sh` wrapper (MM-31). Every number marked
**estimate** was unmeasured on the GPU when this was written; the smoke run measures it.

Status (2026-11-13): runtime prepared, CPU sanity checks passed (section "CPU sanity check"). Proposals are marked
"proposal"; the questions are collected in "Open questions", each marked **[maintainers]** or **[routine]**;
"Decisions taken" at the end records how they were settled.

## Pins

| Item | Value |
|---|---|
| Base image | the area-1 image, `docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266` (as areas 3 and 4) |
| Python | CPython 3.12.16, `<python-root>/cpython-3.12.16-linux-x86_64-gnu` (shared, unchanged) |
| Lock | `uv.lock` (109 packages, `exclude-newer = 2026-10-29`, Linux x86_64), sha256 `502c7ce7…caecda1` (the published run used `8243b02f…ece20d7`, 110 packages: the project was renamed since and python-Levenshtein removed, every other version unchanged; `RESULT.md`) |
| Venv | `.venv` (`uv sync --frozen --compile-bytecode`, copy link mode, 7.1 GB, 50,835 files) |
| Framework | diffusers 0.42.1, peft 0.21.2, transformers 5.21.0, accelerate 1.15.0, torch 2.15.0 (PyPI build, CUDA 13.2 runtime wheels), torchvision 0.29.2, safetensors 0.8.0, pillow 13.0.0, numpy 2.5.4 — **identical to area 3**; huggingface-hub **1.33.0** (area 3: 2.3.0; python-doctr requires `<2.0`) |
| OCR engine A (reward) | paddleocr 2.10.0 (the last 2.x: flow_grpo's API, without 2.9.1's `numpy<2` pin), paddlepaddle **3.0.0** CPU wheel (3.3.1 fails on the CPU predictor, below), rapidfuzz 3.14.6 for the edit distance (the published run: python-Levenshtein 0.27.5, a wrapper of the same rapidfuzz; `RESULT.md`) |
| OCR engine B (deciding) | python-doctr 1.1.0 (PyTorch), opencv-python 5.0.0.93 |
| Algorithm reference | yifan123/flow_grpo `879042cf5707f8b90daa98d147d7deac2317c5da` (2026-05-07, MIT), a read-only clone outside the mounted runtime directory; not installed, not imported |

Rebuild (host, no GPU): `UV_CACHE_DIR=<scratch>/uv-cache
UV_PROJECT_ENVIRONMENT=$PWD/.venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never uv sync --frozen
--compile-bytecode --python <python-root>/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12`
(the uv cache lives outside the area directory; no global cache was written). CUDA: same torch build and fallback as
area 3 README "Pins".

## Trainer choice (a maintainers' decision: row 2b names flow_grpo)

### Does flow_grpo run as-is? No (verified against the clone and the current stack)

| Check | Finding (file:line of `879042cf`) |
|---|---|
| Pins | `setup.py:9-55`: torch 2.6.0, torchvision 0.21.0, **diffusers 0.33.1**, **transformers 4.40.0**, **accelerate 1.4.0**, **peft 0.10.0**, numpy 1.26.4, deepspeed 0.16.4, bitsandbytes, xformers, wandb 0.18.7, huggingface-hub 0.29.1; README.md:185-189 adds paddlepaddle-gpu 2.6.2 and **paddleocr 2.9.1, which requires `numpy<2.0`, `imgaug`, `albumentations==1.4.10`** (PyPI metadata) — none of it installs beside this prototype's framework pins (numpy 2.5.4, diffusers 0.42.1, transformers 5.21.0) |
| Python 3.12 | `config/grpo.py:2` and `:5` use `imp.load_source`; `imp` was removed in Python 3.12 (measured on the shared CPython 3.12.16: `ModuleNotFoundError: No module named 'imp'`) |
| Network | `scripts/train_sd3.py:24`, `:363` import and `wandb.init` unconditionally; `PaddleOCR(...)` downloads its models from paddleocr.bj.bcebos.com at construction (paddleocr ppocr/utils/network.py:106-128); `StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)` with the Hub id (`train_sd3.py:376-378`) — the container runs with network `none` |
| Current diffusers | the diffusers pieces it calls still exist in 0.42.1 (`randn_tensor` utils/torch_utils.py:61, `retrieve_timesteps` pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py:89, `FlowMatchEulerDiscreteScheduler.index_for_timestep` :393), so the algorithm code itself is portable; the blockers are the environment, the network and memory |
| **Memory on 24 GB** | `train_sd3.py:376-378` loads the pipeline **without `torch_dtype`** (fp32); `:409-415` keeps **all three text encoders on the GPU** (fp16) for the whole run (they encode every sampling batch, `:614-621`, and the eval set, `:233-240`) and moves the transformer in **fp32** (`:415`); accelerate's fp16 mixed precision only autocasts. Weights alone: transformer 2.47 B params fp32 ≈ 9.9 GB, T5-XXL fp16 ≈ 9.5 GB (transformers keeps T5's `wo` in fp32, `_keep_in_fp32_modules`, modeling_t5.py:537), CLIP-G 1.4 GB, CLIP-L 0.25 GB, VAE fp32 0.33 GB → **≈ 21.5 GB before LoRA, Adam, the stored chains and activations**; the 1-GPU config trains micro-batches of 8 with CFG (transformer batch 16) without activation checkpointing (`config/grpo.py:460-461`, `config/base.py:24`). It cannot fit a 24 GB card (estimate from file sizes; flow_grpo's "1gpu" config targets an 80 GB GPU) |
| EMA with a short run | `train_sd3.py:444-445`, `:959-960`, `ema.py:35-53`: decay 0.9, updated only when `(step + 1) % 8 == 0`, and the saved LoRA is the EMA (`:323-332`). A 2-epoch smoke (4 optimizer steps) would save the **initial LoRA, identical to the base**; a 10-epoch run (20 steps) saves mostly the weights of step 15 |

### Options

| Option | What it takes | Verdict |
|---|---|---|
| (a) **Minimal port** on the area-3 stack (`train_flow_grpo.py`, line by line with file:line citations, `general_ocr_sd3_1gpu` hyperparameters) | same pins and loader conventions as area 3 (whose port was accepted); the memory fixes are part of the port (below); one venv for trainer, reward and both OCR engines | **recommended (proposal)** |
| (b) Pinned legacy environment running flow_grpo's own `train_sd3.py` | a second interpreter (Python ≤ 3.11 for `imp`), torch 2.6.0 cu124 wheels, diffusers 0.33.1, transformers 4.40.0, peft 0.10.0, numpy 1.26, paddleocr 2.9.1 + paddlepaddle 2.6, wandb offline; and still **patches** for the 24 GB card (fp16 transformer or encoder offload, smaller micro-batches, checkpointing), for the observed consumption log, recorded seeds, atomic cuts and local OCR models — i.e. a fork of `train_sd3.py` on 2025 libraries | not recommended: more moving parts, the same amount of patching, and a second framework stack the adapter would have to pin |

Evidence for (a): area 3 set the precedent (ddpo-sd15's decision 1); the diffusers APIs flow_grpo needs exist
in 0.42.1 (table above); the CPU checks below run the port end to end on the current stack. **The spec text
names flow_grpo, so this is a deviation (open question 1; accepted, see "Decisions taken").** Cost: the "framework" pinned
by MM-33 is diffusers + peft plus this script (its SHA-256), with flow_grpo `879042cf` as the algorithm reference line
in `env-freeze.txt`.

### Changes against flow_grpo (all in `train_flow_grpo.py`)

Same: the SDE step with log-probability (`sde`, noise level 0.7), 10 sampling steps with CFG 4.5 at 512 px, T5 at
128 tokens, training on the first 9 of 10 steps (`timestep_fraction` 0.99) with CFG, per-prompt advantages with the
epoch's global std, advantage clip 5, PPO clip 1e-4, KL β 0.04 against the adapter-disabled transformer, AdamW lr
3e-4 (β 0.9/0.999, wd 1e-4, ε 1e-8), grad-norm clip 1.0, LoRA r 32 / α 64 gaussian on the eight attention
projections of the joint blocks, 8 prompts × 8 images per epoch, two optimizer steps of 32 samples, the
zero-advantage filter with its padding rule, the OCR reward formula, VAE in fp32, fp16 mixed precision with a
gradient scaler. Different, with reason:

| Change | Reason |
|---|---|
| Text encoders run once at start for every prompt of the run (the prompt schedule is fixed by the seed), then are freed | T5-XXL ≈ 9.5 GB; flow_grpo keeps all three resident (memory table). Embeddings are a deterministic function of the base and the prompt, so no separate artifact enters the identity |
| Frozen transformer stored in fp16 and run under `torch.autocast(fp16)`; LoRA parameters fp32 | flow_grpo stores fp32 weights and autocasts to fp16 (accelerate); same matmul precision, half the memory |
| Training micro-batch `train_batch` (2) instead of 8, gradient checkpointing on | memory; the gradient is unchanged: the loss is summed per sample and divided by (flow_grpo micro-batch size × 4 × 9), exactly flow_grpo's mean over 32 samples × 9 steps per optimizer step |
| EMA off by default (`--ema` restores flow_grpo's decay-0.9 EMA, cuts then hold the EMA) | with 4-20 optimizer steps the EMA saves the initial or a lagging LoRA (table above); proposal, open question 2 |
| Prompts: 8 per epoch drawn without replacement from the registered shard; each sampling batch is one prompt's 8 images | flow_grpo's `DistributedKRepeatSampler` draws one prompt per batch with replacement from torch's global RNG (`train_sd3.py:79-122`); consumption by registered shard rows |
| Per-sample noise seed, CPU `torch.Generator` (initial latents and every step's noise) | recorded seeds (MM-29), device-independent noise, common random numbers at evaluation |
| Zero-advantage padding and training order from seeded CPU generators | flow_grpo uses `torch.randperm` on the global RNG (`train_sd3.py:817`, `:835`) |
| OCR engine A built from paddleocr 2.10.0's `TextSystem` with local models; RGB array passed as-is (as flow_grpo, which feeds RGB to an engine expecting BGR); detector `PP-OCRv4_mobile_det` instead of 2.9.1's default `en_PP-OCRv3_det` | offline; the 2.9.1 default detector is only on paddleocr.bj.bcebos.com (not an allowed source); `PP-OCRv4_mobile_det` + `en_PP-OCRv4_mobile_rec` is PaddleOCR 3.7.0's own pairing for `lang="en"` (paddleocr/_pipelines/ocr.py:394-397); the recognizer is the same model as 2.9.1's `en_PP-OCRv4_rec` (dictionary identical, 95 symbols) |
| Paddle's `Config.enable_memory_optim()` disabled for the OCR predictor | paddle 3.0.0 rejects it for PIR (`.json`) models on the CPU ("Not find predictor_id 0 and pass_name memory_optimize_pass"); a memory-planning switch with no effect on results. paddle 3.3.1 additionally fails at run time in oneDNN (`ConvertPirAttribute2RuntimeAttribute not support`), hence the 3.0.0 pin |
| `samples.jsonl` at sampling time, `epoch_plan.jsonl` (kept / filtered) and `train_steps.jsonl` before each optimizer step | observed consumption (MM-24, MM-27, MM-28) |
| Cuts `checkpoint-<epoch>/` via `.partial` + rename, `cut.json` last, safetensors header canonicalized | atomic, complete, reproducible hashes (MM-20), as area 3 |
| No wandb, no in-training evaluation, no `save_freq`/`eval_freq` of 60 epochs | no network; evaluation is the separate per-item evaluator |
| Deployment sampler: Euler ODE (noise level 0) at **10** steps, CFG 4.5 | flow_grpo evaluates with noise level 0 at 40 steps (`train_sd3.py:258`, `config/grpo.py:451`); row 2b says 10 steps and 40 would quadruple evaluation cost; proposal, open question 3 |

## Models (`fetch_models.py`, `MODELS.json`)

- **SD3.5 Medium** (gated: accept its license on the Hub and download it yourself; nothing here downloads it): `models/stable-diffusion-3.5-medium/`,
  `stabilityai/stable-diffusion-3.5-medium` @ `b940f670f0eda2d07fbb75229e779da1ad11eb80` (Stability AI Community
  License). `fetch_models.py` only reads it: every file outside `.cache/` hashed (37 files, 27.4 GB); each file's
  download metadata names that commit, and every LFS file's SHA-256 equals its etag. Nothing in `models/` was
  written, moved or deleted.
- **Byte-identical duplicates (not deleted):** `text_encoder/model.fp16.safetensors` = `text_encoder/model.safetensors`
  (247 MB, `71e183d1…`), `text_encoder_2/model.fp16.safetensors` = `model.safetensors` (1.39 GB, `ec310df2…`),
  `text_encoder_3/model.fp16-0000{1,2}-of-00002.safetensors` = `model-0000{1,2}-of-00002.safetensors` (4.99 + 4.53
  GB, `4f2751ce…`, `f6315453…`): the default files are already fp16. **11.16 GB** are duplicates.
  **Recommendation:** keep the default (non-variant) files — the loaders here call `from_pretrained` without
  `variant`, which reads `model.safetensors` / `model.safetensors.index.json`, and the transformer and VAE exist
  only as default files — and remove the four `*.fp16.*` weight files plus `model.safetensors.index.fp16.json`.
  Whatever is decided must be settled **before** the smoke run, because the base manifest hash covers every
  file of the directory (open question 4). `tokenizer/vocab.json` = `tokenizer_2/vocab.json` is a legitimate
  duplicate (keep).
- **`.cache/huggingface/`** inside the base directory (download metadata and lock files) is excluded from
  `MODELS.json`; the gatedtrain base manifest must exclude it too, or it is moved out (open question 4).
- **OCR engine A**: `ocr-models/paddleocr/PP-OCRv4_mobile_det/` (`PaddlePaddle/PP-OCRv4_mobile_det` @
  `3cc09f3a5b424e8e010abc7a4271aea12999c2f7`) and `ocr-models/paddleocr/en_PP-OCRv4_mobile_rec/`
  (`PaddlePaddle/en_PP-OCRv4_mobile_rec` @ `f97b62fdc0eb71c689393a19a4b21baa1795c9ab`), Apache-2.0, `gated: false`,
  Paddle 3 inference format (`inference.json`, `inference.pdiparams`, `inference.yml`), downloaded with
  `token=False`, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, Xet off, `HF_TOKEN` removed from the environment, a fresh
  `HF_HOME` outside the area; download metadata in `ocr-models/hf-meta/`.
- **OCR engine B**: `ocr-models/doctr/db_resnet50.safetensors` and `crnn_vgg16_bn.safetensors`, converted once
  (`torch.load(weights_only=True)`) from the mindee/doctr GitHub release assets `v0.7.0/db_resnet50-79bd7d70.pt`
  (sha256 `79bd7d70…40ca483`) and `v0.12.0/crnn_vgg16_bn-0417f351.pt` (`0417f351…ddac087`) — the files python-doctr
  1.1.0's `default_cfgs` name; Apache-2.0. The `.pt` pickles stay outside the runtime directory (`--download-dir`, never mounted).
- `MODELS.json` (sha256 `92ae292e…0ad823`) records repo, revision and per-file SHA-256 of all four;
  `fetch_models.py --verify` re-hashes.

## Reward and deciding metric (proposal)

| # | Reward (training) | Deciding metric | Tier by 05 §6.1 | Feasibility | Verdict |
|---|---|---|---|---|---|
| 1 | **OCR accuracy, engine A = PaddleOCR PP-OCRv4** (flow_grpo's `ocr` reward, `general_ocr_sd3_1gpu`) | **OCR accuracy, engine B = docTR** (db_resnet50 + crnn_vgg16_bn), approximate-substring formula (2026-11-14, below) | **T3, construct-aligned (R15-1)**: "OCR edit distance with a different OCR engine" is 05 §6.1's own T3 example; tool = engine (R15-1): different detector (PP-OCRv4 DB-mobile vs DBNet-ResNet50), recognizer (SVTR-LCNet CTC vs VGG16-BiLSTM CTC), framework (Paddle vs PyTorch), vendor and training data | both ungated, Apache-2.0, offline; A on CPU 0.1-0.4 s/image, B on GPU (CPU 1.4 s/image) | **recommended** |
| 2 | JPEG compressibility (as area 3) | lossless WebP bpp | T3 construct-aligned (area 3's decision) | trivial | not recommended: repeats area 3, ignores flow_grpo's text-rendering task |
| 3 | PickScore or LAION aesthetic | a different-backbone learned scorer | T2 at best, needs the §6.2 transfer test with human raters and an agreement-rate validity check (G-21, C6) | needs human labels | not recommended for the prototype |

GenEval-style counting (detector reward) was considered: Flow-GRPO's GenEval reward runs as a separate
`reward-server` (README.md:178-181, mmdet stack) and a T3 metric would need a second detector — more cost for the
same plumbing.

**Formula, engine A (reward and `ocr_a_acc`; flow_grpo ocr.py:32, :43-60).** Target = the first double-quoted span
of the prompt; recognized text = the engine's text lines concatenated (lines with a positive confidence, what
flow_grpo's reward sees); spaces removed and lower case on both sides; score 1 if the target is a substring of the
recognized text, else 1 − min(Levenshtein, len(target)) / len(target). Range [0, 1], higher is better.

**Formula, engine B (deciding `ocr_b_acc`; chosen after the smoke baseline, before registration).** Same target and
normalization; recognized text = docTR's words joined in reading order; score = 1 − min over substrings s of the
recognized text of lev(s, target) / len(target) (Sellers' approximate-substring edit distance, `eval_images.py`
`ocr_substring_accuracy`, O(n·m)); an empty recognized text scores 0; range [0, 1]. Reason ("Decisions taken",
smoke finding of 2026-11-14): on the 40-item GPU smoke base evaluation docTR also read
background texture as text, and the whole-text edit distance clipped `ocr_b_acc` to 0 on 52.5 % of items (MM-38
fail); the substring formula on the same 40 docTR texts gives share at 0 0.175, mean 0.412 (CPU recomputation,
verified against a brute-force reference). The reward and `ocr_a_acc` keep flow_grpo's formula.

**Construct-aligned, stated plainly (R15-1).** Both engines measure the same construct, "the quoted text is
legibly rendered": a pass shows that the reward's gain transfers to an independent OCR engine, not that images
improved otherwise. Since 2026-11-14 the reward and the deciding metric differ in the engine and in the formula
(whole-text against best-substring edit distance; the normalization is shared); under R15-1 the tool is the engine. Both engines are learned models whose training corpora are partly undisclosed (PP-OCR: synthetic plus
public scene-text sets; docTR: synthetic plus Mindee data); public scene-text sets may overlap. Neither saw this
area's prompts or images.

**Validity (G-21) and non-degeneracy.** A deterministic measurement (re-running gives the same score, measured
below); declared range [0, 1]; the CPU check at the real sampler (below) shows a non-degenerate base (per-item
scores 0, 0.25-0.31 and 1 on four items) and item-level agreement of the two engines. No labelled error rate is
proposed: the reference is the prompt's own target string (as area 4's references), not a judgment.

**Hacking risk to report.** Flow-GRPO's OCR task is known to trade image quality for legibility. Engine B does not
detect that; if a guard is wanted, a constraint clause (for example "the image must not be only text") would
need a judge with human-labelled validity (area 3 precedent: reported only). Not proposed now.

## Items, training shard and power (proposal)

- **Source:** flow_grpo `dataset/ocr/{train,test}.txt` at `879042cf` (MIT; copied to `data-source/ocr/`, sha256
  `2eb32ce1…fce5f` train, `4dd8caf3…d4dc` test, with the license). Every prompt has exactly one quoted target (2-27
  characters without spaces, median 15).
- **Registered items** (`items/items.jsonl`, sha256 `3884dcff…20f3e8`): **N = 400** test prompts,
  `numpy.default_rng(SeedSequence([0x5D35A5, 20261113])).permutation` over the 1,017 test prompts that do not occur
  in train.txt (one exact overlap dropped); `item` = `ocr-test-<line>`, fields `prompt`, `target`, `source_line`.
- **Training shard** (`prompts/train.jsonl`, sha256 `75361afd…585fb8`): the 19,649 unique train prompts minus every
  prompt whose normalized target equals an item's target (1,290 dropped: 440 of the 1,018 test targets, such as
  "Open", also occur in train) → **18,359 rows**, `prompt_id` = `ocr-train-<line>`. Disjoint from the items by
  prompt and by target text (checked at generation). Meta in `items/item_set_meta.json`, `prompts/prompts_meta.json`.
- **K = 1** (`eval_seeds = [11]`), one image per item: for a fixed image budget, more items beat more repeats
  (between-item variance dominates the item mean's variance); 400 images per arm.
- **Power** (paired, superiority δ 0, α 0.025 one-sided, power 0.8, one blocking clause): MDE = 2.802 σ_d / √400 =
  **0.140 σ_d** (0.035 at σ_d 0.25, 0.049 at 0.35, 0.063 at 0.45).
- **Sigma plan, after area 3's lesson** (the 2-epoch pilot gave 0.24 against the full run's 0.345): `sigma_plan:
  fixed`, set **before the full run** from the smoke as σ_plan = max(1.5 × ucb80(s_pilot), √2 × ucb80(s_base)),
  where s_pilot is the sd of paired differences between the smoke cut and the base on the smoke items and s_base the
  sd of the base's per-item scores; ucb80 with df = n − 1 (05 §3.6). The 1.5 factor covers area 3's observed 1.43
  ratio; √2 × s_base bounds σ_d whenever the two arms' item scores are non-negatively correlated (common random
  numbers make them positively correlated) and the candidate's spread does not exceed the base's. The smoke evaluates
  **40 items** per arm (not 20) to tighten both. N stays 400 (fixed by budget); the registered planning effect is the
  MDE at N = 400 under σ_plan. Alternatives: an internal pilot from a longer smoke (4 epochs, +0.12 GPU-h), or the
  area 3 rule (open question 6).

## Outputs of `train_flow_grpo.py` (`/out`)

| File | Written | Purpose |
|---|---|---|
| `env-freeze.txt` | first, before torch is imported | MM-33: sorted `name==version`, plus `# flow_grpo algorithm reference yifan123/flow_grpo 879042cf… (ported)` |
| `env-runtime.json` | first | Python, image `CUDA_VERSION`, lock SHA-256, `script_sha256` (EXT-83), `sampler_code_sha256` |
| `run_args.json` | before training | every setting, LoRA config, `seed` and the seed derivation, `base_model` + `base_ref`, `train_file` + `train_file_sha256`, deployment sampler (+ hash) and training sampler, reward engine and model revision, GPU, torch/CUDA, host RSS at start |
| `samples.jsonl` | per epoch after the rewards, fsync | one line per generated sample `{epoch, sample_index, sample_id, prompt_id, prompt, target, member, seed, ocr_text, ocr_error, ocr_acc, reward, attempt, utc}` |
| `epoch_plan.jsonl` | per epoch, before its first optimizer step | `{epoch, kept, filtered, reincluded, filter_reason: "zero_advantage", steps: [first, last]}` |
| `train_steps.jsonl` | before each optimizer step, fsync | `{step, epoch, inner, sample_ids}` |
| `train_log.jsonl` | after each optimizer step | policy loss, KL, approx_kl, clipfrac, grad norm, scaler scale |
| `memory.jsonl` | per epoch | seconds sampling/training, peak allocated/reserved GB, peak RSS, reward mean/sd, kept, zero-std groups |
| `attempts.jsonl` | start and end | attempts (a rerun moves earlier files to `preempted-attempt-<k>/` and restarts) |
| `checkpoint-<E>/` | after epoch E | the cuts |

**Cut layout** (as area 3): `checkpoint-<E>/` written into `.checkpoint-<E>.partial/` and renamed atomically:
`pytorch_lora_weights.safetensors` (peft keys without prefix, fp32, header canonicalized; LoraConfig in
`__metadata__.lora_adapter_metadata`), `adapter_config.json` (`weights: raw | ema`), `sampler.json` (the deployment
sampler: scheduler config, 10 steps, noise level 0, the 11 sigmas, CFG 4.5, 512 px, 128 T5 tokens, noise rule),
`base_ref.json` (repo, revision, per-file SHA-256, `files_sha256`), `trainer_state.json` (resume-only), then
**`cut.json` last** (`epoch`, `global_step`, `samples_consumed`, `samples_filtered`, `files`, `deployable`,
`script_sha256`, `sampler_code_sha256`, `seed`, `profile`, `ema`). Load: `SD3Transformer2DModel.load_lora_adapter(cut,
prefix=None, weight_name="pytorch_lora_weights.safetensors")`.

**Seeds (MM-29).** `derive_seed(*parts)` = first 8 bytes of SHA-256(":".join(parts)) >> 1. Epoch e prompts:
`Generator(PCG64(derive_seed('prompts', seed, e))).choice(18359, 8, replace=False)`; sample i of epoch e:
`derive_seed('sample', seed, e, i)`; zero-advantage padding `derive_seed('mask', seed, e)`; training order
`derive_seed('perm', seed, e, inner)`; LoRA init: torch global RNG seeded with `derive_seed('torch', seed)`.
Proposed run seed 5000. `sample_id` = SHA-256 of canonical `["flow-grpo-sd35", seed, attempt, epoch, sample_index,
prompt_id, noise_seed, member]`. Evaluation noise: `derive_seed('eval', eval_seed, item)`.

## Evaluator outputs (`eval_images.py`, `/out`)

`images/<item>/r<repeat>.png` (SHA-256 in `output_ref`), `scores.jsonl`, `summary.json` (means, sd, share at 0
and 1, A/B agreement, seconds per stage, RSS, tier proposals), `env-runtime.json`, and two MM-1/MM-2 arms with
exactly the MM-2 keys: `arm-ocr_b_acc/` (deciding, evaluator `t2i.sd35_ocr_doctr_items@0.1`) and `arm-ocr_a_acc/`
(supporting T0, `t2i.sd35_ocr_paddle_items@0.1`), each `items.jsonl` + `eval_meta.json` + hard-linked `images/`.
`sub_metrics` hold finite numbers only (MM-1; changed 2026-11-14): both engines' scores
(an engine that raised is left out, never null) and nonfinite; the target and both recognized texts stay in
`scores.jsonl`. Score vs error (MM-4): a non-finite image is
scored on the NaN-zeroed image; an OCR exception is `{class: scorer_error}`. `sampler_hash` = SHA-256 of compact
sorted JSON (as areas 2 and 3).

## Profiles and cost (estimates unless marked measured)

Per epoch: 64 samples (8 prompts × 8), 8 sampling batches of 8 (transformer batch 16 with CFG), 9 trained steps per
sample at micro-batch 2 (transformer batch 4) with gradient checkpointing (4 forward-equivalents per step incl. the
KL reference forward), 2 optimizer steps.

| Profile | Epochs | Samples | Optimizer steps | Cuts | GPU time (estimate) |
|---|---|---|---|---|---|
| smoke | 2 | 128 | 4 | 1, 2 | ~15 min training + ~3 min load/encode + 2 × 40-item evaluation (~2 min) ≈ **0.35 GPU-h** |
| full | 10 (set from the smoke) | 640 | 20 | 2, 4, …, 10 | ≈ **1.2 GPU-h** |
| evaluation | — | 400 images per arm | — | — | ≈ 9-11 min per arm ≈ **0.35 GPU-h** for both |
| cpu-sanity (host only) | 1 | 4 | 2 | 1 | measured below |

**Time per sample (estimate, 4090, fp16, SDPA):** one CFG forward of the 2.47 B transformer at 512 px (1,024 image
+ 205 text tokens, batch 2) ≈ 13 TFLOP ≈ 0.13-0.16 s at 80-100 effective TFLOPS. Sampling 10 steps ≈ 1.5 s;
training 9 steps × 4 forward-equivalents ≈ 5.4 s → ≈ 7 s per sample ≈ **7.5 min per epoch** (±50 %). `--beta 0`
(no KL reference forward) saves ≈ 20 % of training; `--no-gradient-checkpointing` ≈ 25 % if memory allows.
**Area total ≈ 1.9 GPU-h** (target ≤ 2.5): the full-run epoch count is E = floor((2.5 − smoke − 0.35) / measured
hours per epoch), capped at 10.

**Peak GPU memory (estimate):** encoding stage (before the transformer is loaded): T5 fp16 9.5 GB (+ `wo` in fp32) +
CLIP-G 1.4 + CLIP-L 0.25 ≈ 12-13 GB, freed. Training: transformer fp16 4.9 GB + VAE fp32 0.33 + LoRA/Adam/grad
0.3 + stored chains 0.35 + sampling batch 8 with VAE decode 3-5 GB transient + training micro-batch 2 with
checkpointing 3-5 GB → **≈ 11-14 GB**. Evaluation: ≈ 8-10 GB. Host RSS: the T5 load dominates (measured on CPU
15.0 GB with fp16 weights; GPU runs load the encoders with `device_map=cuda`, expected lower), then ≈ 3-5 GB plus the
paddle and docTR engines (≈ 1.8 GB measured).

## CPU sanity check (2026-11-13, host, `CUDA_VISIBLE_DEVICES=` empty, no container, outputs in a scratch directory)

All runs: `env -u HF_TOKEN HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16`, `HOME` in the scratch directory,
under `/usr/bin/time -v` (peak RSS below is its "Maximum resident set size"); the host was shared with other
CPU work (load average ≈ 30).

| Check | Result |
|---|---|
| `uv lock` / `uv sync --frozen` | 110 packages; all imports in one process (torch, diffusers, paddle, paddleocr, docTR, cv2) OK |
| OCR engines on rendered text (DejaVu Sans Bold 44 px on a 512² background) | A: "Forever Yours", "Take With Food" exact, "Step Goal Achievec" (last glyph clipped at the image edge); B: the same words; A 0.1-0.4 s/image (CPU), B 1.4 s/image (CPU); both engines loaded: peak RSS 1.8 GB |
| **Evaluation, base, the real sampler** (512 px, Euler ODE 10 steps, CFG 4.5; transformer fp32, encoders fp16 on CPU), 4 items, K 1 | rc 0; 113 s/image generation, 634 s wall; **peak RSS 15.5 GB** (15.2 GB reached while the T5 encoder is loaded and run; ≈ 11 GB during generation); 0 errors, 0 non-finite. Per item `ocr_a_acc` / `ocr_b_acc`: "Spring Collection 2024" 0 / 0 (no text found), "Step Goal Achieved" 0.31 / 0.25 (both read "Steep Goal Achieved" plus extra characters), "Try Our New Burger" **1 / 1** ("TRY OUR NEW BURGER" on a billboard), "Lost City Near" 0 / 0 (garbled). **Not degenerate**: SD3.5 Medium renders some targets exactly, others partly or not at all, as expected; the two engines agree item by item |
| Training micro-run (`--profile cpu-sanity`: fp32 transformer, fp16 encoders, 128 px, 3 steps (2 trained), 2 prompts × 2 images, 2 optimizer steps, KL β 0.04, gradient checkpointing; host-only reward term below) | rc 0; 110 s wall (encode 49 s, epoch 56 s: sampling 15 s, training 40 s); **peak RSS 16.3 GB** (15.1 GB after the encoders, 16.0 GB after loading the transformer, LoRA and both OCR engines). Step 1: approx_kl 1.8e-15, clipfrac 0, KL 0 (on-policy: sampling and training log-probs agree; the zero-initialized LoRA B equals the reference); step 2: approx_kl 1.6e-4, KL 0.0088 (the KL path through `disable_adapters()` works). `checkpoint-1/` written with `cut.json` last; 75.2 MB LoRA (fp32, r 32) |
| Host-only reward term | at 128 px no text is readable: every OCR reward is 0, every advantage 0 and every sample would be filtered (no update). The `cpu-sanity` profile therefore adds 0.01 × mean brightness (`reward: ocr+brightness_sanity`, recorded in `run_args.json`); `smoke` and `full` use `reward: ocr` only |
| Reproducibility | three CPU training runs gave byte-identical LoRA files and cut files (adapter config, sampler, base ref, trainer state) and identical `train_log.jsonl`; `samples.jsonl` identical except `utc` and the attempt-dependent `sample_id` (one rerun landed in a non-empty scratch directory and became attempt 2, which also exercised the preemption path: earlier contents moved to `preempted-attempt-1/`) |
| Cut → evaluation | `eval_images.py --checkpoint` verifies `cut.json`, the base binding and the sampler code, loads the LoRA (`load_lora_adapter`), and accepts the cut's `sampler.json`; the arm's `sampler_hash` equals the base arm's at the same settings (`3cb6fd57…`); the images differ from the base's; two evaluations of the same cut weights gave byte-identical PNGs; peak RSS 16.3 GB; arms carry exactly the MM-2 keys |
| Stray writes | none outside `HOME` and `--out` (paddle creates an empty `$HOME/.cache/paddle/dataset/`; no `~/.paddleocr`, no docTR cache use) |

Found and fixed during the check: paddle 3.3.1 cannot run the PP-OCR PIR models on the CPU (pinned 3.0.0 plus the
`enable_memory_optim` switch, "Changes"); `PaddleOCR(...)` would download from bcebos (built from `TextSystem`);
the weight precisions were missing from `sampler.json` although they change the images (added; a mismatching
evaluator precision now refuses the cut); `eval_meta.json` must hold exactly the MM-2 keys (tier and role moved to
`summary.json`). Host memory: the T5 load dominates on the CPU (≈ 15 GB); GPU runs load the encoders with
`device_map=cuda` (host RSS there is unmeasured, estimate ≤ 6 GB).

Measured hashes: `sampler_code_sha256` `sha256:83133e24…bff589` (the shared block, byte-identical in both
scripts; since the move of its edit-distance import to rapidfuzz `sha256:bba7ecd5…483cbb6f`, `RESULT.md`); the gate-time `train_flow_grpo.py`, `eval_images.py` (2026-11-14, engine-B substring formula; before: `c100a8fb…8d3b67`, numeric `sub_metrics`; `c28fd482…4a78`; the shared
sampler block unchanged) and `fetch_models.py` are not published (the published scripts differ from these
measured files by the pre-publication cleanup; `RESULT.md` lists the changes). Deployment sigmas (10 steps, shift 3.0): 1.0, 0.9601, 0.9133, 0.8577, 0.7904, 0.7073, 0.6022,
0.4649, 0.2780, 0.0089, 0.


## What gatedtrain needs (adapter proposed `flow_grpo_sd35`, trainer id `rl.flow_grpo@0.1` as spec §6 row 2b)

The run layout follows area 3's, so the adapter can be a near copy of the `ddpo_sd15` adapter (gatedtrain-recipe-ddpo-sd15) with:

1. **Cuts and completeness (MM-20).** `checkpoint-(\d+)/` with `cut.json` (schema `ptc_mm.flow_grpo_sd35_cut`) whose
   `files` SHA-256 all match and whose `epoch` equals the directory number; `.checkpoint-*.partial/` and
   directories without `cut.json` are incomplete; `cut.json` adds `samples_filtered` and `ema` (bool). The last
   cut's epoch equals `run_args.json` `settings.epochs`. (Spec §6 row 2b's `checkpoint-<step>/lora/` was flow_grpo's
   layout; this port writes area 3's.)
2. **Identity (MM-17, MM-18).** Components: `pytorch_lora_weights.safetensors`, `adapter_config.json`, `sampler.json`
   (deployable), `base_ref.json` (reference); no "precomputed T5 embedding manifest" (spec §6 row 2b): embeddings
   are computed in-run from the base. `base_hash` = manifest hash of `models/stable-diffusion-3.5-medium/`
   **excluding `.cache/`** (or after it is moved out) and after the duplicate decision (open question 4);
   `BASE_REQUIRED` = `model_index.json`, `scheduler/scheduler_config.json`, `transformer/config.json`,
   `vae/config.json`, `text_encoder{,_2,_3}/config.json`, `tokenizer{,_2,_3}/tokenizer_config.json`; weight
   directories `transformer`, `vae`, `text_encoder`, `text_encoder_2`, `text_encoder_3`. Baseline arm:
   `checkpoint.model_hash` = base hash.
3. **Base binding (MM-21).** `base_ref.json` = `stabilityai/stable-diffusion-3.5-medium` @ `b940f670…eb80`, `files`
   equal `MODELS.json`'s record; the model is **gated**: the adapter must never fetch it (it only hashes the local
   directory).
4. **Data binding.** `run_args.json` `train_file` = `<runtime>/prompts/train.jsonl`,
   `train_file_sha256` = `sha256:75361afd…585fb8`.
5. **Seed and pins.** `seed` = the launch record's `training_seed` (proposed 5000); `env-freeze.txt` pins diffusers,
   peft, torch, transformers plus `paddleocr`, `paddlepaddle`, `python-doctr` (reward and evaluator engines), and
   the line `# flow_grpo algorithm reference yifan123/flow_grpo <40 hex> (ported)`; `script_sha256` (EXT-83);
   `sampler_code_sha256` equal in cut and evaluator (shared block byte-identical, `sha256:bba7ecd5…483cbb6f`; the
   published run's `sha256:83133e24…bff589`).
6. **Consumption (MM-24, MM-28): precision `row`, derivation `observed`.** Per sample from `samples.jsonl` ⋈
   `epoch_plan.jsonl` ⋈ `train_steps.jsonl`: `data_ref.snapshot` = manifest hash of `prompts/`, shard
   `train.jsonl`, `row` = sha256(`prompt_id`), `seed_inputs.seed` = noise seed, outcome **`consumed`** with `step` =
   the step whose `sample_ids` contain it, or **`filtered`** with reason `zero_advantage` (in `epoch_plan.filtered`;
   no step). **New against area 3:** `filtered` occurs (flow_grpo drops samples whose prompt group has equal rewards,
   frequent when a prompt is always or never rendered). Refuse when: an epoch's samples ≠ 64; a sample is not in
   exactly one of `kept`/`filtered`; `len(kept)` is not a multiple of 8; a kept sample is not in exactly one step of
   its epoch (per inner epoch); a filtered one is in any step; steps are not consecutive from 1; an epoch with no kept
   samples has steps; a cut's `samples_consumed` / `samples_filtered` ≠ the counts up to its epoch. Records after
   the last cut's `global_step` are orphaned (MM-27).
7. **Exact overlap (MM-11).** Item `prompt` strings against every `prompt` of `samples.jsonl` and the shard: 0 by
   construction; additionally target-text overlap 0 (`item_set_meta.json`).
8. **Arms.** `eval_images.py` writes two arms in one run, `<out>/arm-ocr_b_acc` (deciding) and `<out>/arm-ocr_a_acc`
   (supporting), as area 3's `ext-collect --eval-* <out>/arm-<metric>` path; `higher_is_better`, range (0, 1);
   `eval_meta` lacks `model_hash` and `item_set_hash` unless passed (`--model-hash`, `--item-set-hash`) or filled.
9. **Registry rows (proposal).** Trainer: `framework: "diffusers"`, `pins: ("diffusers", "peft", "torch",
   "transformers", "paddleocr", "paddlepaddle")`, `consumption {target: row, fallback: stage}`, `gpu_h_estimate
   1.9`, `input_roots`: the Python directory, `<runtime>`,
   `<output-root>/flow-grpo-sd35-ocr`; `scan_skip`: the area's `.venv` (50,835 files; credential-like
   names only `certifi/cacert.pem`) and the interpreter's site-packages. Evaluators (`higher_is_better`, `samples
   "400, 1"`, `error_classes ("scorer_error",)`, `payload_text_field "prompt"`, `item_payload ("prompt", "target",
   "source_line")`): `t2i.sd35_ocr_doctr_items@0.1` (`ocr_b_acc`, (0, 1), T3 construct-aligned proposed),
   `t2i.sd35_ocr_paddle_items@0.1` (`ocr_a_acc`, (0, 1), T0). Scorer `code_sha256` = SHA-256 of `eval_images.py`,
   `weights_sha256` = hash of the engine's files in `MODELS.json`.

## Launch request facts

| Host | Container | Mode |
|---|---|---|
| `<python-root>` | same | ro |
| `<runtime>` (this directory: scripts, lock, venv, `models/` with the SD3.5 directory, `ocr-models/`, `prompts/`, `items/`, `data-source/`, `MODELS.json`) | same | ro |
| evaluation of a cut only: `<output-root>/flow-grpo-sd35-ocr/<run>/checkpoint-<E>` | `/inputs/cut` | ro |
| `<output-root>/flow-grpo-sd35-ocr/<run>` (new, empty, owned by the launching user) | `/out` | rw |
| tmpfs | `/tmp` | `rw,exec,nosuid,nodev,size=8g` |

The model directory sits inside the area mount (its own mount would nest, EXT-94); the reference
clone, the uv cache, the `.pt` downloads and the download HF_HOME stay outside it and are **not** mounted. Environment names (none a credential):
`HF_HUB_OFFLINE=1`, `HOME=/tmp/home`, `XDG_CACHE_HOME=/tmp/cache`, `PYTHONDONTWRITEBYTECODE=1`,
`PYTHONUNBUFFERED=1`, `TOKENIZERS_PARALLELISM=false`, `DOCTR_CACHE_DIR=/tmp/cache/doctr` (docTR's cache root; it
downloads nothing here because weights are loaded from files), optional `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
`env_paths`: `HOME`, `XDG_CACHE_HOME`, `DOCTR_CACHE_DIR`. **`HF_TOKEN` must not be passed** (EXT-55 refuses it
anyway). Network `none`, `gpus: true`, user `1000:1000`, `--cap-drop ALL`, `--security-opt no-new-privileges`,
`--shm-size=8g`. argv[0] `<runtime>/.venv/bin/python`. Paddle writes nothing outside
`HOME` (checked: no `~/.paddleocr` download, models are passed as directories).

```
smoke:     argv [<python>, "<runtime>/train_flow_grpo.py",
                 "--profile", "smoke", "--out", "/out", "--seed", "5000"]
           training_seed 5000, label mm-a5-sd35-smoke, gpu_est_h 0.3
full:      argv [<python>, ".../train_flow_grpo.py", "--profile", "full", "--out", "/out", "--seed", "5000",
                 "--epochs", "<from the smoke>"]          # default 10
           training_seed 5000, label mm-a5-sd35-full, gpu_est_h 1.2
baseline:  argv [<python>, "<runtime>/eval_images.py", "--base-only", "--out", "/out", "--eval-seeds", "11",
                 "--base-model-hash", "<H>", "--model-hash", "<H base>", "--item-set-hash", "<H>"]
                                                          # smoke: add "--limit", "40"
candidate: argv [<python>, ".../eval_images.py", "--checkpoint", "/inputs/cut", "--out", "/out",
                 "--eval-seeds", "11", "--model-hash", "<H>", "--base-model-hash", "<H>", "--item-set-hash", "<H>"]
```

`framework`: `{name: "diffusers", version: "0.42.1", commit: null}` (flow_grpo reference commit in
`env-freeze.txt`); `runtime`: `{kind: "venv_in_container", image: <area-1 digest>, venv_lock:
"<runtime>/uv.lock"}`.

## Open questions

1. **[maintainers] Trainer deviation.** Accept the port (a) instead of flow_grpo's own `train_sd3.py` in a pinned legacy
   environment (b). Recommended: (a); evidence in "Trainer choice".
2. **[maintainers] EMA.** flow_grpo saves EMA weights (decay 0.9, every 8 steps); with this budget's 4-20 optimizer steps
   that saves the initial or a lagging LoRA. Options: (a) EMA off, cuts hold the raw LoRA (recommended); (b) EMA on
   (`--ema`), faithful but the smoke cut equals the base; (c) EMA on with interval 1.
3. **[maintainers] Deployment sampler.** (a) Euler ODE, 10 steps, CFG 4.5 (recommended: row 2b's 10 steps, a quarter of
   flow_grpo's evaluation cost); (b) 40 steps as flow_grpo evaluates (+≈0.5 GPU-h for both arms at N 400, or N 100);
   (c) the training SDE sampler (noise 0.7) with recorded noise.
4. **[maintainers] Base directory contents.** Remove the 11.16 GB of byte-identical `*.fp16.*` duplicates (recommended)
   or keep them; and move `.cache/huggingface/` out of the model directory or
   have the adapter exclude it. Either way, decided before the smoke, because `base_hash` covers the directory.
5. **[maintainers] Tier of `ocr_b_acc`.** (a) T3 construct-aligned (R15-1), deciding and blocking (recommended; 05 §6.1
   names "OCR edit distance with a different OCR engine" as T3); (b) treat the shared formula as a shared answer
   parser → T0, gate non-blocking. If (a), confirm that two learned OCR engines with partly undisclosed training data
   count as different tools, not "unknown lineage → T1" (05 §6.1 last rule is written for learned scorers).
6. **[routine] Sigma plan.** `fixed` from the smoke with headroom (formula above, recommended), or an internal pilot
   from a 4-epoch smoke (+0.12 GPU-h), or area 3's ucb80 rule.
7. **[routine] Full-run epochs** from the smoke's measured seconds per epoch (default 10, cap 10), and `--beta 0`
   only if the measured epoch time leaves fewer than 6 epochs.
8. **[routine] N and K.** N 400, K 1 (proposal); 40 smoke items per arm.
9. **[routine] Reward/metric degeneracy at the smoke.** If the smoke baseline's `ocr_b_acc` share at 1 exceeds ~0.6
   or at 0 exceeds ~0.6 (MM-38), revisit before the full run.
10. **[routine] Adapter work.** New adapter `flow_grpo_sd35` (item list above), including the `filtered` outcome.
11. **[routine] Record keeping.** Spec §6 row 2b text (layout `checkpoint-<E>/`, no T5 embedding manifest, the
    evaluator ids) updated after the decisions.

## Decisions taken (2026-11-13 and 2026-11-14)

The maintainers settled the open questions as follows. 1: the port (a) accepted (flow_grpo's own script does not fit
24 GB and pins conflicting dependencies; the deviations are listed above). 2: (a) EMA off; a cut holds the raw LoRA
weights. 3: (a) Euler ODE, 10 steps, CFG 4.5 for deployment and evaluation. 4: the model directory stays untouched:
the byte-identical `*.fp16.*` duplicates stay and are hashed, and the adapter leaves `.cache/` out of the base hash.
5: (a) `ocr_b_acc` (docTR) is T3, deciding and blocking, construct-aligned under R15-1 (the tool is the OCR engine;
the shared normalization and edit distance are not an engine), disclosed in the registration and the report;
`ocr_a_acc` (PaddleOCR, the reward engine) is a supporting arm. 6: `sigma_plan: fixed` = max(1.5 × ucb80(s_pilot),
√2 × ucb80(s_base)) from the smoke. 7: full-run epochs from the smoke's measured time. 8: N 400, K 1, eval seed 11,
40 smoke items per arm; run seed 5000. 10: adapter `flow_grpo_sd35` with `filtered` (`zero_advantage`) consumption
outcomes.

**Smoke finding (2026-11-14).** The first smoke attempt failed at `import cv2` (the CUDA image lacks libxcb); the
derived image `ptc-a5-runtime:1` (`container/Containerfile`, tagged `gatedtrain-a5-runtime:1` since the rename) adds the X11/GL system libraries. The second attempt ran
(172 s per epoch, peak 13.2 GB). The 40-item base evaluation failed MM-38 on the deciding metric: `ocr_b_acc` was 0 on
52.5 % of items, mostly because docTR reads background texture as text and flow_grpo's formula clips the whole-page
edit distance to 0. The maintainers then chose the best-matching-substring formula for `ocr_b_acc` (1 − min over
substrings s of lev(s, target) / len(target), spaces removed, lower case); the training reward (engine A, flow_grpo's
formula) is unchanged. CPU recomputation on the same 40 images: share at 0 0.175, mean 0.412. The formula was chosen
after the smoke baseline and before the registration, and both disclose it.
