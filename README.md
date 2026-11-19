# gatedtrain recipe: Flow-GRPO SD3.5 OCR

A [gatedtrain](https://github.com/gatedtrain/gatedtrain) recipe: Flow-GRPO trains a LoRA on Stable Diffusion 3.5 Medium to render the
quoted text of a prompt (flow_grpo's OCR task), with a minimal port of flow_grpo (`yifan123/flow_grpo` at `879042cf`,
MIT) on diffusers + peft. The training reward is the OCR accuracy of PaddleOCR PP-OCRv4; gatedtrain gates the trained LoRA
against the base model on 400 held-out prompts, scored by a second OCR engine (docTR). Prototype.

Code comments, tests and `RESULT.md` cite records of the framework's development: the maintainers' decision record
("Area 5 decisions"; `runtime/README.md` "Decisions taken" summarizes how its questions were settled), the
framework's review rounds ("Review 10 M1") and ids of its internal issue lists (`G-21`). Those records are not
published.

## Layout

| Path | What |
|---|---|
| `src/gatedtrain_recipe_flow_grpo_sd35_ocr/` | the gatedtrain adapter `flow_grpo_sd35` (`adapter.py`, a subclass of gatedtrain's `gatedtrain.families.flow_grpo.FlowGrpoAdapter`) and its registry rows (`rows.py`), registered through the `gatedtrain.adapters` entry point |
| `runtime/` | what runs in the container: `train_flow_grpo.py` (the Flow-GRPO port), `eval_images.py` (image generation and both OCR scorers; `--make-items` builds the item set and the training shard), `fetch_models.py` (OCR engine weights, hashes of every model file in `MODELS.json`), the registered item set (`items/`), the training shard's metadata (`prompts/prompts_meta.json`), and its own `pyproject.toml` and `uv.lock` |
| `container/Containerfile` | the runtime image: the pinned CUDA base image plus the X11/GL system libraries that OpenCV loads; Python and the runtime venv are mounted read-only from the host |
| `tests/` | the adapter's tests on fixture runs (no GPU, no model needed) and gatedtrain's conformance check |
| `results/2026-11-14/` | the gate files of the published result (redacted copies: host paths replaced by placeholders; see `RESULT.md`) and the smoke base evaluation's scores |

Not in the repository (git-ignored, regenerated): `runtime/models/stable-diffusion-3.5-medium/` (a gated Hugging Face
model: accept its license and download revision `b940f670f0eda2d07fbb75229e779da1ad11eb80` yourself; `fetch_models.py`
only hashes it), `runtime/ocr-models/` (`fetch_models.py`), `runtime/data-source/ocr/{train,test}.txt` (flow_grpo's
`dataset/ocr/` at `879042cf`; hashes in `runtime/prompts/prompts_meta.json`), `runtime/prompts/train.jsonl`
(`eval_images.py --make-items` from the data source; its SHA-256 is in `prompts_meta.json`), `runtime/.venv`.

## Setup

```bash
pip install "gatedtrain @ git+https://github.com/gatedtrain/gatedtrain@v0.3.0"   # the framework, at the release this recipe was tested with
pip install -e .                      # the adapter, into the environment that runs gatedtrain
cd runtime && uv sync --frozen        # the training and evaluation environment (diffusers, peft, torch, paddleocr, python-doctr)
python fetch_models.py                # OCR engine weights; hashes the SD3.5 Medium directory into MODELS.json
```

The adapter finds the runtime at `runtime/` of this checkout, located from its own source tree, so a non-editable
install (a wheel) must set `GATEDTRAIN_RECIPE_FLOW_GRPO_SD35_OCR_RUNTIME` to the runtime directory. It reads
the interpreter tree from `runtime/.venv/pyvenv.cfg`; the runs' outputs go under the host's gatedtrain output root
(`<output_root>/flow-grpo-sd35-ocr`). `runtime/README.md` has the commands, mounts, profiles and the decisions behind
the metric.

Trust: `fetch_models.py` unpickles the two docTR `.pt` downloads (with `torch.load(weights_only=True)`, after checking
their full SHA-256) and converts them to safetensors; the LoRA cuts are safetensors. Load only checkpoints and model
files you produced or verified against the pinned hashes.

## Tests

```bash
python -m unittest discover -s tests -t tests
```

## Result

See `RESULT.md`.

## Data and licenses

| What | Source | License | In this repository |
|---|---|---|---|
| Stable Diffusion 3.5 Medium (the base model) | `stabilityai/stable-diffusion-3.5-medium` @ `b940f670f0eda2d07fbb75229e779da1ad11eb80` (gated) | Stability AI Community License (model card; `LICENSE.md` in the model repository); it also governs LoRA weights trained on the model | no: accept the license and download it yourself |
| PaddleOCR PP-OCRv4 detector (training reward, supporting arm) | `PaddlePaddle/PP-OCRv4_mobile_det` @ `3cc09f3a5b424e8e010abc7a4271aea12999c2f7` | Apache-2.0 (model card) | no: `fetch_models.py` downloads it |
| PaddleOCR PP-OCRv4 English recognizer | `PaddlePaddle/en_PP-OCRv4_mobile_rec` @ `f97b62fdc0eb71c689393a19a4b21baa1795c9ab` | Apache-2.0 (model card) | no |
| docTR db_resnet50 and crnn_vgg16_bn (deciding metric) | mindee/doctr GitHub release assets `v0.7.0/db_resnet50-79bd7d70.pt`, `v0.12.0/crnn_vgg16_bn-0417f351.pt` (full SHA-256 pinned in `fetch_models.py`) | Apache-2.0 (the mindee/doctr repository license) | no |
| OCR engines (libraries) | paddleocr 2.10.0, paddlepaddle 3.0.0, python-doctr 1.1.0 (`runtime/uv.lock`) | Apache-2.0 (PaddlePaddle/PaddleOCR, mindee/doctr repositories) | no: installed by `uv sync` |
| OCR prompts (items and training shard) | yifan123/flow_grpo @ `879042cf` `dataset/ocr/{train,test}.txt` | MIT (`THIRD_PARTY_NOTICES.md`) | the registered items and test fixtures (prompts copied from those files); the source files and the training shard are regenerated |
| Flow-GRPO algorithm | yifan123/flow_grpo @ `879042cf5707f8b90daa98d147d7deac2317c5da` | MIT | ported in `runtime/train_flow_grpo.py` and `runtime/eval_images.py` (`THIRD_PARTY_NOTICES.md`) |
| CUDA base image | `docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266` | NVIDIA Deep Learning Container License (the image's Docker Hub page) | no: `container/Containerfile` builds on it and adds Ubuntu packages under their own licenses |
| Python packages of the runtime | `runtime/uv.lock` | each package's own license | no: installed by `uv sync` |
| Edit distance of the OCR reward and `ocr_a_acc` | rapidfuzz 3.14.6 (`runtime/uv.lock`); the published run used python-Levenshtein 0.27.5, which returns the same values (`RESULT.md`) | MIT (rapidfuzz) | no: installed by `uv sync` |

### Stable Diffusion 3.5 Medium: Stability AI Community License

This repository contains no Stability AI model weights and no weights derived from them. The base model, and every
LoRA trained on it with this recipe, are governed by the Stability AI Community License Agreement ("Last Updated: July
5, 2024", `LICENSE.md` of `stabilityai/stable-diffusion-3.5-medium` at revision
`b940f670f0eda2d07fbb75229e779da1ad11eb80`). Read the agreement itself; in short, at that revision:

- The model is gated: Hugging Face shows the download only after you accept the agreement ("By clicking "Agree", you
  agree to the License Agreement and acknowledge Stability AI's Privacy Policy", the model card's gating prompt).
- It applies to derivatives: "Derivative Work(s)" include "any other model created which is based on or derived from
  the Model or the Model's output", naming "low-rank adaptation" models, "but do not include the output of any Model"
  (section V), so a LoRA cut of this recipe is one.
- Attribution, when you distribute the model or a Derivative Work, or a product or service that uses them (section
  IV.a): provide a copy of the agreement; keep, in a "Notice" text file distributed with the copies, the notice "This
  Stability AI Model is licensed under the Stability AI Community License, Copyright ©  Stability AI Ltd. All Rights
  Reserved" (and, for a Derivative Work, state there that you changed the Stability AI Materials and how); and
  "prominently display "Powered by Stability AI" on a related website, user interface, blogpost, about page, or
  product documentation".
- Commercial use (section III): "If You are using or distributing the Stability AI Materials for a Commercial
  Purpose, You must register with Stability AI"; if "You or Your Affiliate(s), either individually or in aggregate,
  generate more than USD $1,000,000 in annual revenue", the licenses under the agreement terminate and you must
  request a license from Stability AI.
- Use restrictions (section IV.b): use must comply with applicable laws and follow Stability AI's Acceptable Use
  Policy (AUP), which the agreement incorporates.

## Responsible use

This recipe fine-tunes and samples Stable Diffusion 3.5 Medium, a text-to-image model, here trained to render text in
images. Use it, and anything trained with it, only with care:

- Do not use it to create content that is illegal, infringes others' rights, harasses or demeans people, depicts real
  people without their consent, or is meant to deceive (for example fabricated news or evidence).
- Label generated images as synthetic wherever they are published or shared.
- Do not use the text-rendering ability to forge documents, signatures, labels or other text that others rely on.
- Follow the Stability AI Community License and the Acceptable Use Policy it incorporates (section IV.b, above); they
  also govern a LoRA trained with this recipe.
- Comply with the laws that apply to synthetic media where you use it.

## License

Apache License 2.0 (`LICENSE`); attribution notice in `NOTICE`. Ported code and copied data, with their licenses
(yifan123/flow_grpo, MIT; huggingface/diffusers, Apache-2.0): `THIRD_PARTY_NOTICES.md`. Model weights, datasets and
the container image are not part of this repository and keep their own licenses ("Data and licenses" above).

Names of third-party projects and products are trademarks of their owners; this project is not affiliated with or
endorsed by them.
