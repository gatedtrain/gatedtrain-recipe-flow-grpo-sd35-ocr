# Third-party notices

This repository is licensed under the Apache License 2.0 (`LICENSE`). It contains code ported from, and data derived
from, the projects below: yifan123/flow_grpo (MIT License) and, through it, huggingface/diffusers (Apache License
2.0). Model weights are not part of this repository; their
licenses are listed in `README.md` ("Data and licenses").

## yifan123/flow_grpo

- Source: https://github.com/yifan123/flow_grpo, commit `879042cf5707f8b90daa98d147d7deac2317c5da`
- License: MIT (the `LICENSE` file of that commit; a byte copy is `runtime/data-source/ocr/LICENSE.flow_grpo`, and its
  text is reproduced below)
- Used in this repository:
  - `runtime/train_flow_grpo.py`: a port of the SDE step with log-probability, the sampling loop with classifier-free
    guidance, the prompt encoding, `compute_log_prob`, the GRPO update with the KL term, the advantage statistics and
    the zero-advantage filter, the optional EMA, the OCR reward and the `general_ocr_sd3_1gpu` hyperparameters onto
    current diffusers and peft; every ported piece is cited by file and line of that commit in the script.
  - `runtime/eval_images.py`: the shared sampler block (the same ported sampling code) and flow_grpo's OCR accuracy
    formula (`ocr.py`).
  - Prompts from flow_grpo's `dataset/ocr/{train,test}.txt` at that commit: the registered items
    (`runtime/items/items.jsonl`) and the test fixtures under `tests/fixtures/flow_grpo/` hold prompts copied from
    those files. The files themselves (`runtime/data-source/ocr/{train,test}.txt`) and the training shard built from
    them are not committed; their SHA-256 are in `runtime/prompts/prompts_meta.json`.

```
MIT License

Copyright (c) 2025 Jie Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## huggingface/diffusers (Apache-2.0)

- Source: https://github.com/huggingface/diffusers. flow_grpo names no diffusers commit for its copies; the commits
  below are the last ones that changed each file before flow_grpo first committed its copy, and the code is used as
  copied by flow_grpo (commit `879042cf5707f8b90daa98d147d7deac2317c5da`).
- Files:
  - flow_grpo's `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py` is copied, with modifications, from
    `src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py` (its first line says so; diffusers
    commit `9c7e205176c30b27c5f44ec7650a8dfcc12dde86`). The sampling loop with classifier-free guidance in
    `runtime/train_flow_grpo.py` and `runtime/eval_images.py` (the shared sampler block) is ported from that copy.
  - flow_grpo's `flow_grpo/diffusers_patch/train_dreambooth_lora_sd3.py` is the diffusers example
    `examples/dreambooth/train_dreambooth_lora_sd3.py` (diffusers commit
    `66e50d4e248a32ef8f8698cf3e6f0e1040f74cfc`), modified, with its copyright header. The prompt encoding in the
    shared sampler block is ported from it.
  - flow_grpo's `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py` says it is adapted from kvablack/ddpo-pytorch's
    `ddim_with_logprob.py`, itself copied from diffusers' `src/diffusers/schedulers/scheduling_ddim.py` (commit
    `fc6acb6b97e93d58cb22b5fee52d884d77ce84d8`); the SDE step in the shared sampler block is ported from it.
- Copyright lines of the diffusers files at those commits:
  - `pipeline_stable_diffusion_3.py`: "Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All
    rights reserved."
  - `train_dreambooth_lora_sd3.py`: "Copyright 2025 The HuggingFace Inc. team. All rights reserved."
  - `scheduling_ddim.py`: "Copyright 2023 Stanford University Team and The HuggingFace Team. All rights reserved."
- License: Licensed under the Apache License, Version 2.0 (the same text as `LICENSE`). diffusers has no NOTICE file.
