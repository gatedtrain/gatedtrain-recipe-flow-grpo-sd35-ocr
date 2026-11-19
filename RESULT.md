Prototype result of a single registered run; not a benchmark claim; not affiliated with or endorsed by the owners of the models, datasets, or frameworks used.

# Result (2026-11-14): gate PASS

Flow-GRPO + LoRA on Stable Diffusion 3.5 Medium (revision `b940f670`), the minimal port `runtime/train_flow_grpo.py`
of flow_grpo `879042cf` on diffusers 0.42.1 + peft 0.21.2, EMA off, runtime image `ptc-a5-runtime:1`
(`container/Containerfile`; the tag is `gatedtrain-a5-runtime:1` since the rename below: ptc, ptc-skel and ptc-hub
are earlier names of gatedtrain, see "Rename to gatedtrain 0.3.0"). 30 epochs x 64 samples (1,920 samples, 60 optimizer steps), cuts every 2 epochs, run
seed 5000, 1.49 GPU-hours. Training reward: OCR accuracy of PaddleOCR PP-OCRv4 (flow_grpo's formula). Deployment and
evaluation sampler: 10-step Euler ODE, CFG 4.5, 512 px.

| | |
|---|---|
| Deciding metric | `ocr_b_acc`: docTR (db_resnet50 + crnn_vgg16_bn) accuracy of the best-matching substring of the recognized text (T3, construct-aligned with the reward: disclosed in the registration and the report) |
| Baseline (SD3.5 Medium) | 0.475 |
| Candidate (cut 30) | 0.520 |
| Delta, 95% cluster-bootstrap interval | +0.0449 [0.0200, 0.0703], p 1.9e-04 |
| Items, samples per item | 400, 1 (eval seed 11) |
| Sd of paired differences | 0.251 (planned sigma 0.48, fixed from the smoke) |
| Achieved power | 0.9999 |
| Exact overlap with training | 0 (prompts and target texts) |
| Consumption precision | `row` (observed): 1,872 consumed and 48 filtered (`zero_advantage`) records |

Supporting arm (not gating): `ocr_a_acc` (PaddleOCR, the reward engine, T0) 0.450 to 0.479. The impact query on the
training shard (`prompts/train.jsonl`) returns the 15 cuts at `row`.

The deciding formula (best-matching substring) was chosen after the smoke baseline, where docTR's whole-page edit
distance scored 0 on 52.5 % of the 40 base items because docTR reads background texture as text, and before the
registration (`results/2026-11-14/smoke-evidence/` holds that smoke's scores; the adapter tests recompute the
decision's numbers from them).

Re-collected on 2026-11-16 with this repository's adapter (installed as a plugin of the framework, then named ptc) on
the original run's outputs: every field of the gate report equals the pre-migration re-collect (status PASS, Δ̂ 0.044918
[0.0200, 0.0703]) except the run-id-derived `lockfile_hash`; the ledger (1,920 records) and the lineage are equal
except their run-id-derived columns; the impact query on the training shard again returns the 15 cuts at `row`.

## Published files

The files under results/2026-11-14/ are redacted copies (host paths replaced by `<data-root>`, internal process
references removed) of `gate/registration.yaml` (`gate/spec.yaml` pins the original's hash) and `gate/spec.yaml`; the
gate-time originals are not published. `gate/trainer.yaml`
(`fbb22e15e9e49394e28af750036cded1acced8ea9aeffac979567fd032d139da`) and `smoke-evidence/` are unchanged.

The published runtime files differ from the ones that produced this result only by a cleanup before publication (the
original files are not published): comments and docstrings in `runtime/eval_images.py` (now
`sha256:622e17f34c4b506401b30637747ba19f1a94cac209d6721c88edd2d9b513bc2d`) and `runtime/train_flow_grpo.py`
(now `sha256:882574bd870f569bb3a702bed1cbe49c534f7c68b9a9c9550613379f5692b765`), whose shared sampler block and so
`sampler_code_sha256` are unchanged; the `downloaded_by` text in `runtime/MODELS.json` (now
`sha256:92ae292e0f1e599d2ca1492db8ee56faf70d9a9a212fa8f76aa37692b10ad823`; every model record is unchanged); and
`runtime/fetch_models.py` (now `sha256:11c4c744ad92e4149dbe8dd042ddea64ec23bf80190ce71ac6e40d6824e06f2d`), which now pins the
`PP-OCRv4_mobile_det` revision (`3cc09f3a5b424e8e010abc7a4271aea12999c2f7`, the one the result used) and the full
SHA-256 of both docTR downloads (the files the result used) instead of their 8-hex prefixes. The scorer pin
`code_sha256` in `rows.py` followed the published `eval_images.py`, so the identity hash of
`t2i.sd35_ocr_doctr_items@0.1` became `sha256:e9e42facebabd0e8669171b07f8f1b6167dd4c5046ade34f543965320de37dbc`
instead of the `sha256:55f0ff6a420e2ba7f9d74d1946e446d446114701985e59003f5aa80046363194` that
`gate/registration.yaml` pins (it changed again with the rename below). The result above was re-collected unchanged
on 2026-11-16, before this cleanup.

## Rename to ptc-hub 0.2.0

ptc-hub 0.2.0 renamed the persisted identifiers (schema ids, ledger writer id, analysis code version). The files under
results/2026-11-14/ and the byte-copied fixtures under tests/fixtures/flow_grpo/ (`eval_meta.*.cpu-sanity.json`) keep
the names ptc-skel 0.1.0 wrote, as do the run's records and outputs; they are read by ptc `78eb913` (this and the other
framework commits cited here, `e20736e` and `372c315` among them, are unpublished pre-release commits of the
framework, earlier named ptc-skel and ptc-hub; the published framework is `gatedtrain` `v0.3.0`), and the gate
verdict PASS above remains reproducible with ptc `78eb913`, the original `gate/registration.yaml` and
`gate/spec.yaml`, and the original runtime files (none of them published). The gate
definition is carried forward by results/2026-11-14/gate/registration-ptc-hub.yaml (amends
`sha256:2570e56e0ae63755f79d213a96246a4102486f3433f61e37bd669ab2f7cbbec1`, the published `gate/registration.yaml`, which
differs from the gate-time file only in redacted comments; post_hoc: true) and spec-ptc-hub.yaml. Re-pinned for the rename:
`runtime/eval_images.py` `sha256:622e17f34c4b506401b30637747ba19f1a94cac209d6721c88edd2d9b513bc2d` ->
`sha256:f9766b70a617a65fb53b544d460bc3c78a19da932862eefbe55f0c199c779f6a` (it writes the schema id
`ptc_hub.item_eval`; historical runs used the older versions; only the new pin is accepted). `runtime/MODELS.json`,
`runtime/fetch_models.py`, `runtime/train_flow_grpo.py`, the items and the OCR models are unchanged.

The amendment is post hoc (05 section 3.7): the scorer the registration pins changed after the gate by the
publication cleanup, before the rename, so the amendment is not version-only and also carries the new evaluator
identity `sha256:963834434c3b8ff661d97c9a711a41326dbbdf9b9242961093faaa536ed7d73e` (an evaluator change, not only a
rename; restoring the gate-time scorer would undo the cleanup). Under ptc-hub 0.2.0 the historical PASS is therefore
reported INCONCLUSIVE with reason `post_hoc`; the original verdict reproduces at ptc `78eb913` as described above.

Re-judged on 2026-11-17 with ptc `e20736e` on a scratch replay of the run (nothing under the run's directories was
written): copies of `full-1.launch.json`, `smoke-2.smoke.json` and the two `arm-ocr_b_acc` arms with only the schema
ids replaced by the ptc_hub ids and, in the arms' `eval_meta.json`, `scorer.code_sha256` rewritten from the gate-time
pin to the new pin `sha256:f9766b70...` (scratch copies only); in the launch record copy the values
derived from its content were recomputed with ptc's own functions on a scratch host-work directory (the hashed `label`
with `launch_label`, and `host_work`, `gpu_step_sha256`, `host_config_sha256`; the run's guard-log lines copied under
the new label). `ptc-hub ext-collect results/2026-11-14/gate/spec-ptc-hub.yaml --adapter flow_grpo_sd35`
(`<data-root>` substituted, runtime override at the original runtime directory, the original input roots) gave status
INCONCLUSIVE, reasons `[post_hoc]`, `post_hoc: true`, `registration_amends sha256:2570e56e...` (the published copy). Against the published
report every other field is equal, among them `delta_hat` 0.044918, the interval [0.0200, 0.0703], `p_value`
1.9e-04, `achieved_power` 0.9999, `n_items` 400, the baseline and candidate scores, the item-set, sampler, model and
checkpoint hashes, contamination, consumption precision and `gpu_seconds`, except: `status`/`status_reasons`
(PASS -> INCONCLUSIVE `post_hoc`), `post_hoc`, `registration_hash`, `registration_amends`, `evaluator_hash` and
`evaluator_hash_registered` (`sha256:96383443...`), `analysis_code_version`, `schema`, `lockfile_hash`, `run_id` and
`created_at`. The 1,920 ledger rows are equal except `writer_id` (`ptc-hub.ledger@0.2.0`) and the run-id-derived
`run_id` and `sample_id`; the lineage is equal except the registration hash and run-id-derived activity ids.

## Rename to gatedtrain 0.3.0 (2026-11-18)

The framework was renamed again, from ptc-hub to gatedtrain (distribution and import `gatedtrain`, version 0.3.0),
and this recipe with it: distribution `gatedtrain-recipe-flow-grpo-sd35-ocr`, import package
`gatedtrain_recipe_flow_grpo_sd35_ocr`, entry-point group `gatedtrain.adapters`, runtime override
`GATEDTRAIN_RECIPE_FLOW_GRPO_SD35_OCR_RUNTIME`, runtime project `gatedtrain-recipe-flow-grpo-sd35-ocr-runtime`, and
the documented image tag `gatedtrain-a5-runtime:1` (`container/Containerfile`: only the comment changed; its RUN
instruction is byte-identical, including the package list path `/etc/ptc-a5-packages.txt` it writes, and the
published runs record the image by ID). gatedtrain 0.3.0 renamed the persisted identifiers once more (schema ids
`gatedtrain.*`, ledger writer id `gatedtrain.ledger@0.3.0`, analysis code version `gatedtrain 0.3.0`). The files
under results/2026-11-14/ keep their bytes: `registration.yaml`, `spec.yaml`, `trainer.yaml` and `smoke-evidence/`
the names ptc-skel 0.1.0 wrote, `registration-ptc-hub.yaml` and `spec-ptc-hub.yaml` the ptc-hub 0.2.0 names, as do
the run's records and outputs and the byte-copied fixtures under tests/fixtures/flow_grpo/. The recipe-runtime schema
ids `ptc_mm.*` (cut markers, run arguments, models, item set and prompt metadata, whose hashes the run and the
registration record) are data-format identifiers of the published run and are kept.

### python-Levenshtein replaced by rapidfuzz

The published 2026-11-14 run computed the training reward (PaddleOCR accuracy, flow_grpo ocr.py:46-60) and
`ocr_a_acc` with `Levenshtein.distance` from python-Levenshtein 0.27.5 (GPL-2.0-or-later). The scripts now import
the same function from rapidfuzz 3.14.6 (MIT), the library python-Levenshtein wraps and the version the run's lock
already held; `runtime/pyproject.toml` lists `rapidfuzz==3.14.6` instead of `levenshtein`. The deciding metric
`ocr_b_acc` (docTR, the approximate-substring distance written out in `eval_images.py`) never used either library.
Equivalence: `runtime/check_edit_distance.py`, run in the published run's environment (python-Levenshtein 0.27.5,
rapidfuzz 3.14.6), compares both functions on 20,006 string pairs (six edge cases and 20,000 random pairs of 0-40
characters over letters, digits, spaces, punctuation and non-ASCII characters, seed 0): 0 mismatches of the
distance, and 0 mismatches of `ocr_accuracy` on the 19,490 pairs with a non-empty target.

The import lies inside the SHARED SAMPLER block of `runtime/train_flow_grpo.py` and `runtime/eval_images.py`, so the
block's hash changed. The whole difference of the block, in both files:

```diff
-    from Levenshtein import distance
+    from rapidfuzz.distance.Levenshtein import distance
```

`runtime/uv.lock` was regenerated with `uv lock` (the cache lacked paddleocr's metadata for `--offline`, so uv read
public PyPI metadata; no upgrade, `exclude-newer` unchanged): `levenshtein 0.27.5` is removed and the project renamed;
the other 108 packages keep their versions, artifacts and dependencies.

### Pins (old -> new; only the new ones are accepted)

- `sampler_code_sha256` (the SHARED SAMPLER block, `adapter.SAMPLER_CODE_SHA256`):
  `sha256:83133e246f4b73daa8cf3576b2e7152960a98fb7f19b5e92ad36cf3b5bbff589` ->
  `sha256:bba7ecd54c681151f4f7514cde9844431b5706623ad2233bfd8034a6483cbb6f`.
- `runtime/eval_images.py` (scorer `code_sha256`):
  `sha256:f9766b70a617a65fb53b544d460bc3c78a19da932862eefbe55f0c199c779f6a` ->
  `sha256:2872dc8b38b2811191e1dcb237c8478acbbe2a043e5c1d64d1f2d2810c16e8a7` (the import, the MIT attribution of its
  ported code in its docstring, the schema id `gatedtrain.item_eval`, the recorded library name).
- `runtime/train_flow_grpo.py`: `sha256:882574bd870f569bb3a702bed1cbe49c534f7c68b9a9c9550613379f5692b765` ->
  `sha256:1cd2bf0c41b327ef65f150c10fe2ae2013a8fc9249dc801b8c1d35fe2ff4825b` (the import).
- `runtime/fetch_models.py`: `sha256:11c4c744ad92e4149dbe8dd042ddea64ec23bf80190ce71ac6e40d6824e06f2d` ->
  `sha256:135648a31456ce202855254e04098ad246e34bffe40e159f853babc0e9e2b0df` (its User-Agent string).
- `runtime/uv.lock`: `sha256:8243b02f8287944026511dff3489f2cb23c91b88a0fd0d7a1b8f22678ece20d7` ->
  `sha256:502c7ce70e0950cf36cbd108cd9566e0193cbcabe8914c8d0c5bbd231caecda1`.
- `runtime/MODELS.json`, the items and the OCR models are unchanged.

### Amendment

The gate definition is carried forward by results/2026-11-14/gate/registration-gatedtrain.yaml (amends
`sha256:8938e7b4202b9b242317ccd47f0bdd4eff264b8387be27b8e011b4d62871af35`, `registration-ptc-hub.yaml`; post_hoc:
true, as the amendment it carries forward) and spec-gatedtrain.yaml. Besides `analysis_code_version`, `amends` and
`created_at`, only the evaluator identity of `t2i.sd35_ocr_doctr_items@0.1` differs from the prior amendment:
`sha256:96383443...` -> `sha256:4377c5c2d0e5894685764a1c30dd0973fec837cd24288efa4bc28449f3eae908`.
`gatedtrain validate results/2026-11-14/gate/spec-gatedtrain.yaml` (`<data-root>` substituted) exits 0;
`spec-ptc-hub.yaml` is refused (REG-FIELD, old analysis code version, with the amendment fix). The original verdict
PASS remains reproducible with ptc `78eb913` as described above; under the amendment a re-collect would report it
INCONCLUSIVE with reason `post_hoc`, as the 2026-11-17 re-judgement did.

### No re-collect under gatedtrain 0.3.0; a numeric cross-check instead

The published run cannot be re-collected under the new pins without changing what identifies its checkpoint: every
cut's `cut.json` records `sampler_code_sha256` (as do `run_args.json` and `env-runtime.json`, which the adapter
requires to agree with it and with the pin), and `cut.json` is one of the files whose manifest hash is the cut's
checkpoint hash (the report's `candidate_checkpoint_hash` `sha256:4c3574e0...` and the lineage's checkpoint hashes).
Rewriting it in a scratch copy, as the earlier re-judgements rewrote schema ids, would change those hashes, so it is
not done. The run's per-item scores do not depend on the import change.

Cross-check on 2026-11-18 with gatedtrain `372c315`: `gatedtrain.gates.evaluate_items`, the function the gate
report is computed with, applied to scratch copies of the published run's two arms (`eval/baseline` and
`eval/candidate` `items.jsonl`, read with `gatedtrain.ext.arm.read_lines`), the registered item set (400 items), the
clause of spec-gatedtrain.yaml, the registration's bootstrap seed 20261114 and 10,000 replicates, `min_paired`,
`max_unpaired`, the evaluator's score range, the run's plan and the arms' model hashes. All 27 result fields the
report also has are equal to the published report: `delta_hat` 0.044918, the interval [0.0200, 0.0703], `p_value`
1.9e-04, `achieved_power` 0.9999, `mde` 0.0351, `sd_diff` 0.251, `n_items` 400, the baseline (0.4751) and candidate
(0.5200) scores, and the pass rule's outcome PASS, which the amendment's `post_hoc: true` reports as INCONCLUSIVE.
This is a cross-check of the statistics, not a gate report; the last full re-judgement is the 2026-11-17 one above
(ptc-hub 0.2.0).
