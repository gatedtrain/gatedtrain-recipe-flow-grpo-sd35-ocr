"""External-run adapters, slice 5: area "image generation, SD3.5" (Flow-GRPO + LoRA on SD3.5 Medium, OCR text
rendering; tests/README.md section 11.22, EXT-160 to EXT-178).

Sources: MULTIMODAL-SPEC.md MM-1 to MM-39 (section 6 row 2b and its deviations, MM-17, MM-18, MM-20, MM-21, MM-24,
MM-27, MM-28, MM-29, MM-38), design revision 16: 05 section 6.1 (R15-1: a construct-aligned T3 metric, stated in the
manifest, the registration and the report), section 11 (lineage, `lora_compose`), sections 12.1 to 12.3 (a `row`
record; the one outcome enum: `filtered` with reason `zero_advantage`), section 11.2 (derivation `observed`);
runtime/README.md "Decisions taken"; the runtime `runtime/` (README
"Outputs of `train_flow_grpo.py`", "Evaluator outputs", "What gatedtrain needs", "Launch request facts";
`train_flow_grpo.py`, `eval_images.py`; read 2026-11-13) and its CPU sanity outputs. Helpers: `tests/_flow_grpo.py`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

import _flow_grpo as F
from gatedtrain.testing.ext import IMAGE_ID, sha_of, write_canonical
from gatedtrain.testing.support import (EXIT_OK, EXIT_REFUSED, describe, file_sha256, manifest_hash, parse_errors,
                                 parse_run_dir, run_cli)

NOT_PASS = 1


def registry():
    from gatedtrain.ext import registry as reg
    return reg


def adapter():
    return registry().adapter(F.ADAPTER)


def _root_can_read() -> bool:
    return os.geteuid() == 0


class FlowCase(unittest.TestCase):
    """One scenario per test that changes it (`fresh`)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-flow-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.n = 0

    def fresh(self, **kw) -> F.FlowGrpoScenario:
        type(self).n += 1
        return F.FlowGrpoScenario(self.tmp / f"sc-{type(self).n}", **kw)

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def passes(self, sc: F.FlowGrpoScenario, **kw) -> Path:
        proc = sc.collect(self.runs(), **kw)
        self.assertIn(proc.returncode, (EXIT_OK, NOT_PASS), describe(proc))
        return parse_run_dir(proc.stdout)

    def refused(self, rule: str, field_prefix: str | None = None, *, sc: F.FlowGrpoScenario, text: str | None = None,
                **kw):
        runs = self.runs()
        proc = sc.collect(runs, **kw)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_prefix is None or e["field"].startswith(field_prefix))
                and (text is None or text in e["message"])]
        self.assertTrue(hits, f"expected error[{rule}] {field_prefix}... {text!r}; got {errs}\n{describe(proc)}")
        self.assertFalse(runs.exists() and any(runs.iterdir()), "a refusal writes no run directory")
        return hits

    def unreadable(self, p: Path) -> None:
        if _root_can_read():
            self.skipTest("running as root: chmod 0 does not make a file unreadable")
        p.chmod(0)
        self.addCleanup(p.chmod, 0o644)


# ---------------------------------------------------------------------------
# EXT-160: registry rows
# ---------------------------------------------------------------------------

class TestRegistryRows(unittest.TestCase):

    def test_trainer_row(self):
        row = registry().TRAINER_ROWS[F.ADAPTER]
        self.assertEqual(list(row["uses"]), [F.TRAINER_ID])
        self.assertEqual(list(row["evaluators"]), [F.EVALUATOR_ID, F.SUPPORT_ID])     # deciding first (MM-38)
        self.assertEqual(row["consumption"], {"target": "row", "fallback": None})
        self.assertEqual(set(row["pins"]), {"diffusers", "peft", "torch", "transformers", "paddleocr", "paddlepaddle"})
        self.assertEqual(row["framework"], "diffusers")
        self.assertEqual(tuple(row["components"]), F.DEPLOYABLE)
        self.assertEqual(row["gpu_h_estimate"], 1.9)
        self.assertFalse(row["spec_binding"])

    def test_evaluator_rows(self):
        rows = registry().EVALUATOR_ROWS
        for eid, metric, tier, scorer in ((F.EVALUATOR_ID, F.METRIC, "T3", F.SCORER),
                                          (F.SUPPORT_ID, F.SUPPORT_METRIC, "T0", F.SUPPORT_SCORER)):
            with self.subTest(evaluator=eid):
                row = rows[eid]
                self.assertEqual((row["metric"], row["direction"], tuple(row["score_range"]), row["tier"]),
                                 (metric, "higher_is_better", (0, 1), tier))
                self.assertFalse(row["binary"])
                self.assertEqual(tuple(row["error_classes"]), ("scorer_error",))
                self.assertEqual(tuple(row["score_outcomes"]), ("nonfinite_image",))
                self.assertEqual(row["payload_text_field"], "prompt")
                self.assertEqual(tuple(row["item_payload"]), ("prompt", "target"))
                self.assertFalse(row["normalizer"] or row["env_version"])
                self.assertEqual(set(row["sampler_keys"]), set(F.SAMPLER_KEYS))
                self.assertEqual(row["scorer"], scorer)
                self.assertEqual(row["samples"], "400, 1")

    def test_construct_alignment_declared(self):
        """R15-1 (decision "Area 5 decisions", not published): the docTR metric is T3 and construct-aligned with the
        PaddleOCR reward; the PaddleOCR arm is the reward itself (T0) and declares nothing; no shared annotation is declared."""
        rows = registry().EVALUATOR_ROWS
        ca = rows[F.EVALUATOR_ID]["construct_aligned"]
        self.assertEqual(ca["reward"], "ocr_accuracy_paddleocr")
        self.assertIn("another OCR engine", ca["statement"])
        self.assertIn("not that", ca["statement"])
        self.assertIsNone(rows[F.SUPPORT_ID].get("construct_aligned"))
        self.assertIsNone(rows[F.EVALUATOR_ID].get("shared_annotation"))

    def test_identity_hash(self):
        self.assertEqual(registry().evaluator_identity_hash(F.EVALUATOR_ID),
                         sha_of({"evaluator": F.EVALUATOR_ID, "metric": F.METRIC, "direction": "higher_is_better",
                                 "scorer": F.SCORER}))

    def test_item_set_rows(self):
        """The real items: `prompt` and `target` strings (`source_line`, an integer, is not a payload field)."""
        from gatedtrain.ext.collect import read_item_set
        row = registry().EVALUATOR_ROWS[F.EVALUATOR_ID]
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-flow-items-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copyfile(F.FG / "items.jsonl", tmp / "items.jsonl")
        rows, bad = read_item_set(tmp, tuple(row["item_payload"]), tuple(row.get("item_payload_objects", ())))
        self.assertEqual((len(rows), bad), (40, []))


# ---------------------------------------------------------------------------
# EXT-161: cuts and completeness
# ---------------------------------------------------------------------------

class TestCuts(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-flow-cuts-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base = F.build_base(self.tmp / "base")
        self.out = self.tmp / "run"
        for e, gs in ((2, 4), (4, 8), (6, 12)):
            F.build_cut(self.out, e, global_step=gs, samples_consumed=gs * 6, samples_filtered=e, base=self.base,
                        script_sha256="sha256:" + "1" * 64)

    def found(self):
        complete, incomplete = adapter().find_cuts(self.out)
        return [s for s, _ in complete], sorted(Path(p).name for p in incomplete)

    def rebuild(self, epoch=4, **kw):
        shutil.rmtree(self.out / f"checkpoint-{epoch}")
        args = dict(global_step=8, samples_consumed=48, samples_filtered=4, base=self.base,
                    script_sha256="sha256:" + "1" * 64)
        args.update(kw)
        return F.build_cut(self.out, epoch, **args)

    def test_complete_by_global_step(self):
        self.assertEqual(self.found(), ([4, 8, 12], []))
        self.assertEqual([e for e, _ in adapter().checkpoint_dirs(self.out)], [2, 4, 6])

    def test_partial_directory_ignored(self):
        shutil.copytree(self.out / "checkpoint-6", self.out / ".checkpoint-8.partial")
        self.assertEqual(self.found(), ([4, 8, 12], []))

    def test_no_update_yet_is_a_cut(self):
        """Every sample of the first epochs filtered: the cut holds the initial LoRA at global step 0 with 0 samples
        consumed (train_flow_grpo.py trains no step for an epoch without kept samples, :499, :512)."""
        self.rebuild(epoch=2, global_step=0, samples_consumed=0, samples_filtered=16)
        self.assertEqual(self.found(), ([0, 8, 12], []))

    def test_incomplete_cuts(self):
        def edit(d, fn):
            doc = json.loads((d / "cut.json").read_text()); fn(doc)
            (d / "cut.json").write_text(json.dumps(doc))
        def no_marker(d): (d / "cut.json").unlink()
        def changed(d): (d / "adapter_config.json").write_text("{}")
        def wrong_epoch(d): edit(d, lambda doc: doc.update(epoch=5))
        def extra(d): (d / "notes.txt").write_text("x")
        def missing(d): (d / "trainer_state.json").unlink()
        def unlisted(d): edit(d, lambda doc: doc["files"].pop("sampler.json"))
        def deployable(d): edit(d, lambda doc: doc.update(deployable=[F.LORA_WEIGHTS, "adapter_config.json"]))
        def no_global_step(d): edit(d, lambda doc: doc.pop("global_step"))
        def negative_consumed(d): edit(d, lambda doc: doc.update(samples_consumed=-1))
        def no_filtered(d): edit(d, lambda doc: doc.pop("samples_filtered"))
        def ema_not_bool(d): edit(d, lambda doc: doc.update(ema="false"))
        def truncated(d):
            data = (d / F.LORA_WEIGHTS).read_bytes()[:-4]
            (d / F.LORA_WEIGHTS).write_bytes(data)
            edit(d, lambda doc: doc["files"].update({F.LORA_WEIGHTS: file_sha256(d / F.LORA_WEIGHTS)}))
        def marker_garbage(d): (d / "cut.json").write_text("{not json")
        for fn in (no_marker, changed, wrong_epoch, extra, missing, unlisted, deployable, no_global_step,
                   negative_consumed, no_filtered, ema_not_bool, truncated, marker_garbage):
            with self.subTest(case=fn.__name__):
                d = self.rebuild()
                fn(d)
                self.assertEqual(self.found(), ([4, 12], ["checkpoint-4"]))

    def test_base_directory_is_a_model_directory(self):
        """The SD3.5 Medium base (37 files and `.cache/`, whole safetensors) is a model directory; a truncated weight or
        a missing T5 config is not."""
        a = adapter()
        self.assertIsNone(a.model_dir_problem(self.base))
        self.assertIsNone(a.model_dir_problem(self.out / "checkpoint-6"))
        w = self.base / "text_encoder_3" / "model-00002-of-00002.safetensors"
        w.write_bytes(w.read_bytes()[:-3])
        self.assertIn("text_encoder_3", a.model_dir_problem(self.base))
        other = F.build_base(self.tmp / "base2")
        (other / "text_encoder_3" / "config.json").unlink()
        self.assertIn("text_encoder_3/config.json", a.model_dir_problem(other))
        (self.tmp / "empty").mkdir()
        self.assertIsNotNone(a.model_dir_problem(self.tmp / "empty"))


# ---------------------------------------------------------------------------
# EXT-162: identity (MM-17, MM-18) and the base hash without `.cache/`
# ---------------------------------------------------------------------------

class TestIdentity(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-flow-id-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base = F.build_base(self.tmp / "base")
        self.cut = F.build_cut(self.tmp / "run", 2, global_step=4, samples_consumed=28, samples_filtered=4,
                               base=self.base, script_sha256="sha256:" + "1" * 64)

    def test_base_alone_without_cache(self):
        """The baseline arm evaluates the base alone; its model hash is the manifest of every file outside `.cache/`
        (decision "Area 5 decisions", not published): the `*.fp16.*` duplicates are hashed, the download metadata is
        not."""
        ident = adapter().identity(self.base)
        self.assertEqual(ident["model_hash"], F.base_manifest_hash(self.base))
        self.assertEqual(ident["cut_hash"], F.base_manifest_hash(self.base))
        self.assertNotEqual(ident["model_hash"], manifest_hash(self.base))
        self.assertEqual(sorted(ident["components"]), sorted(F.BASE_FILES))
        self.assertIsNone(ident["base_ref"])

    def test_cache_contents_do_not_count(self):
        before = adapter().identity(self.base)["model_hash"]
        (self.base / ".cache" / "huggingface" / "download" / "extra.lock").write_text("lock\n")
        self.assertEqual(registry().adapter(F.ADAPTER).identity(self.base)["model_hash"], before)
        (self.base / "text_encoder" / "model.fp16.safetensors").write_bytes(b"other")
        self.assertNotEqual(registry().adapter(F.ADAPTER).identity(self.base)["model_hash"], before)

    def test_nested_cache_is_hashed(self):
        """Only the base directory's own top-level `.cache/` is left out; a `.cache` deeper down is a file of the base."""
        before = adapter().identity(self.base)["model_hash"]
        (self.base / "vae" / ".cache").mkdir()
        (self.base / "vae" / ".cache" / "x").write_text("x\n")
        self.assertNotEqual(registry().adapter(F.ADAPTER).identity(self.base)["model_hash"], before)

    def test_lora_cut(self):
        ident = adapter().identity(self.cut, self.base)
        self.assertEqual(ident["model_hash"], F.lora_model_hash(self.cut, self.base))
        self.assertEqual(ident["adapter_hash"], F.adapter_hash(self.cut))
        self.assertEqual(ident["cut_hash"], manifest_hash(self.cut))
        self.assertEqual(sorted(ident["components"]), sorted(F.DEPLOYABLE))
        self.assertEqual(ident["base_ref"], {"repo": F.REPO, "revision": F.REVISION,
                                             "manifest_hash": F.base_manifest_hash(self.base)})

    def test_resume_state_not_a_component(self):
        before = adapter().identity(self.cut, self.base)["model_hash"]
        (self.cut / "trainer_state.json").write_text('{"epoch": 2, "other": 1}\n')
        self.assertEqual(adapter().identity(self.cut, self.base)["model_hash"], before)
        (self.cut / "sampler.json").write_text('{"num_steps": 40}\n')        # the deployment sampler is a component
        self.assertNotEqual(adapter().identity(self.cut, self.base)["model_hash"], before)

    def test_cut_without_base_refused(self):
        with self.assertRaises(ValueError):
            adapter().identity(self.cut)

    def test_base_walked_once(self):
        """The 27 GB base is walked once per adapter instance (the area-3 cache, EXT-135)."""
        from unittest import mock
        import gatedtrain.families.flow_grpo as mod             # FlowGrpoAdapter.identity walks the base
        a = adapter()
        real = mod.manifest_entries
        with mock.patch.object(mod, "manifest_entries", side_effect=real) as walk:
            a.identity(self.cut, self.base)
            a.identity(self.base)
            a.identity(self.base, self.base)
        walked = [Path(c.args[0]).resolve() for c in walk.call_args_list]
        self.assertEqual(walked.count(self.base.resolve()), 1, walked)


# ---------------------------------------------------------------------------
# EXT-172: the whole run collected (and EXT-162, EXT-165 outputs)
# ---------------------------------------------------------------------------

class TestCollectPass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-flow-pass-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = F.FlowGrpoScenario(cls.tmp / "sc")
        cls.proc = cls.sc.collect(cls.tmp / "runs")
        cls.run_dir = parse_run_dir(cls.proc.stdout) if cls.proc.returncode == EXIT_OK else None

    def setUp(self):
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))

    def report(self) -> dict:
        return json.loads((self.run_dir / "report" / "gate_report.json").read_text(encoding="utf-8"))

    def ledger(self) -> list[dict]:
        import pyarrow.parquet as pq
        return pq.read_table(self.run_dir / "ledger" / "consumption.parquet").to_pylist()

    def test_pass_and_hashes(self):
        rep = self.report()
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual((rep["adapter"], rep["independence_tier"], rep["direction"]),
                         (F.ADAPTER, "T3", "higher_is_better"))
        self.assertEqual(rep["baseline_model_hash"], self.sc.base_model)
        self.assertEqual(rep["candidate_model_hash"], self.sc.cand_model)
        self.assertEqual(rep["contamination"], {"exact_overlap": 0, "allowed": False})
        self.assertEqual(rep["consumption_precision"], "row")
        self.assertEqual((rep["n_items"], rep["samples_per_item"]), (40, 1))
        self.assertGreater(rep["delta_hat"], 0)

    def test_base_manifest_without_cache(self):
        """The run's baseline manifest lists the base's files outside `.cache/` (the base hash, decision)."""
        doc = json.loads((self.run_dir / "manifests" / "baseline_checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["manifest_hash"], self.sc.base_model)
        self.assertFalse([e for e in doc["entries"] if e["path"].startswith(".cache/")])
        self.assertEqual(len(doc["entries"]), 37)

    def test_construct_alignment_reported(self):
        rep = self.report()
        ca = rep["construct_alignment"]
        self.assertEqual(ca["reward"], "ocr_accuracy_paddleocr")
        self.assertIn("independent OCR engine", ca["statement"])
        self.assertNotIn("shared_annotation", rep)
        text = (self.run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertIn("construct_alignment: ", text)

    def test_gpu_accounting(self):
        rep = self.report()
        self.assertEqual((rep["gpu_seconds"], rep["wall_seconds"]), (3600, 3600))

    def test_ledger_consumed_rows(self):
        """EXT-165 (MM-24): one observed `row` record per (kept sample, optimizer step that applied it) of epochs 1-3;
        data_ref.row = sha256(prompt_id); step = the step whose sample_ids hold the sample; seed inputs = the noise
        seed; rewards = [the OCR reward]; sample_id = H(run, attempt, epoch, step, prompt_id, seed, member)."""
        import hashlib
        rows = [r for r in self.ledger() if r["outcome"] == "consumed"]
        samples, plans, steps = self.sc.logs()
        by_id = {s["sample_id"]: s for s in samples}
        step_of = {sid: st["step"] for st in steps for sid in st["sample_ids"]}
        kept = [sid for p in plans for sid in p["kept"]]
        self.assertEqual(len(rows), len(kept))
        self.assertEqual(len(rows), 40)                     # 16 + 12 + 12
        snapshot = manifest_hash(self.sc.data)
        consumed_samples = [s for s in samples if s["sample_id"] in step_of]
        for r, s in zip(rows, consumed_samples):
            self.assertEqual((r["precision"], r["derivation"], r["reason"]), ("row", "observed", None))
            self.assertEqual(r["data_ref"]["snapshot"], snapshot)
            self.assertEqual(r["data_ref"]["shard"], "train.jsonl")
            self.assertEqual(r["data_ref"]["row"], "sha256:" + hashlib.sha256(s["prompt_id"].encode()).hexdigest())
            self.assertEqual(r["step"], step_of[s["sample_id"]])
            self.assertEqual(r["seed_inputs"], {"seed": s["seed"]})
            self.assertEqual(r["rewards"], [s["reward"]])
            self.assertEqual(r["sampler_config_hash"], sha_of(F.SAMPLER))
            key = "\x00".join(str(x) for x in (r["run_id"], r["attempt_id"], s["epoch"], r["step"], s["prompt_id"],
                                               s["seed"], s["member"]))
            self.assertEqual(r["sample_id"], "sha256:" + hashlib.sha256(key.encode()).hexdigest())
            self.assertIn(s["sample_id"], by_id)

    def test_ledger_filtered_rows(self):
        """EXT-165 (MM-28, 05 section 12.3): every sample the zero-advantage filter dropped is a record with outcome
        `filtered`, reason `zero_advantage`, no step; its sample_id puts `filtered` in the step's place."""
        import hashlib
        rows = [r for r in self.ledger() if r["outcome"] == "filtered"]
        samples, plans, _ = self.sc.logs()
        filtered = {sid for p in plans for sid in p["filtered"]}
        want = [s for s in samples if s["sample_id"] in filtered]
        self.assertEqual(len(rows), 8)                      # epoch 2: 4, epoch 3: 4
        self.assertEqual(len(rows), len(want))
        for r, s in zip(rows, want):
            self.assertEqual((r["reason"], r["step"], r["precision"], r["derivation"]),
                             ("zero_advantage", None, "row", "observed"))
            self.assertEqual(r["rewards"], [s["reward"]])
            self.assertEqual(r["seed_inputs"], {"seed": s["seed"]})
            key = "\x00".join(str(x) for x in (r["run_id"], r["attempt_id"], s["epoch"], "filtered", s["prompt_id"],
                                               s["seed"], s["member"]))
            self.assertEqual(r["sample_id"], "sha256:" + hashlib.sha256(key.encode()).hexdigest())
        ids = [r["sample_id"] for r in self.ledger()]
        self.assertEqual(len(set(ids)), len(ids))

    def test_reincluded_samples_consumed(self):
        """Samples the padding rule re-included (zero advantage, kept) are consumed, never filtered."""
        rows = self.ledger()
        _, plans, _ = self.sc.logs()
        self.assertEqual(len(plans[2]["reincluded"]), 2)
        samples = {s["sample_id"]: s for s in self.sc.logs()[0]}
        for sid in plans[2]["reincluded"]:
            s = samples[sid]
            hits = [r for r in rows if r["seed_inputs"] == {"seed": s["seed"]}]
            self.assertEqual([r["outcome"] for r in hits], ["consumed"])

    def test_lineage(self):
        edges = json.loads((self.run_dir / "lineage" / "edges.json").read_text(encoding="utf-8"))
        train = [a for a in edges["activities"] if a["kind"] == "training_stage"][0]
        model = [u for u in train["used"] if u["role"] == "model"][0]
        self.assertEqual((model["hash"], model["model_hash"]), (self.sc.base_model, self.sc.base_model))
        self.assertEqual([g["last_step_included"] for g in train["generated"]], [4, 6])
        compose = [a for a in edges["activities"] if a["kind"] == "lora_compose"]
        self.assertEqual(len(compose), 2)
        for act, cut in zip(compose, self.sc.cuts):
            used = {u["role"]: u["hash"] for u in act["used"]}
            self.assertEqual(used, {"base": self.sc.base_model, "adapter": F.adapter_hash(cut)})

    def test_impact_on_the_shard(self):
        """A `row` path answers "consumed"; filtered records do not count as use (05 section 11.2, 12.3)."""
        proc = run_cli(["impact", str(self.sc.data / "train.jsonl"), "--run-dir", str(self.run_dir), "--json"],
                       cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        rows = json.loads(proc.stdout)["results"]
        self.assertEqual([(r["last_step_included"], r["first_step"], r["precision"], r["answer"]) for r in rows],
                         [(4, 1, "row", "consumed"), (6, 1, "row", "consumed")])

    def test_consumption_note(self):
        notes = " ".join(self.report()["notes"])
        self.assertIn("8 filtered", notes)
        self.assertIn("zero_advantage", notes)

    def test_arm_images_copied(self):
        line = F.read_jsonl(self.run_dir / "eval" / "candidate" / "items.jsonl")[0]
        self.assertEqual(file_sha256(self.run_dir / "eval" / "candidate" / line["output_ref"]["path"]),
                         line["output_ref"]["sha256"])


class TestFilterShapes(FlowCase):
    """EXT-165: an epoch whose samples are all filtered trains no step; a later cut still collects."""

    def test_all_filtered_epoch(self):
        import pyarrow.parquet as pq
        sc = self.fresh(plan={2: ["flat0", "flat1", "flat0", "flat1"]})
        _, plans, steps = sc.logs()
        self.assertEqual((plans[1]["kept"], plans[1]["steps"]), ([], []))
        self.assertEqual(sorted({st["epoch"] for st in steps}), [1, 3])
        self.assertEqual(json.loads((sc.cuts[0] / "cut.json").read_text())["global_step"], 2)
        run = self.passes(sc)
        rows = pq.read_table(run / "ledger" / "consumption.parquet").to_pylist()
        self.assertEqual(sum(r["outcome"] == "filtered" for r in rows), 16)     # epochs 1 and 3: every group varied
        self.assertEqual(sum(r["outcome"] == "consumed" for r in rows), 32)

    def test_two_inner_epochs(self):
        """num_inner_epochs 2: each kept sample is applied once per inner epoch (two records), filtered ones once."""
        import pyarrow.parquet as pq
        sc = self.fresh(inner_epochs=2)
        steps = F.read_jsonl(sc.stage_out / "train_steps.jsonl")
        self.assertEqual(len(steps), 12)
        run = self.passes(sc)
        rows = pq.read_table(run / "ledger" / "consumption.parquet").to_pylist()
        self.assertEqual(sum(r["outcome"] == "consumed" for r in rows), 80)
        self.assertEqual(sum(r["outcome"] == "filtered" for r in rows), 8)

    def test_two_cuts_at_one_global_step(self):
        """Epoch 3's samples all filtered: cuts 2 and 3 share global step 4; each cut keeps its own `lora_compose`
        activity, whose id names the cut directory (EXT-172)."""
        sc = self.fresh(plan={3: ["flat0", "flat1", "flat0", "flat1"]})
        self.assertEqual([json.loads((c / "cut.json").read_text())["global_step"] for c in sc.cuts], [4, 4])
        run = self.passes(sc)
        edges = json.loads((run / "lineage" / "edges.json").read_text(encoding="utf-8"))
        ids = [a["id"] for a in edges["activities"]]
        self.assertEqual(len(ids), len(set(ids)), ids)
        compose = [a for a in edges["activities"] if a["kind"] == "lora_compose"]
        self.assertEqual([a["id"].rsplit("/", 1)[1] for a in compose], ["checkpoint-2", "checkpoint-3"])
        self.assertEqual([a["generated"][0]["last_step_included"] for a in compose], [4, 4])

    def test_records_after_the_last_cut_orphaned(self):
        sc = self.fresh()
        a = adapter()
        cuts = [(4, sc.cuts[0])]                                          # only cut epoch 2 (global step 4)
        self.assertEqual(a.consumption_problems(sc.stage_out, sc.data, cuts), [])
        cons = a.consumption(sc.stage_out, sc.data, cuts, run_id="r")
        self.assertEqual(sorted({r["step"] for r in cons["rows"] if r["step"] is not None}), [1, 2, 3, 4])
        self.assertEqual((sum(r["outcome"] == "consumed" for r in cons["rows"]),
                          sum(r["outcome"] == "filtered" for r in cons["rows"])), (28, 4))


# ---------------------------------------------------------------------------
# EXT-163: base binding (MM-18, MM-21)
# ---------------------------------------------------------------------------

class TestBaseBinding(FlowCase):

    def test_base_ref_files_differ(self):
        sc = self.fresh()
        def other(d):
            d["files"]["model_index.json"] = "sha256:" + "0" * 64
            d["files_sha256"] = F.compact_hash(d["files"])
        for cut in sc.cuts:
            sc.edit_json(cut / "base_ref.json", other)
            sc.reseal(cut)
        fs = json.loads((sc.final_cut / "base_ref.json").read_text())["files_sha256"]
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(files_sha256=fs))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="not the SHA-256 of the spec's base")

    def test_base_ref_names_cache_files(self):
        """A base reference that counted `.cache/` files is another base than the spec's (which excludes them)."""
        sc = self.fresh()
        def add(d):
            d["files"][F.CACHE_FILES[0]] = file_sha256(sc.base / F.CACHE_FILES[0])
            d["files_sha256"] = F.compact_hash(d["files"])
        for cut in sc.cuts:
            sc.edit_json(cut / "base_ref.json", add)
            sc.reseal(cut)
        fs = json.loads((sc.final_cut / "base_ref.json").read_text())["files_sha256"]
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(files_sha256=fs))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="not the SHA-256 of the spec's base")

    def test_revision_not_a_commit(self):
        sc = self.fresh()
        for cut in sc.cuts:
            sc.edit_json(cut / "base_ref.json", lambda d: d.update(revision="main"))
            sc.edit_json(cut / "adapter_config.json", lambda d: d.update(base_model_revision="main"))
            sc.reseal(cut)
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(revision="main"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="40-hex")

    def test_base_model_not_the_spec_base(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(base_model="/inputs/a5/models/other"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="base_model")

    def test_base_not_verified(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(verified=False))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="verified")

    def test_adapter_config_names_another_base(self):
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "adapter_config.json",
                     lambda d: d.update(base_model_name_or_path="stabilityai/stable-diffusion-3.5-large"))
        sc.reseal(sc.final_cut)
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="adapter_config.json")

    def test_cut_as_the_spec_base_refused(self):
        sc = self.fresh()
        sc.render(spec_mutate=lambda d: d["artifacts"]["base"].update(uri=f"file:{sc.cuts[0]}"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc)

    def test_symlink_in_base(self):
        sc = self.fresh()
        (sc.base / "extra.json").symlink_to(sc.base / "model_index.json")
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc)

    def test_cache_not_a_plain_directory(self):
        """Review 10 L4, probe P7: the excluded `.cache` must be a plain directory (T0-09 H2 holds for the base); a
        symbolic link or a file there refuses."""
        for kind in ("symlink", "file"):
            with self.subTest(kind=kind):
                sc = self.fresh()
                shutil.rmtree(sc.base / ".cache")
                if kind == "symlink":
                    other = sc.root / "elsewhere"
                    other.mkdir()
                    (sc.base / ".cache").symlink_to(other)
                else:
                    (sc.base / ".cache").write_text("x\n")
                self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text=".cache")

    def test_base_pin_leaves_out_the_cache(self):
        """Found at the area-5 collect (2026-11-14): a spec that pins the base (`file:<base>@sha256:<hash>`) pins the
        adapter's base hash, without `.cache/`; the generic pin check leaves it out too. A pin that counted `.cache/`
        is another base and refuses with ARTIFACT-HASH."""
        from gatedtrain.canonical import entries_hash, manifest_entries
        sc = self.fresh()
        without = entries_hash(manifest_entries(sc.base, exclude=(".cache",)))
        with_cache = entries_hash(manifest_entries(sc.base))
        self.assertNotEqual(without, with_cache)
        sc.render(spec_mutate=lambda d: d["artifacts"]["base"].update(uri=f"file:{sc.base}@{without}"))
        self.passes(sc)
        sc = self.fresh()
        sc.render(spec_mutate=lambda d: d["artifacts"]["base"].update(uri=f"file:{sc.base}@{with_cache}"))
        self.refused("ARTIFACT-HASH", "artifacts.base.uri", sc=sc)


# ---------------------------------------------------------------------------
# EXT-164: data binding
# ---------------------------------------------------------------------------

class TestDataBinding(FlowCase):

    def test_train_file_outside_the_mounts(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(train_file="/elsewhere/train.jsonl"))
        self.refused("EXT-DATA", "artifacts.prompts.uri", sc=sc, text="train_file")

    def test_train_file_hash_differs(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(train_file_sha256="sha256:" + "1" * 64))
        self.refused("EXT-DATA", "artifacts.prompts.uri", sc=sc, text="train_file_sha256")

    def test_row_count_differs(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(n_train_prompts=18359))
        self.refused("EXT-DATA", "artifacts.prompts.uri", sc=sc, text="n_train_prompts")

    def test_two_shards_refused(self):
        sc = self.fresh()
        shutil.copyfile(sc.data / "train.jsonl", sc.data / "train-b.jsonl")
        self.refused("EXT-DATA", "artifacts.prompts.uri", sc=sc, text="shard")


# ---------------------------------------------------------------------------
# EXT-166: the observed consumption log refuses (MM-24, MM-27, MM-28)
# ---------------------------------------------------------------------------

class TestConsumptionLog(FlowCase):

    def check(self, fn, text, **kw):
        sc = self.fresh(**kw)
        sc.edit_logs(fn)
        return self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text=text)

    def test_kept_and_filtered(self):
        self.check(lambda s, p, st: p[1]["filtered"].append(p[1]["kept"][0]), "both kept and filtered")

    def test_in_neither(self):
        def fn(s, p, st):
            gone = p[1]["filtered"].pop()
            assert gone
        self.check(fn, "neither kept nor filtered")

    def test_duplicate_in_kept(self):
        self.check(lambda s, p, st: p[0]["kept"].append(p[0]["kept"][0]), "twice")

    def test_unknown_reason(self):
        self.check(lambda s, p, st: p[1].update(filter_reason="dynamic_sampling"), "filter_reason")

    def test_plan_names_an_unknown_sample(self):
        self.check(lambda s, p, st: p[1]["filtered"].append("sha256:" + "9" * 64), "not a sample of epoch 2")

    def test_step_applies_a_filtered_sample(self):
        def fn(s, p, st):
            f = p[1]["filtered"][0]
            step = [x for x in st if x["epoch"] == 2][0]
            step["sample_ids"][0] = f
        self.check(fn, "filtered sample")

    def test_step_applies_an_unplanned_sample(self):
        def fn(s, p, st):
            st[0]["sample_ids"][0] = "sha256:" + "8" * 64
        self.check(fn, f"not in samples.jsonl")

    def test_step_applies_a_sample_of_another_epoch(self):
        def fn(s, p, st):
            other = [x for x in st if x["epoch"] == 2][0]
            st[0]["sample_ids"][0], other["sample_ids"][0] = other["sample_ids"][0], st[0]["sample_ids"][0]
        self.check(fn, "a sample of epoch")

    def test_kept_not_a_multiple(self):
        def fn(s, p, st):
            sid = p[1]["kept"].pop()
            p[1]["filtered"].append(sid)
            for x in st:
                if sid in x["sample_ids"]:
                    x["sample_ids"].remove(sid)
        self.check(fn, "multiple of num_batches_per_epoch")

    def test_reincluded_not_kept(self):
        self.check(lambda s, p, st: p[2]["reincluded"].append(p[2]["filtered"][0]), "reincluded")

    def test_reincluded_beyond_padding(self):
        """The padding re-includes fewer than num_batches_per_epoch samples (train_flow_grpo.py :486-493)."""
        self.check(lambda s, p, st: p[0].update(reincluded=p[0]["kept"][:4]), "reincluded")

    def test_kept_order(self):
        self.check(lambda s, p, st: p[0]["kept"].reverse(), "sample order")

    def test_epoch_without_kept_samples_has_steps(self):
        """An epoch whose samples are all filtered trains no step (train_flow_grpo.py :499, :512); a step logged for
        it is refused."""
        sc = self.fresh(plan={2: ["flat0", "flat1", "flat0", "flat1"]})
        def fn(s, p, st):
            k = max(x["step"] for x in st if x["epoch"] == 1)
            for x in st:
                if x["step"] > k:
                    x["step"] += 1
            st.insert(k, {"epoch": 2, "inner": 0, "sample_ids": [], "step": k + 1, "utc": "x"})
        sc.edit_logs(fn)
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="no kept samples")

    def test_plan_steps_range(self):
        self.check(lambda s, p, st: p[0].update(steps=[1, 3]), "the epoch's steps")

    def test_plan_line_missing(self):
        self.check(lambda s, p, st: p.pop(1), "epoch_plan.jsonl")

    def test_plan_line_twice(self):
        self.check(lambda s, p, st: p.insert(1, dict(p[1])), "epoch_plan.jsonl")

    def test_flat_group_kept(self):
        """A prompt group whose rewards are all equal has advantage 0 for every member; such a sample is filtered or
        re-included by the padding, never plainly kept (train_flow_grpo.py :481-498)."""
        def fn(s, p, st):
            for x in s:
                if x["epoch"] == 1 and x["sample_index"] < 4:
                    x["reward"] = x["ocr_acc"] = 0.0
        self.check(fn, "zero advantage")

    def test_varied_group_filtered(self):
        """Review 10 M1, probe P1: a group with distinct rewards (every advantage nonzero) relabelled `filtered`, the
        steps rebalanced and the cut counts adjusted, is refused: a filtered sample's reward equals its group mean."""
        sc = self.fresh()
        def fn(samples, plans, steps):
            p = plans[0]
            g0 = p["kept"][:4]
            order = [x["sample_id"] for x in samples]
            p["kept"] = p["kept"][4:]
            p["filtered"] = sorted(g0 + p["filtered"], key=order.index)
            ids = [s for st in steps if st["epoch"] == 1 for s in st["sample_ids"] if s not in g0]
            e1 = [st for st in steps if st["epoch"] == 1]
            e1[0]["sample_ids"], e1[1]["sample_ids"] = ids[:6], ids[6:]
        sc.edit_logs(fn)
        for cut in sc.cuts:
            sc.edit_json(cut / "cut.json", lambda d: d.update(samples_consumed=d["samples_consumed"] - 4,
                                                              samples_filtered=d["samples_filtered"] + 4))
        self.assertTrue(any("group mean" in m for m in adapter().consumption_problems(
            sc.stage_out, sc.data, [(4, sc.cuts[0]), (6, sc.cuts[1])])))
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="group mean")

    def test_varied_samples_reincluded(self):
        """Review 10 M1, probe P5: kept samples of a varied group listed as `reincluded` (only zero-advantage samples
        are re-included by the padding)."""
        self.check(lambda s, p, st: p[0].update(reincluded=p[0]["kept"][:3]), "group mean")

    def test_advantage_mode_unknown(self):
        """Review 10 M1: the zero-advantage rule is reproduced for the runtime's per-prompt tracker (`global_std` a
        boolean); any other advantage setting refuses instead of being accepted unchecked."""
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["settings"].update(global_std="batch"))
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="global_std")

    def test_plan_steps_of_an_all_filtered_epoch(self):
        """Review 10 L2, probe P2: an epoch without kept samples names no steps (train_flow_grpo.py :499, :503)."""
        sc = self.fresh(plan={2: ["flat0", "flat1", "flat0", "flat1"]})
        sc.edit_logs(lambda s, p, st: p[1].update(steps=[7, 9]))
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="names steps [7, 9]")

    def test_boolean_step(self):
        """Review 10 L3, probe P3: `"step": true` equals 1 in Python; it is refused (exit 3), never a traceback."""
        self.check(lambda s, p, st: st[0].update(step=True), "consecutive")

    def test_step_size(self):
        def fn(s, p, st):
            st[1]["sample_ids"].append(st[0]["sample_ids"].pop())
        self.check(fn, "applies 7 sample ids")

    def test_inner_epoch_missing_a_sample(self):
        sc = self.fresh(inner_epochs=2)
        def fn(s, p, st):
            one = [x for x in st if x["epoch"] == 1 and x["inner"] == 1]
            one[0]["sample_ids"][0] = one[1]["sample_ids"][0]
        sc.edit_logs(fn)
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="inner 1")

    def test_gap(self):
        def fn(s, p, st):
            for x in st[3:]:
                x["step"] += 1
        self.check(fn, "consecutive")

    def test_short_epoch(self):
        def fn(s, p, st):
            gone = [x for x in s if x["epoch"] == 2][-1]["sample_id"]
            s[:] = [x for x in s if x["sample_id"] != gone]
        self.check(fn, "samples_per_epoch")

    def test_epoch_beyond_the_run(self):
        def fn(s, p, st):
            p.append(dict(p[-1], epoch=4, kept=[], filtered=[], reincluded=[], steps=[]))
        self.check(fn, "settings.epochs")

    def test_cut_counts_disagree(self):
        for key in ("samples_consumed", "samples_filtered", "global_step"):
            with self.subTest(key=key):
                sc = self.fresh()
                sc.edit_json(sc.cuts[0] / "cut.json", lambda d: d.update({key: d[key] + 1}))
                self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text=key)

    def test_missing_logs_refused(self):
        for name in ("samples.jsonl", "epoch_plan.jsonl", "train_steps.jsonl"):
            with self.subTest(log=name):
                sc = self.fresh()
                (sc.stage_out / name).unlink()
                self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text=name)

    def test_unreadable_plan(self):
        sc = self.fresh()
        self.unreadable(sc.stage_out / "epoch_plan.jsonl")
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc)


# ---------------------------------------------------------------------------
# EXT-167: every field train_flow_grpo.py derives is re-derived
# ---------------------------------------------------------------------------

class TestDerivedFields(FlowCase):

    @staticmethod
    def reid(x: dict, plans: list, steps: list, **changes) -> None:
        """Change fields of a sample line and re-derive its sample_id everywhere, so only the changed field is wrong."""
        old = x["sample_id"]
        x.update(changes)
        x["sample_id"] = F.sha256_text(json.dumps(["flow-grpo-sd35", F.SEED, x["attempt"], x["epoch"], x["sample_index"],
                                                   x["prompt_id"], x["seed"], x["member"]], sort_keys=True,
                                                  separators=(",", ":")).encode())
        for p in plans:
            for key in ("kept", "filtered", "reincluded"):
                p[key] = [x["sample_id"] if s == old else s for s in p[key]]
        for st in steps:
            st["sample_ids"] = [x["sample_id"] if s == old else s for s in st["sample_ids"]]

    def check(self, fn, text):
        sc = self.fresh()
        sc.edit_logs(fn)
        return self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text=text)

    def test_seed(self):
        self.check(lambda s, p, st: self.reid(s[0], p, st, seed=12345), "derive_seed")

    def test_sample_id(self):
        def fn(s, p, st):
            self.reid(s[0], p, st)
            new = "sha256:" + "7" * 64
            for pl in p:
                pl["kept"] = [new if x == s[0]["sample_id"] else x for x in pl["kept"]]
            for x in st:
                x["sample_ids"] = [new if y == s[0]["sample_id"] else y for y in x["sample_ids"]]
            s[0]["sample_id"] = new
        self.check(fn, "sample_id is not")

    def test_member(self):
        self.check(lambda s, p, st: self.reid(s[0], p, st, member=3), "images_per_prompt")

    def test_target(self):
        self.check(lambda s, p, st: s[0].update(target="Another Text"), "target")

    def test_reward_is_the_ocr_accuracy(self):
        self.check(lambda s, p, st: s[1].update(reward=s[1]["reward"] + 0.5), "ocr_acc")

    def test_ocr_accuracy_range(self):
        self.check(lambda s, p, st: s[1].update(reward=1.5, ocr_acc=1.5), "ocr_acc")

    def test_attempt(self):
        self.check(lambda s, p, st: self.reid(s[0], p, st, attempt=2), "attempt")

    def test_prompt_text(self):
        self.check(lambda s, p, st: s[0].update(prompt=s[0]["prompt"] + " "), "prompt")

    def test_unknown_prompt(self):
        self.check(lambda s, p, st: self.reid(s[0], p, st, prompt_id="ocr-train-99999"), "ocr-train-99999")

    def test_prompt_group(self):
        """Sample i of an epoch belongs to prompt group i // images_per_prompt (train_flow_grpo.py :441); a member with
        another prompt than its group's is refused."""
        sc = self.fresh()
        other = sc.train_rows[-1]
        def fn(s, p, st):
            x = s[1]
            self.reid(x, p, st, prompt_id=other["prompt_id"], prompt=other["prompt"],
                      target=F.target_text(other["prompt"]))
        sc.edit_logs(fn)
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="group")


# ---------------------------------------------------------------------------
# EXT-168: exact overlap (MM-11)
# ---------------------------------------------------------------------------

class TestOverlap(FlowCase):

    def test_overlap_refused(self):
        sc = self.fresh(overlap=2)
        self.refused("EXT-CONTAMINATION", "registration.contamination.allow_exact_overlap", sc=sc, text="2 item(s)")

    def test_target_overlap_refused(self):
        """README item 7 (EXT-176): an item's target text equal to a training prompt's after normalization (spaces
        removed, lower case) refuses, whatever allow_exact_overlap says."""
        for allow in (False, True):
            with self.subTest(allow=allow):
                sc = self.fresh(reg_mutate=lambda r: r["contamination"].update(allow_exact_overlap=allow))
                rows = F.read_jsonl(sc.data / "train.jsonl")
                text = " " + sc.item_rows[3]["target"].upper().replace(" ", "  ")
                rows[5]["prompt"], rows[5]["target"] = f'A banner that reads "{text}" in red.', text
                F.write_jsonl(sc.data / "train.jsonl", rows)
                hits = self.refused("EXT-CONTAMINATION", "artifacts.items.uri", sc=sc, text="target text")
                self.assertIn(sc.item_rows[3]["item"], hits[0]["message"])

    def test_target_from_the_prompt(self):
        """A shard row without `target` is judged by its prompt's first quoted text."""
        sc = self.fresh()
        rows = F.read_jsonl(sc.data / "train.jsonl")
        for r in rows:
            r.pop("target")
        rows[7]["prompt"] = f'A sign that reads "{sc.item_rows[0]["target"].lower()}" on a wall.'
        F.write_jsonl(sc.data / "train.jsonl", rows)
        self.assertTrue(adapter().item_overlap_problems(sc.item_rows, sc.data))
        rows[7]["prompt"] = 'A sign that reads "something else" on a wall.'
        F.write_jsonl(sc.data / "train.jsonl", rows)
        self.assertEqual(adapter().item_overlap_problems(sc.item_rows, sc.data), [])

    def test_stated_target_differs_from_the_prompt(self):
        """Review 10 L1, probe P4: the trainer and the scorer use the prompt's first quoted text; a shard row or an
        item whose `target` field says otherwise refuses, and the overlap uses the derived target."""
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-flow-tgt-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        items = [{"item": "i1", "prompt": 'a mug with "hello world"', "target": "hello world"}]
        (tmp / "train.jsonl").write_text(json.dumps({"prompt_id": "x", "prompt": 'a sign that reads "Hello World"',
                                                     "target": "decoy"}) + "\n", encoding="utf-8")
        msgs = adapter().item_overlap_problems(items, tmp)
        self.assertTrue([m for m in msgs if "quoted text" in m and "decoy" in m], msgs)
        self.assertTrue([m for m in msgs if "target text" in m and "i1" in m], msgs)
        (tmp / "train.jsonl").write_text(json.dumps({"prompt_id": "x", "prompt": 'a sign that reads "Bye"',
                                                     "target": "Bye"}) + "\n", encoding="utf-8")
        self.assertEqual(adapter().item_overlap_problems(items, tmp), [])
        bad_item = [dict(items[0], target="something else")]
        self.assertTrue([m for m in adapter().item_overlap_problems(bad_item, tmp) if "quoted text" in m])

    def test_item_target_refused_at_collect(self):
        sc = self.fresh()
        rows = F.read_jsonl(sc.items_dir / "items.jsonl")
        rows[2]["target"] = rows[2]["target"] + " extra"
        F.write_jsonl(sc.items_dir / "items.jsonl", rows)
        self.refused("EXT-CONTAMINATION", "artifacts.items.uri", sc=sc, text="quoted text")

    def test_training_payloads(self):
        sc = self.fresh()
        texts, why = adapter().training_payloads(sc.stage_out, sc.base, [(6, sc.final_cut)], data=sc.data)
        self.assertIsNone(why)
        shard = {r["prompt"] for r in F.read_jsonl(sc.data / "train.jsonl")}
        self.assertEqual(texts, shard | {s["prompt"] for s in F.read_jsonl(sc.stage_out / "samples.jsonl")})


# ---------------------------------------------------------------------------
# EXT-169: seed, pins, profile, last cut, script, sampler, EMA, reward
# ---------------------------------------------------------------------------

class TestRunChecks(FlowCase):

    def test_seed(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(seed=1))
        self.refused("EXT-LAUNCH", "--launch-record.training_seed", sc=sc)

    def test_cut_seed(self):
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "cut.json", lambda d: d.update(seed=1))
        self.refused("EXT-LAUNCH", "--launch-record.training_seed", sc=sc, text="cut.json")

    def test_framework_version(self):
        sc = self.fresh()
        p = sc.stage_out / "env-freeze.txt"
        p.write_text(p.read_text().replace("diffusers==0.42.1", "diffusers==0.41.0"))
        self.refused("EXT-FREEZE", "--stage-out", sc=sc)

    def test_pins_and_reference_line(self):
        for drop, text in (("peft==", "peft"), ("paddleocr==", "paddleocr"), ("paddlepaddle==", "paddlepaddle"),
                           ("# flow_grpo algorithm reference", "yifan123/flow_grpo")):
            with self.subTest(drop=drop):
                sc = self.fresh()
                p = sc.stage_out / "env-freeze.txt"
                p.write_text("".join(l + "\n" for l in p.read_text().splitlines() if not l.startswith(drop)))
                self.refused("EXT-FREEZE", "--stage-out", sc=sc, text=text)

    def test_profile(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(profile="smoke"))
        self.refused("EXT-PROFILE", "--launch-record.profile", sc=sc)

    def test_last_cut(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["settings"].update(epochs=4))
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="epochs")

    def test_script(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "env-runtime.json", lambda d: d.update(script_sha256="sha256:" + "2" * 64))
        self.refused("EXT-LAUNCH", "--launch-record.script_sha256", sc=sc)
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda d: d.update(script_sha256="sha256:" + "2" * 64))
        self.refused("EXT-LAUNCH", "--launch-record.script_sha256", sc=sc, text="run_args.json")

    def test_sampler_code(self):
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "cut.json", lambda d: d.update(sampler_code_sha256="sha256:" + "3" * 64))
        self.refused("EXT-SAMPLER", "--stage-out", sc=sc, text="sampler_code_sha256")

    def test_sampler_code_not_the_evaluators(self):
        """Consistent everywhere in the run, but not the evaluator's SHARED SAMPLER block (README item 5)."""
        sc = self.fresh()
        other = "sha256:" + "4" * 64
        for cut in sc.cuts:
            sc.edit_json(cut / "cut.json", lambda d: d.update(sampler_code_sha256=other))
        sc.edit_json(sc.stage_out / "env-runtime.json", lambda d: d.update(sampler_code_sha256=other))
        sc.edit_json(sc.stage_out / "run_args.json", lambda d: d.update(sampler_code_sha256=other))
        self.refused("EXT-SAMPLER", "--stage-out", sc=sc, text="evaluator's sampler code")

    def test_sampler_sha256(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda d: d.update(sampler_sha256="sha256:" + "5" * 64))
        self.refused("EXT-SAMPLER", "--stage-out", sc=sc, text="sampler_sha256")

    def test_cut_sampler_not_registered(self):
        sc = self.fresh(reg_mutate=lambda r: r["sampler"].update(num_steps=40))
        self.refused("EXT-SAMPLER", "registration.sampler", sc=sc)

    def test_ema_refused(self):
        """Decision "Area 5 decisions" (not published): EMA off, a cut holds the raw LoRA weights."""
        for case in ("cut", "adapter_config", "settings"):
            with self.subTest(case=case):
                sc = self.fresh()
                if case == "cut":
                    sc.edit_json(sc.final_cut / "cut.json", lambda d: d.update(ema=True))
                elif case == "adapter_config":
                    sc.edit_json(sc.final_cut / "adapter_config.json", lambda d: d.update(weights="ema"))
                    sc.reseal(sc.final_cut)
                else:
                    sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["settings"].update(ema=True))
                self.refused("EXT-STAGE", "--stage-out", sc=sc, text="EMA")

    def test_reward_not_ocr(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["settings"].update(reward="ocr+brightness_sanity"))
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="settings.reward")


# ---------------------------------------------------------------------------
# EXT-170: ext-arm-fill on the evaluator's arm-<metric> directories
# ---------------------------------------------------------------------------

class TestArmFill(FlowCase):

    def meta(self, arm: Path) -> dict:
        return json.loads((arm / "eval_meta.json").read_text(encoding="utf-8"))

    def test_baseline_arm(self):
        """The base alone (with its `.cache/`): model_hash = the base hash without `.cache/`; the evaluator's compact
        sampler hash (floats such as 4.5 and 1.0) is converted to the RFC 8785 hash."""
        sc = self.fresh(filled=False)
        arm = sc.base_eval / f"arm-{F.METRIC}"
        self.assertNotEqual(F.compact_hash(F.SAMPLER), sha_of(F.SAMPLER))
        proc = sc.fill(arm, sc.base, base=sc.base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        meta = self.meta(arm)
        self.assertEqual(meta["checkpoint"], {"model_hash": sc.base_model, "base_model_hash": sc.base_model})
        self.assertEqual(meta["item_set_hash"], sc.item_set_hash)
        self.assertEqual(meta["sampler_hash"], sha_of(F.SAMPLER))
        self.assertIn("converted sampler_hash", proc.stdout)

    def test_candidate_arm_needs_the_base(self):
        sc = self.fresh(filled=False)
        arm = sc.cand_eval / f"arm-{F.METRIC}"
        proc = sc.fill(arm, sc.final_cut)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["field"] == "--base"])
        proc = sc.fill(arm, sc.final_cut, base=sc.base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(self.meta(arm)["checkpoint"], {"model_hash": sc.cand_model, "base_model_hash": sc.base_model})

    def test_supporting_arm(self):
        sc = self.fresh(filled=False)
        arm = sc.cand_eval / f"arm-{F.SUPPORT_METRIC}"
        proc = sc.fill(arm, sc.final_cut, base=sc.base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(self.meta(arm)["evaluator_id"], F.SUPPORT_ID)

    def test_fill_then_collect(self):
        sc = self.fresh(filled=False)
        for arm, ck in ((sc.baseline, sc.base), (sc.candidate, sc.final_cut)):
            proc = sc.fill(arm, ck, base=sc.base)
            self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.passes(sc)


# ---------------------------------------------------------------------------
# EXT-171: registration (sampler keys, construct alignment)
# ---------------------------------------------------------------------------

class TestRegistration(unittest.TestCase):

    def reg(self=None) -> dict:
        text = (F.FG / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", "sha256:" + "a" * 64).replace("{sampler}", json.dumps(F.SAMPLER))
        return yaml.safe_load(text.replace("{evaluator_hash}", "sha256:" + "b" * 64))

    def fields(self, reg: dict, evaluator: str = F.EVALUATOR_ID) -> list[str]:
        from gatedtrain.registration import check_contents
        return [p.field for p in check_contents(reg, evaluator_id=evaluator)]

    def test_template_accepted(self):
        self.assertEqual(self.fields(self.reg()), [])

    def test_sampler_keys(self):
        reg = self.reg()
        reg["sampler"]["eta"] = 1.0
        del reg["sampler"]["sigmas"]
        f = self.fields(reg)
        self.assertIn("registration.sampler.eta", f)
        self.assertIn("registration.sampler.sigmas", f)

    def test_construct_alignment_required(self):
        reg = self.reg()
        del reg["construct_alignment"]
        self.assertIn("registration.construct_alignment", self.fields(reg))

    def test_not_allowed_for_the_t0_arm(self):
        reg = self.reg()
        reg["evaluators"] = [{"id": F.SUPPORT_ID, "manifest_hash": "sha256:" + "c" * 64}]
        reg["clause"]["metric"] = F.SUPPORT_METRIC
        self.assertIn("registration.construct_alignment", self.fields(reg, F.SUPPORT_ID))
        del reg["construct_alignment"]
        self.assertNotIn("registration.construct_alignment", self.fields(reg, F.SUPPORT_ID))

    def test_shared_annotation_refused(self):
        reg = self.reg()
        reg["shared_annotation"] = "no shared annotation here"
        self.assertIn("registration.shared_annotation", self.fields(reg))


# ---------------------------------------------------------------------------
# EXT-173: host rows and launch; EXT-174: smoke on the deciding metric
# ---------------------------------------------------------------------------

class TestLaunch(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-flow-launch-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.sc = F.FlowGrpoScenario(self.tmp / "sc", n_items=12)
        self.out = self.sc.output_root / "area5-sd35" / "full-2"
        self.out.mkdir()

    def refused(self, proc, out, rule, field):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        self.assertTrue([e for e in errs if e["rule"] == rule and e["field"] == field], f"{rule} {field}: {errs}")
        self.assertFalse(out.exists())

    def test_gpu_request_accepted(self):
        proc, out = self.sc.launch_request(self.sc.request(self.out), "ok")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn("--runtime=nvidia", proc.stdout)
        rec = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual((rec["adapter"], rec["framework"]["name"]), (F.ADAPTER, "diffusers"))
        self.assertIn("DOCTR_CACHE_DIR", rec["env_names"])
        proc2 = run_cli(["ext-launch", "--adapter", F.ADAPTER, "--recheck", str(out)], cwd=self.sc.root, env=self.sc.env())
        self.assertEqual(proc2.returncode, EXIT_OK, describe(proc2))

    def test_hf_token_refused(self):
        """README "Launch request facts": HF_TOKEN is never passed (the base is gated and is only hashed locally)."""
        req = self.sc.request(self.out)
        req["env"]["HF_TOKEN"] = "hf_" + "x" * 20
        proc, out = self.sc.launch_request(req, "token")
        self.refused(proc, out, "EXT-LAUNCH", "--request.env.HF_TOKEN")
        self.assertNotIn("hf_" + "x" * 20, proc.stdout + proc.stderr)

    def test_other_env_name_refused(self):
        req = self.sc.request(self.out)
        req["env"]["HF_HOME"] = "/tmp/hf"
        proc, out = self.sc.launch_request(req, "hfhome")
        self.refused(proc, out, "EXT-LAUNCH", "--request.env.HF_HOME")

    def test_doctr_cache_is_a_path_name(self):
        req = self.sc.request(self.out)
        req["env"]["DOCTR_CACHE_DIR"] = "/home/user/.cache/doctr"
        proc, out = self.sc.launch_request(req, "doctr")
        self.refused(proc, out, "EXT-LAUNCH", "--request.env.DOCTR_CACHE_DIR")

    def test_ema_refused_at_launch(self):
        req = self.sc.request(self.out)
        req["argv"] = req["argv"] + ["--ema"]
        proc, out = self.sc.launch_request(req, "ema")
        self.refused(proc, out, "EXT-LAUNCH", "--request.argv")

    def test_spec_optional_and_checked(self):
        proc, out = self.sc.launch_request(self.sc.request(self.out), "spec",
                                           extra=["--spec", str(self.sc.paths["spec"])])
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        req = self.sc.request(self.out)
        req["mounts"] = [m for m in req["mounts"] if m["mode"] == "rw"]
        proc, out = self.sc.launch_request(req, "nospec", extra=["--spec", str(self.sc.paths["spec"])])
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        fields = [e["field"] for e in parse_errors(proc.stderr)]
        self.assertIn("artifacts.base.uri", fields)
        self.assertIn("artifacts.prompts.uri", fields)

    def test_registry_host_rows(self):
        reg = registry()
        from gatedtrain.ext.rowkit import python_tree
        from gatedtrain_recipe_flow_grpo_sd35_ocr.rows import OUT_NAME, RUNTIME
        tree = python_tree(RUNTIME)
        self.assertEqual(reg.input_roots(F.ADAPTER), ((str(Path(tree).parent),) if tree else ())
                         + (RUNTIME, "{output_root}/" + OUT_NAME))
        self.assertEqual(set(reg.env_names(F.ADAPTER)), set(F.ENV_NAMES) | {"PYTORCH_CUDA_ALLOC_CONF"})
        self.assertNotIn("HF_TOKEN", reg.env_names(F.ADAPTER))
        self.assertEqual(set(reg.env_paths(F.ADAPTER)), {"HOME", "XDG_CACHE_HOME", "DOCTR_CACHE_DIR"})
        self.assertEqual(reg.scan_skip(F.ADAPTER)[0], f"{RUNTIME}/.venv")
        self.assertEqual(reg.interpreter(F.ADAPTER), f"{RUNTIME}/.venv/bin/python")
        self.assertEqual(reg.interpreter_trees(F.ADAPTER), (tree,) if tree else ())

    def test_full_run_smoke_on_the_deciding_metric(self):
        """EXT-174 (MM-38): the smoke is judged on ocr_b_acc ([0, 1]: mean, mode and sd rules); a baseline that
        renders nothing for half the items is degenerate."""
        good = write_canonical(self.tmp / "smoke.json", F.smoke_record(base_model=self.sc.base_model,
                                                                       item_set=self.sc.item_set_hash))
        proc, out = self.sc.launch_request(self.sc.request(self.out), "full", profile="full", smoke=good)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        flat = write_canonical(self.tmp / "smoke-flat.json", F.smoke_record(
            base_model=self.sc.base_model, item_set=self.sc.item_set_hash,
            baseline_scores=[0.0] * 20 + [0.25 * (i % 4) + 0.1 for i in range(20)]))
        proc, out = self.sc.launch_request(self.sc.request(self.out), "full-flat", profile="full", smoke=flat)
        self.refused(proc, out, "EXT-SMOKE", "--smoke-record")
        self.assertIn("degenerate_mode", proc.stderr)


class TestArgv(unittest.TestCase):
    """EXT-173 (as EXT-133, EXT-150, EXT-156): the launch-side argv rules, options resolved as argparse does."""

    RT = "/srv/recipes/flow-grpo-sd35-ocr/runtime"

    def problems(self, argv: list, profile: str | None = None) -> list[str]:
        return adapter().argv_problems(argv, profile=profile)

    def test_training(self):
        py, tr = f"{self.RT}/.venv/bin/python", f"{self.RT}/train_flow_grpo.py"
        ok = [py, tr, "--profile", "smoke", "--out", "/out", "--seed", "5000"]
        self.assertEqual(self.problems(ok), [])
        self.assertEqual(self.problems([py, tr, "--profile", "full", "--out", "/out", "--seed", "5000", "--epochs", "10"],
                                       "full"), [])
        for bad in (["--ema"], ["--em"], ["--no-verify-base"], ["--no-verify"]):
            with self.subTest(bad=bad):
                self.assertTrue(self.problems(ok + bad), bad)
        for prof in ("cpu-sanity", "--profile=cpu-sanity"):
            argv = [py, tr, "--out", "/out"] + (["--profile", prof] if not prof.startswith("--") else [prof])
            self.assertTrue(self.problems(argv), argv)

    def test_evaluation(self):
        py, ev = f"{self.RT}/.venv/bin/python", f"{self.RT}/eval_images.py"
        base = [py, ev, "--base-only", "--out", "/out", "--eval-seeds", "11"]
        self.assertEqual(self.problems(base), [])
        self.assertEqual(self.problems(base + ["--base", f"{self.RT}/models/stable-diffusion-3.5-medium"]), [])
        self.assertEqual(self.problems(base + ["--items", f"{self.RT}/items"]), [])
        for bad in (["--base", "/inputs/other"], ["--items", "/inputs/items-copy"], ["--items=/inputs/x"]):
            with self.subTest(bad=bad):
                self.assertTrue(self.problems(base + bad))
        self.assertEqual(self.problems(base + ["--limit", "40"], "smoke"), [])
        self.assertTrue(self.problems(base + ["--limit", "40"], "full"))
        self.assertTrue(self.problems(base + ["--limit=40"], "full"))

    def test_script_pinned(self):
        py = f"{self.RT}/.venv/bin/python"
        for bad in ("/srv/out/x/eval_images.py", f"{self.RT}/fetch_models.py",
                    "/srv/recipes/ddpo-sd15/runtime/eval_images.py"):
            with self.subTest(script=bad):
                self.assertTrue(self.problems([py, bad, "--out", "/out"]))


# ---------------------------------------------------------------------------
# EXT-175: unreadable files and the runtime's real outputs
# ---------------------------------------------------------------------------

class TestUnreadable(FlowCase):

    def test_last_cut_file(self):
        sc = self.fresh()
        self.unreadable(sc.final_cut / "trainer_state.json")
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="cannot be read")

    def test_freeze(self):
        sc = self.fresh()
        self.unreadable(sc.stage_out / "env-freeze.txt")
        self.refused("EXT-FREEZE", "--stage-out", sc=sc)

    def test_base_file(self):
        sc = self.fresh()
        self.unreadable(sc.base / "model_index.json")
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc)


class TestRuntimeFacts(unittest.TestCase):
    """The real cpu-sanity outputs (fixtures, copied) and the runtime files (skipped where absent)."""

    def test_real_logs_pass_the_consumption_check(self):
        """The 2026-11-13 cpu-sanity run's samples.jsonl, epoch_plan.jsonl, train_steps.jsonl, run_args.json and cut
        marker, against the two shard rows it sampled: no problem, 4 consumed records (its reward adds a brightness
        term, so `reward` = `ocr_acc` is not required there; collect refuses that profile anyway)."""
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-flow-real-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        out = tmp / "run"
        shutil.copytree(F.CPU, out)
        data = tmp / "prompts"
        data.mkdir()
        shutil.move(str(out / "train.jsonl"), str(data / "train.jsonl"))
        a = adapter()
        cuts = [(2, out / "checkpoint-1")]
        self.assertEqual(a.consumption_problems(out, data, cuts), [])
        cons = a.consumption(out, data, cuts, run_id="r")
        self.assertEqual([r["outcome"] for r in cons["rows"]], ["consumed"] * 4)
        self.assertEqual(sorted(r["step"] for r in cons["rows"]), [1, 1, 2, 2])

    def test_real_cut_marker_shape(self):
        doc = json.loads((F.CPU / "checkpoint-1" / "cut.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], "ptc_mm.flow_grpo_sd35_cut")
        self.assertEqual(sorted(doc["files"]), sorted(F.CUT_FILES))
        self.assertEqual(doc["deployable"], list(F.DEPLOYABLE))
        self.assertIs(doc["ema"], False)
        self.assertEqual(doc["sampler_code_sha256"], F.SAMPLER_CODE_SHA_RUN)

    def arm_problems(self, lines: list[dict]) -> list[str]:
        from gatedtrain.ext.arm import read_lines
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-flow-arm-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        for line in lines:
            line["output_ref"] = None                 # the images are not copied into the fixtures
        F.write_jsonl(tmp / "items.jsonl", lines)
        _, why = read_lines(tmp, seeds=[11], score_range=(0, 1), error_classes=("scorer_error",))
        return why

    def test_real_arm_lines_accepted(self):
        """The arm lines of the 2026-11-14 CPU evaluation (eval_images.py fdad8eb9..., decision of 2026-11-14: numeric
        sub_metrics only) pass the MM-1 line rules."""
        lines = F.read_jsonl(F.FG / "items.ocr_b_acc.cpu-sanity.jsonl")
        self.assertEqual(sorted(lines[0]["sub_metrics"]), ["nonfinite", "ocr_a_acc", "ocr_b_acc"])
        self.assertEqual(self.arm_problems(lines), [])

    def test_string_sub_metrics_refused(self):
        """The earlier eval_images.py (c28fd482...) wrote the target and the OCR texts into sub_metrics; such a line
        is refused (MM-1), as is a null value."""
        for extra in ({"target": "Spring Collection 2024"}, {"ocr_a_acc": None}):
            with self.subTest(extra=extra):
                lines = F.read_jsonl(F.FG / "items.ocr_b_acc.cpu-sanity.jsonl")
                lines[0]["sub_metrics"].update(extra)
                self.assertTrue([w for w in self.arm_problems(lines) if "sub_metrics" in w])

    def test_fixture_scorers_are_the_real_arms(self):
        for name, row_id in (("ocr_b_acc", F.EVALUATOR_ID), ("ocr_a_acc", F.SUPPORT_ID)):
            meta = json.loads((F.FG / f"eval_meta.{name}.cpu-sanity.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["evaluator_id"], meta["metric"]), (row_id, name))
            self.assertEqual(meta["scorer"]["code_sha256"], F.EVAL_IMAGES_SHA256_RUN)
            self.assertEqual(set(meta["sampler"]), set(F.SAMPLER_KEYS))

    def test_fixture_sampler_keys(self):
        """The fixture deployment sampler has exactly the keys of the cpu-sanity cut's sampler.json."""
        real = json.loads((F.CPU / "checkpoint-1" / "sampler.json").read_text(encoding="utf-8"))
        self.assertEqual(set(F.SAMPLER), set(real))
        self.assertEqual((F.SAMPLER["num_steps"], F.SAMPLER["noise_level"], F.SAMPLER["guidance_scale"],
                          F.SAMPLER["resolution"], len(F.SAMPLER["sigmas"])), (10, 0.0, 4.5, 512, 11))

    @unittest.skipUnless(F.EVAL_IMAGES.is_file() and F.TRAIN_SCRIPT.is_file(), "the area-5 runtime is absent")
    def test_runtime_hashes(self):
        self.assertEqual(file_sha256(F.EVAL_IMAGES), F.EVAL_IMAGES_SHA256)
        self.assertEqual(F.shared_sampler_sha256(F.TRAIN_SCRIPT), F.SAMPLER_CODE_SHA)
        self.assertEqual(F.shared_sampler_sha256(F.EVAL_IMAGES), F.SAMPLER_CODE_SHA)
        from gatedtrain_recipe_flow_grpo_sd35_ocr import adapter as mod
        self.assertEqual(mod.SAMPLER_CODE_SHA256, F.SAMPLER_CODE_SHA)

    @staticmethod
    def runtime_module():
        import importlib.util
        spec = importlib.util.spec_from_file_location("area5_eval_images", F.EVAL_IMAGES)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                  # top level imports the standard library only
        return mod

    @staticmethod
    def brute_substring_accuracy(recognized: str, target: str) -> float:
        """The decision's definition, literally: 1 - min over substrings s of lev(s, target) / len(target)."""
        def lev(a, b):
            prev = list(range(len(b) + 1))
            for i, ca in enumerate(a, 1):
                cur = [i]
                for j, cb in enumerate(b, 1):
                    cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
                prev = cur
            return prev[-1]
        r, t = recognized.replace(" ", "").lower(), target.replace(" ", "").lower()
        if not r:
            return 0.0
        return 1 - min(lev(r[i:j], t) for i in range(len(r) + 1) for j in range(i, len(r) + 1)) / len(t)

    def test_deciding_formula_is_the_substring_one(self):
        """Smoke finding of 2026-11-14 (runtime/README.md "Decisions taken"): the scorer version of ocr_b_acc names the
        approximate-substring formula; the supporting arm keeps flow_grpo's (no such phrase)."""
        rows = registry().EVALUATOR_ROWS
        self.assertIn("approximate-substring edit distance", rows[F.EVALUATOR_ID]["scorer"]["version"])
        self.assertNotIn("substring", rows[F.SUPPORT_ID]["scorer"]["version"])
        self.assertIn("another formula", rows[F.EVALUATOR_ID]["construct_aligned"]["statement"])

    @unittest.skipUnless(F.EVAL_IMAGES.is_file(), "the area-5 runtime is absent")
    def test_runtime_substring_scorer(self):
        """eval_images.py `ocr_substring_accuracy` (Sellers DP) equals the brute-force definition on random strings
        (spaces and case included), scores an empty recognition 0 and a contained target 1; flow_grpo's
        `ocr_accuracy` (the reward, in the shared sampler block) is unchanged."""
        import random
        mod = self.runtime_module()
        rng = random.Random(20261114)
        for _ in range(400):
            r = "".join(rng.choice("abcAB c") for _ in range(rng.randint(0, 12)))
            t = "".join(rng.choice("abcAB c") for _ in range(rng.randint(1, 7)))
            if not t.replace(" ", ""):
                continue
            self.assertAlmostEqual(mod.ocr_substring_accuracy(r, t), self.brute_substring_accuracy(r, t), places=12,
                                   msg=(r, t))
        self.assertEqual(mod.ocr_substring_accuracy("", "Open"), 0.0)
        self.assertEqual(mod.ocr_substring_accuracy("xxTRY OUR NEW BURGERyy", "Try Our New Burger"), 1.0)
        self.assertEqual(mod.ocr_substring_accuracy("noiseOpemnoise", "Open"), 0.75)
        with self.assertRaises(ValueError):
            mod.ocr_substring_accuracy("abc", "  ")

    # The GPU smoke base evaluation's scores (results/, a copy of the run's scores.jsonl).
    SMOKE_SCORES = (Path(__file__).resolve().parents[1] / "results" / "2026-11-14" / "smoke-evidence"
                    / "eval-smoke-base-1" / "scores.jsonl")

    @unittest.skipUnless(F.EVAL_IMAGES.is_file() and SMOKE_SCORES.is_file(), "the area-5 smoke evaluation is absent")
    def test_smoke_rows_recomputed(self):
        """The 40 docTR texts of the GPU smoke base evaluation (read only): the substring formula gives share at 0
        0.175 and mean 0.412 (the decision's numbers); the old whole-text formula gave 0.525 at 0."""
        mod = self.runtime_module()
        rows = F.read_jsonl(self.SMOKE_SCORES)
        new = [mod.ocr_substring_accuracy(r["ocr_b_text"], r["target"]) for r in rows]
        self.assertEqual(len(new), 40)
        self.assertEqual(sum(v == 0 for v in new) / 40, 0.175)
        self.assertAlmostEqual(sum(new) / 40, 0.412, places=3)
        self.assertEqual(sum(r["ocr_b_acc"] == 0 for r in rows) / 40, 0.525)

    @unittest.skipUnless((F.AREA5 / "items" / "items.jsonl").is_file(), "the area-5 runtime is absent")
    def test_real_items(self):
        from gatedtrain.ext.collect import read_item_set
        row = registry().EVALUATOR_ROWS[F.EVALUATOR_ID]
        rows, bad = read_item_set(F.AREA5 / "items", tuple(row["item_payload"]))
        self.assertEqual((len(rows), bad), (400, []))


# ---------------------------------------------------------------------------
# EXT-228: the flow_grpo family; area 5 keeps its facts (moved from gatedtrain's TestSiblings with the recipe)
# ---------------------------------------------------------------------------

class TestFamily(unittest.TestCase):

    def test_area5_facts(self):
        """The hooks a sibling adapter on the same layout overrides (gatedtrain's Wan adapter) keep area 5's values; the
        adapter extends the flow_grpo family base directly, not another recipe's adapter."""
        from gatedtrain.families.flow_grpo import FlowGrpoAdapter
        from gatedtrain_recipe_flow_grpo_sd35_ocr import adapter as a5
        cls = a5.FlowGrpoSd35Adapter
        self.assertEqual((cls.sample_id_domain, cls.sampler_code_sha256, cls.base_required, cls.pins),
                         ("flow-grpo-sd35", a5.SAMPLER_CODE_SHA256, a5.BASE_REQUIRED, a5.PINS))
        self.assertEqual(cls.__bases__, (FlowGrpoAdapter,))
        self.assertIn("not an SD3.5 Medium directory", cls.base_dir_problem(Path("/nonexistent/nothing")) or "")
        self.assertEqual(registry().adapter(F.ADAPTER)._settings_problems({"reward": "anything"}), [])

    def test_uses_id(self):
        """MM-13: `trainer_ids()` maps the spec's `uses` id `rl.flow_grpo@0.1` to this adapter."""
        self.assertEqual(registry().trainer_ids()[F.TRAINER_ID], F.ADAPTER)


if __name__ == "__main__":
    unittest.main()
