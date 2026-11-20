"""The runtime scripts that run inside the GPU container (runtime/train_flow_grpo.py, runtime/eval_images.py,
runtime/check_edit_distance.py), on CPU and without torch, diffusers, PaddleOCR or docTR: the OCR reward and the
deciding metric's formulas, the settings, the advantages and the zero-advantage filter, the cut layout, the base
binding and the run files, checked against what the adapter and gatedtrain read from their outputs at collect, and
against the committed outputs of real runs (the cpu-sanity run under fixtures/flow_grpo/cpu-sanity, the smoke base
evaluation under results/2026-11-14/smoke-evidence).

Why: these scripts run only on a GPU, so a fault in a part that needs no GPU (the reward or the metric formula, the
target text of a prompt, the filter, the sample-id encoding, the cut layout) was found only after GPU time, or never
(a wrong reward trains silently).

The scripts import only the standard library at module level, so they load as they are. The edit distance comes from
rapidfuzz, which CI does not install: when it is missing, `rapidfuzz.distance.Levenshtein.distance` is a plain
dynamic-programming Levenshtein distance defined here (the formulas around it are what is tested; the published run's
python-Levenshtein -> rapidfuzz equivalence is check_edit_distance.py's job). Nothing in runtime/ is changed: the
scripts' SHA-256s are recorded in RESULT.md, eval_images.py's is the registered scorer pin and the shared sampler
block's is the adapter's SAMPLER_CODE_SHA256.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import random
import re
import shutil
import stat
import struct
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

import yaml

from gatedtrain.ext.sdk import compact_sha256, derive_seed as sdk_derive_seed, safetensors_problem
from gatedtrain.families.flow_grpo import target_text as family_target_text

from gatedtrain_recipe_flow_grpo_sd35_ocr import adapter as A
from gatedtrain_recipe_flow_grpo_sd35_ocr import rows as R

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime"
FG = Path(__file__).resolve().parent / "fixtures" / "flow_grpo"
CPU = FG / "cpu-sanity"
SMOKE_SCORES = REPO / "results" / "2026-11-14" / "smoke-evidence" / "eval-smoke-base-1" / "scores.jsonl"
GATE = REPO / "results" / "2026-11-14" / "gate"


def levenshtein(a: str, b: str) -> int:
    """Reference Levenshtein distance (insert, delete, substitute; unit costs)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


HAVE_RAPIDFUZZ = importlib.util.find_spec("rapidfuzz") is not None
if not HAVE_RAPIDFUZZ:
    _rf = types.ModuleType("rapidfuzz")
    _rf_distance = types.ModuleType("rapidfuzz.distance")
    _rf_lev = types.ModuleType("rapidfuzz.distance.Levenshtein")
    _rf_lev.distance = levenshtein
    _rf.distance = _rf_distance
    _rf_distance.Levenshtein = _rf_lev
    sys.modules.update({"rapidfuzz": _rf, "rapidfuzz.distance": _rf_distance, "rapidfuzz.distance.Levenshtein": _rf_lev})


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = load_script(RUNTIME / "train_flow_grpo.py", "_sd35_runtime_train")
E = load_script(RUNTIME / "eval_images.py", "_sd35_runtime_eval")
C = load_script(RUNTIME / "check_edit_distance.py", "_sd35_runtime_check")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def help_options(parse_args) -> set[str]:
    buf = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
        parse_args(["--help"])
    return set(re.findall(r"(--[a-z][a-z-]*)", buf.getvalue())) - {"--help"}


class FakeTransformer:
    """`SD3Transformer2DModel.save_lora_adapter` as far as save_cut uses it: a safetensors file, mode 0600, whose
    header tensor table and LoraConfig metadata (target_modules a list of a set) come in an arbitrary order."""

    def __init__(self, order_seed: int = 0):
        self.order_seed = order_seed

    def save_lora_adapter(self, save_directory, adapter_name, safe_serialization, weight_name):
        assert adapter_name == "default" and safe_serialization is True
        rnd = random.Random(self.order_seed)
        targets = list(T.LORA["target_modules"])
        rnd.shuffle(targets)
        names = [f"transformer_blocks.0.{t}.lora_{ab}.weight" for t in ("attn.to_q", "attn.add_k_proj") for ab in "AB"]
        tensors = {n: {"dtype": "F32", "shape": [2], "data_offsets": [8 * i, 8 * i + 8]} for i, n in enumerate(names)}
        keys = list(tensors)
        rnd.shuffle(keys)
        meta = {"format": "pt", "lora_adapter_metadata": json.dumps({"r": 32, "target_modules": targets})}
        text = json.dumps({"__metadata__": meta, **{k: tensors[k] for k in keys}}, separators=(",", ":")).encode()
        path = Path(save_directory) / weight_name
        path.write_bytes(struct.pack("<Q", len(text)) + text + bytes(range(8 * len(names))))
        os.chmod(path, 0o600)


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sd35-runtime-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def patch(self, obj, name, value):
        old = getattr(obj, name)
        setattr(obj, name, value)
        self.addCleanup(setattr, obj, name, old)

    def base(self) -> Path:
        """A tiny plain base directory with a Hugging Face `.cache/` (left out by both the binding and the adapter),
        and MODELS.json for it, patched into both scripts (they read HERE/MODELS.json)."""
        d = self.tmp / "base"
        files = {"model_index.json": b"{}\n", "transformer/diffusion_pytorch_model.safetensors": b"weights",
                 "scheduler/scheduler_config.json": b"{}\n"}
        for rel, data in {**files, ".cache/huggingface/download/x.metadata": b"etag"}.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(data)
        record = {"repo": "stabilityai/stable-diffusion-3.5-medium", "revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
                  "files": {rel: {"sha256": T.sha256_bytes(data)} for rel, data in files.items()}}
        mj = self.tmp / "MODELS.json"
        mj.write_text(json.dumps({"models": {T.BASE_NAME: record}}), encoding="utf-8")
        self.patch(T, "MODELS_JSON", mj)
        self.patch(E, "MODELS_JSON", mj)
        return d


# ---------------------------------------------------------------------------
# The OCR reward (engine A, flow_grpo's formula) and the deciding metric (engine B, approximate substring)
# ---------------------------------------------------------------------------

class TestTargets(unittest.TestCase):
    def prompts(self):
        return ([r["prompt"] for r in read_jsonl(RUNTIME / "items" / "items.jsonl")]
                + [r["prompt"] for r in read_jsonl(FG / "train.jsonl")] + [r["prompt"] for r in read_jsonl(CPU / "train.jsonl")])

    def test_every_prompt_has_a_scorable_target(self):
        # PaddleOcrA.__call__ takes the target outside its try: a prompt without a quoted span would stop the run;
        # an empty normalized target divides by zero in ocr_accuracy (the reward) and raises in ocr_b_acc.
        for p in self.prompts():
            t = T.target_text(p)
            self.assertEqual(t, E.target_text(p))
            self.assertEqual(t, family_target_text(p))
            self.assertTrue(t.replace(" ", ""), p)

    def test_items_carry_their_targets(self):
        items = E.read_items(RUNTIME / "items")
        self.assertEqual(len(items), 400)
        for it in items:
            self.assertEqual(it["target"], E.target_text(it["prompt"]))
        meta = json.loads((RUNTIME / "items" / "item_set_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["items_sha256"], E.sha256_file(RUNTIME / "items" / "items.jsonl"))
        # The training prompts the tests carry are disjoint from the items by target (make_items' rule).
        item_targets = {E.norm_target(it["prompt"]) for it in items}
        for r in read_jsonl(FG / "train.jsonl"):
            self.assertNotIn(E.norm_target(r["prompt"]), item_targets)

    def test_target_is_the_first_quoted_span(self):
        for f in (T.target_text, E.target_text, family_target_text):
            self.assertEqual(f('a sign "Open" and "Closed"'), "Open")
            self.assertEqual(f('"" then "x"'), "")
            self.assertEqual(f('a sign " Hi there  " here'), " Hi there  ")        # kept as is (the samples' target)
        with self.assertRaises(IndexError):
            T.target_text("no quotes")
        self.assertIsNone(family_target_text("no quotes"))


class TestOcrAccuracy(unittest.TestCase):
    def test_flow_grpo_formula(self):
        f = T.ocr_accuracy
        self.assertEqual(f("WELCOME HOME", "Welcome Home"), 1.0)                # spaces and case
        self.assertEqual(f("xx welcome home yy", "Welcome Home"), 1.0)          # a substring scores 1
        self.assertEqual(f("", "abcd"), 0.0)                                    # distance clipped at len(target)
        self.assertEqual(f("a much longer unrelated text", "abc"), 0.0)
        self.assertEqual(f("abxd", "abcd"), 0.75)
        self.assertEqual(f("ab", "abcd"), 0.5)
        with self.assertRaises(ZeroDivisionError):
            f("anything", "   ")
        rnd = random.Random(1)
        alpha = "abcAB 12é"
        for _ in range(300):
            a = "".join(rnd.choice(alpha) for _ in range(rnd.randint(0, 12)))
            b = "".join(rnd.choice(alpha) for _ in range(rnd.randint(1, 8)))
            if not b.replace(" ", ""):
                continue
            ra, rb = a.replace(" ", "").lower(), b.replace(" ", "").lower()
            want = 1.0 if rb in ra else 1 - min(levenshtein(ra, rb), len(rb)) / len(rb)
            self.assertEqual(f(a, b), want)
            self.assertEqual(E.ocr_accuracy(a, b), want)
            self.assertEqual(C.ocr_accuracy(a, b, levenshtein), want)     # the equivalence check's copy
            self.assertTrue(0.0 <= want <= 1.0)

    def test_substring_edit_distance_is_the_minimum_over_substrings(self):
        rnd = random.Random(2)
        for _ in range(200):
            text = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 9)))
            pat = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 6)))
            subs = {text[i:j] for i in range(len(text) + 1) for j in range(i, len(text) + 1)}
            self.assertEqual(E.substring_edit_distance(text, pat), min(levenshtein(s, pat) for s in subs),
                             (text, pat))

    def test_deciding_metric(self):
        f = E.ocr_substring_accuracy
        self.assertEqual(f("noise STEPGOAL achieved noise", "Step Goal Achieved"), 1.0)
        self.assertEqual(f("SteepGoalAchieved43VousMita", "Step Goal Achieved"), 1 - 1 / 16)
        self.assertEqual(f("", "abc"), 0.0)
        self.assertEqual(f("zzzz", "abc"), 0.0)
        with self.assertRaises(ValueError):
            f("abc", "  ")
        rnd = random.Random(3)
        for _ in range(200):
            a = "".join(rnd.choice("ab cd") for _ in range(rnd.randint(0, 10)))
            b = "".join(rnd.choice("abcd") for _ in range(rnd.randint(1, 5)))
            self.assertTrue(0.0 <= f(a, b) <= 1.0)
            self.assertGreaterEqual(f(a, b), E.ocr_accuracy(a, b) if a.replace(" ", "") else 0.0)

    def test_smoke_evaluation_scores_are_reproduced(self):
        # The base smoke evaluation's recorded texts, re-scored. That smoke ran before the deciding formula was
        # chosen, so its ocr_b_acc is flow_grpo's whole-text formula on docTR's text (README "Decisions taken"):
        # every recorded score is reproduced, and the 52.5 % of zeros that motivated the change.
        import statistics

        rows = read_jsonl(SMOKE_SCORES)
        self.assertEqual(len(rows), 40)
        for r in rows:
            self.assertFalse(r["errors"])
            self.assertEqual(E.ocr_accuracy(r["ocr_a_text"], r["target"]), r["ocr_a_acc"], r["item"])
            self.assertEqual(E.ocr_accuracy(r["ocr_b_text"], r["target"]), r["ocr_b_acc"], r["item"])
            self.assertEqual(r["noise_seed"], E.derive_seed("eval", r["seed"], r["item"]))
        self.assertEqual(sum(r["ocr_b_acc"] == 0 for r in rows) / len(rows), 0.525)
        # The registration's pilot figures are the published deciding formula on the same texts.
        reg = yaml.safe_load((GATE / "registration-gatedtrain.yaml").read_text(encoding="utf-8"))
        v = [E.ocr_substring_accuracy(r["ocr_b_text"], r["target"]) for r in rows]
        self.assertEqual(round(statistics.mean(v), 3), reg["baseline_pilot_mean"])
        self.assertEqual(round(statistics.stdev(v), 3), 0.301)                    # "sd 0.301"
        self.assertEqual(max(v.count(x) for x in v) / len(v), 0.175)             # "mode share 0.175"

    @unittest.skipUnless(importlib.util.find_spec("Levenshtein") and HAVE_RAPIDFUZZ,
                         "python-Levenshtein and rapidfuzz are not both installed (only the published run's venv has "
                         "python-Levenshtein)")
    def test_check_edit_distance(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(C.main(), 0)


class TestRewardEngine(unittest.TestCase):
    """PaddleOcrA.__call__ (the training reward) without PaddleOCR: its recognizer replaced by a stub."""

    def engine(self, text_fn):
        eng = object.__new__(T.PaddleOcrA)
        eng.text = text_fn
        return eng

    def test_rewards_and_the_error_path(self):
        prompts = ['a sign that reads "Open Late"', 'a cake with "Happy 30th"']
        out = self.engine(lambda arr: {"a": "OPEN LATE!", "b": "hapy 30"}[arr])(["a", "b"], prompts)
        self.assertEqual(out[0], {"text": "OPEN LATE!", "reward": 1.0, "error": None})
        self.assertEqual(out[1]["reward"], 1 - 3 / 9)                     # "hapy30" -> "happy30th"

        def boom(arr):
            raise RuntimeError("paddle failed")

        out = self.engine(boom)(["a"], prompts[:1])
        # The adapter's rule for an OCR error: ocr_acc 0 and no ocr_text (adapter._reward_field_problems).
        self.assertEqual(out, [{"text": None, "reward": 0.0, "error": "RuntimeError: paddle failed"}])
        x = {"reward": out[0]["reward"], "ocr_acc": out[0]["reward"], "ocr_text": out[0]["text"], "ocr_error": out[0]["error"]}
        self.assertEqual(A.FlowGrpoSd35Adapter()._reward_field_problems(x, 1, {}, True), [])


# ---------------------------------------------------------------------------
# Settings, sampler, shared code (pinned: README "Changes against flow_grpo", the registration, the cpu-sanity run)
# ---------------------------------------------------------------------------

class TestSettings(unittest.TestCase):
    def settings(self, *argv):
        return T.resolve_settings(T.parse_args(["--out", "/tmp/x", *argv]))

    def test_cpu_sanity_settings_are_the_recorded_runs(self):
        ra = json.loads((CPU / "run_args.json").read_text(encoding="utf-8"))
        self.assertEqual(self.settings("--profile", "cpu-sanity"), ra["settings"])

    def test_training_settings(self):
        s = self.settings("--profile", "full")
        want = {"num_steps": 10, "guidance_scale": 4.5, "noise_level": 0.7, "resolution": 512, "max_sequence_length": 128,
                "deploy_num_steps": 10, "deploy_noise_level": 0.0, "learning_rate": 3.0e-4, "clip_range": 1.0e-4,
                "adv_clip_max": 5.0, "beta": 0.04, "timestep_fraction": 0.99, "num_train_timesteps": 9,
                "global_std": True, "ema": False, "reward": "ocr", "num_inner_epochs": 1, "precision": "fp16",
                "prompts_per_epoch": 8, "images_per_prompt": 8, "sample_batch": 8, "train_batch": 2,
                "samples_per_epoch": 64, "num_batches_per_epoch": 8, "gradient_accumulation_steps": 4, "epochs": 10,
                "save_every": 2, "seed": 5000}
        self.assertEqual({k: s[k] for k in want}, want)
        self.assertEqual((self.settings("--profile", "smoke")["epochs"], self.settings("--profile", "smoke")["save_every"]),
                         (2, 1))
        self.assertEqual(T.LORA["r"], 32)
        self.assertEqual(T.LORA["lora_alpha"], 64)
        self.assertEqual(sorted(T.LORA["target_modules"]), T.LORA["target_modules"])
        o = self.settings("--profile", "full", "--beta", "0", "--ema", "--epochs", "3", "--train-batch", "4")
        self.assertEqual((o["beta"], o["ema"], o["epochs"], o["train_batch"]), (0.0, True, 3, 4))

    def test_refused_settings(self):
        for patch in ({"sample_batch": 4}, {"prompts_per_epoch": 3, "images_per_prompt": 8}):
            prof = {**T.PROFILES["smoke"], **patch}
            with unittest.mock.patch.dict(T.PROFILES, {"smoke": prof}), self.assertRaises(SystemExit):
                self.settings("--profile", "smoke")

    def test_adapter_knows_every_option(self):
        self.assertEqual(help_options(T.parse_args), set(A.TRAIN_OPTIONS))
        self.assertEqual(help_options(E.parse_args), set(A.EVAL_OPTIONS))
        self.assertEqual(set(A.PROFILES) | {"cpu-sanity"}, set(T.PROFILES))

    def test_deployment_sampler_is_the_registered_one(self):
        reg = yaml.safe_load((GATE / "registration-gatedtrain.yaml").read_text(encoding="utf-8"))["sampler"]
        c = T.COMMON
        doc = T.sampler_doc(c["deploy_num_steps"], c["deploy_noise_level"], c["guidance_scale"], c["resolution"],
                            c["max_sequence_length"], reg["scheduler_config"], reg["sigmas"], c["precision"],
                            c["encoder_precision"])
        self.assertEqual(doc, reg)
        self.assertEqual({k: doc[k] for k in E.SAMPLER_DEFAULT}, E.SAMPLER_DEFAULT)

    def test_cpu_sanity_sampler_is_reproduced(self):
        ra = json.loads((CPU / "run_args.json").read_text(encoding="utf-8"))
        s = ra["settings"]
        for key, steps, noise in (("sampler", "deploy_num_steps", "deploy_noise_level"),
                                  ("training_sampler", "num_steps", "noise_level")):
            rec = ra[key]
            cfg = dict(rec["scheduler_config"])
            cfg["_use_default_values"] = list(reversed(cfg["_use_default_values"]))    # a set's order varies
            doc = T.sampler_doc(s[steps], s[noise], s["guidance_scale"], s["resolution"], s["max_sequence_length"], cfg,
                                rec["sigmas"], s["precision"], s["encoder_precision"])
            self.assertEqual(doc, rec, key)
        self.assertEqual(T.sha256_bytes(T.canonical(ra["sampler"])), ra["sampler_sha256"])
        self.assertEqual(json.loads((CPU / "checkpoint-1" / "sampler.json").read_text(encoding="utf-8")), ra["sampler"])

    def test_shared_code_and_pins(self):
        def block(path):
            text = path.read_text(encoding="utf-8")
            return text[text.index("# BEGIN SHARED SAMPLER"):text.index("# END SHARED SAMPLER")]

        self.assertEqual(block(RUNTIME / "train_flow_grpo.py"), block(RUNTIME / "eval_images.py"))
        self.assertEqual(T.sampler_code_sha256(), A.SAMPLER_CODE_SHA256)
        self.assertEqual(E.sampler_code_sha256(), A.SAMPLER_CODE_SHA256)
        self.assertEqual(T.sha256_file(RUNTIME / "eval_images.py"), R.EVAL_IMAGES_SD35_SHA256)
        for parts in [("sample", 5000, 1, 0), ("eval", 11, "ocr-test-0001"), ("mask", 5000, 3), ("perm", 5000, 2, 0)]:
            self.assertEqual(T.derive_seed(*parts), sdk_derive_seed(*parts))
            self.assertEqual(E.derive_seed(*parts), sdk_derive_seed(*parts))


# ---------------------------------------------------------------------------
# Advantages and the zero-advantage filter (flow_grpo stat_tracking.py, train_sd3.py:783-823)
# ---------------------------------------------------------------------------

class TestAdvantages(unittest.TestCase):
    def test_grpo_advantages(self):
        import numpy as np

        rewards = [0.1, 0.3, 0.5, 0.5, 0.9, 0.9, 0.2, 0.0]
        prompts = ["a"] * 4 + ["b"] * 4
        adv = T.PerPromptStatTracker(True).update(prompts, rewards)
        r = np.array(rewards)
        for k in (slice(0, 4), slice(4, 8)):
            np.testing.assert_allclose(adv[k], (r[k] - r[k].mean()) / (r.std() + 1e-4))
        adv = T.PerPromptStatTracker(False).update(prompts, rewards)
        for k in (slice(0, 4), slice(4, 8)):
            np.testing.assert_allclose(adv[k], (r[k] - r[k].mean()) / (r[k].std() + 1e-4))

    def test_a_flat_group_has_exactly_zero_advantage(self):
        # The filter keeps a sample iff its advantage != 0; a group of G equal rewards must give exact zeros (G = 8
        # and 2: the profiles' group sizes), whatever the reward value.
        rnd = random.Random(4)
        tr = T.PerPromptStatTracker(True)
        for G in (8, 2):
            for _ in range(200):
                v = rnd.random()
                vals = [v] * G + [rnd.random() for _ in range(G)]
                adv = tr.update(["flat"] * G + ["other"] * G, vals)
                self.assertTrue(all(a == 0.0 for a in adv[:G]), (G, v))

    def test_cpu_sanity_run_is_reproduced(self):
        # The recorded cpu-sanity run: seeds and sample ids from the runtime's functions, the recorded rewards give
        # nonzero advantages, so the epoch plan keeps every sample (as recorded).
        ra = json.loads((CPU / "run_args.json").read_text(encoding="utf-8"))
        samples = read_jsonl(CPU / "samples.jsonl")
        for x in samples:
            self.assertEqual(x["seed"], T.derive_seed("sample", ra["seed"], x["epoch"], x["sample_index"]))
            self.assertEqual(x["sample_id"], T.sha256_bytes(T.canonical(
                ["flow-grpo-sd35", ra["seed"], x["attempt"], x["epoch"], x["sample_index"], x["prompt_id"], x["seed"],
                 x["member"]])))
            self.assertEqual(x["target"], T.target_text(x["prompt"]))
            self.assertEqual(x["member"], x["sample_index"] % ra["settings"]["images_per_prompt"])
        adv = T.PerPromptStatTracker(ra["settings"]["global_std"]).update([x["prompt_id"] for x in samples],
                                                                          [x["reward"] for x in samples])
        plan = read_jsonl(CPU / "epoch_plan.jsonl")[0]
        self.assertEqual([x["sample_id"] for x, a in zip(samples, adv) if a != 0], plan["kept"])


# ---------------------------------------------------------------------------
# Cuts, base binding, run files
# ---------------------------------------------------------------------------

class TestCutsAndBinding(Tmp):
    def cut(self, epoch=2, ema=False, transformer=None):
        base = self.base()
        ref = T.base_binding(base, verify=True)
        ra = json.loads((CPU / "run_args.json").read_text(encoding="utf-8"))
        out = self.tmp / "out"
        out.mkdir(exist_ok=True)
        d = T.save_cut(transformer or FakeTransformer(), out, epoch, 8, 128, 0, ref, ra["sampler"], {"r": 32},
                       {"script_sha256": "sha256:" + "1" * 64, "sampler_code_sha256": T.sampler_code_sha256(),
                        "seed": 5000, "profile": "full", "ema": ema})
        return d, ref, base

    def test_cut_is_what_the_adapter_and_the_evaluator_accept(self):
        d, ref, base = self.cut()
        ad = A.FlowGrpoSd35Adapter()
        self.assertIsNone(ad.cut_problem(d, 2))
        doc = json.loads((d / "cut.json").read_text(encoding="utf-8"))
        recorded = json.loads((CPU / "checkpoint-1" / "cut.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(doc), sorted(recorded))                  # the layout of the recorded run's cut
        self.assertEqual(sorted(doc["deployable"]), sorted(R.TRAINER_ROWS["flow_grpo_sd35"]["components"]))
        self.assertEqual(stat.S_IMODE((d / T.LORA_WEIGHTS).stat().st_mode), 0o644)
        self.assertIsNone(safetensors_problem(d / T.LORA_WEIGHTS))
        raw = (d / T.LORA_WEIGHTS).read_bytes()
        header = json.loads(raw[8:8 + struct.unpack("<Q", raw[:8])[0]])
        meta = json.loads(header["__metadata__"]["lora_adapter_metadata"])
        self.assertEqual(meta["target_modules"], sorted(T.LORA["target_modules"]))
        b = json.loads((d / "base_ref.json").read_text(encoding="utf-8"))
        self.assertEqual(b["files"], ad._base_files(base))                # .cache/ left out on both sides
        self.assertEqual(b["files_sha256"], compact_sha256(b["files"]))
        cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
        self.assertEqual((cfg["weights"], cfg["base_model_revision"]), ("raw", ref["revision"]))
        self.assertEqual(E.check_cut(d)["epoch"], 2)
        (d / "trainer_state.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            E.check_cut(d)

    def test_cut_bytes_do_not_depend_on_the_header_order(self):
        a, _, _ = self.cut(transformer=FakeTransformer(1))
        shutil.move(str(a), str(self.tmp / "first"))
        b, _, _ = self.cut(transformer=FakeTransformer(9))
        self.assertEqual((self.tmp / "first" / T.LORA_WEIGHTS).read_bytes(), (b / T.LORA_WEIGHTS).read_bytes())

    def test_ema_cut_says_so(self):
        d, _, _ = self.cut(ema=True)
        self.assertEqual(json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))["weights"], "ema")

    def test_eval_refuses_another_sampler_code(self):
        d, _, _ = self.cut()
        self.patch(E, "sampler_code_sha256", lambda: "sha256:" + "0" * 64)
        with self.assertRaises(SystemExit):
            E.check_cut(d)

    def test_base_binding_refusals(self):
        base = self.base()
        (base / "extra").write_text("x", encoding="utf-8")
        with self.assertRaises(SystemExit):
            T.base_binding(base, verify=False)
        (base / "extra").unlink()
        (base / "model_index.json").write_bytes(b"{} changed")
        self.assertFalse(T.base_binding(base, verify=False)["verified"])
        with self.assertRaises(SystemExit):
            T.base_binding(base, verify=True)
        (base / "model_index.json").unlink()
        os.symlink(self.tmp / "MODELS.json", base / "model_index.json")
        with self.assertRaises(SystemExit):
            T.base_binding(base, verify=False)

    def test_run_files(self):
        out = self.tmp / "out"
        self.assertEqual(T.prepare_out(out), 1)
        rt = T.write_env_freeze(out)
        lines = (out / "env-freeze.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(any(A.REFERENCE_LINE_RE.fullmatch(x) for x in lines))
        self.assertEqual(rt["script_sha256"], T.sha256_file(RUNTIME / "train_flow_grpo.py"))
        self.assertEqual(rt["sampler_code_sha256"], A.SAMPLER_CODE_SHA256)
        recorded = json.loads((CPU / "env-runtime.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(rt), sorted(recorded))
        self.assertEqual(T.prepare_out(out), 2)
        self.assertTrue((out / "preempted-attempt-1" / "env-freeze.txt").is_file())
        d = self.tmp / "p"
        d.mkdir()
        (d / "train.jsonl").write_text('{"prompt_id": "a"}\n{"prompt_id": "a"}\n', encoding="utf-8")
        with self.assertRaises(SystemExit):
            T.read_prompts(d)


# ---------------------------------------------------------------------------
# A whole run's logs, written with the runtime's functions, through the adapter's consumption check
# ---------------------------------------------------------------------------

class TestRunThroughTheLedger(Tmp):
    """samples.jsonl, epoch_plan.jsonl, train_steps.jsonl and the cuts as train_flow_grpo.py main writes them, with
    the runtime's derive_seed, canonical, target_text, PerPromptStatTracker, resolve_settings, prepare_out and
    save_cut; main's bookkeeping (prompt draw :707-710, sample records :791-807, the filter and its padding :809-834,
    step grouping :843-905) is followed statement by statement, with numpy permutations standing in for the
    torch.randperm calls (the adapter re-derives neither). Rewards are chosen so that some prompt groups are flat
    (zero advantage, filtered) and the padding re-includes some of them."""

    def run_logs(self, epochs=3):
        import numpy as np

        out = self.tmp / "out"
        args = T.parse_args(["--profile", "full", "--epochs", str(epochs), "--out", str(out)])
        attempt = T.prepare_out(out)
        s = T.resolve_settings(args)
        self.data = self.tmp / "prompts"                 # the spec's one training shard
        self.data.mkdir()
        shutil.copyfile(FG / "train.jsonl", self.data / "train.jsonl")
        prompts, _ = T.read_prompts(self.data)
        T.write_json(out / "run_args.json", {"profile": "full", "attempt": attempt, "settings": s, "seed": s["seed"]})
        ref = T.base_binding(self.base(), verify=True)
        P, G, E_, nb = s["prompts_per_epoch"], s["images_per_prompt"], s["epochs"], s["num_batches_per_epoch"]
        plans = {}
        for epoch in range(1, E_ + 1):
            rng = np.random.Generator(np.random.PCG64(T.derive_seed("prompts", s["seed"], epoch)))
            plans[epoch] = [prompts[int(c)] for c in rng.choice(len(prompts), size=P, replace=False)]
        tracker = T.PerPromptStatTracker(s["global_std"])
        global_step = consumed_total = filtered_total = 0
        for epoch in range(1, E_ + 1):
            plan = [(row, g) for row in plans[epoch] for g in range(G)]
            sample_ids, prompt_ids, rewards = [], [], []
            first_step = global_step + 1
            for i, (row, g) in enumerate(plan):
                seed = T.derive_seed("sample", s["seed"], epoch, i)
                sid = T.sha256_bytes(T.canonical(["flow-grpo-sd35", s["seed"], attempt, epoch, i, row["prompt_id"], seed,
                                                  g]))
                group = i // G
                # A flat group (all advantages 0, filtered), a group with two nonzero advantages (6 zeros, so the
                # padding re-includes some), the others varied.
                member_acc = {0: 0.25, 1: 0.75}.get(g, 0.5)
                acc = (0.5 if group % 3 == epoch % 3 else member_acc if group % 3 == (epoch + 1) % 3
                       else round((i * 7 % 11) / 10, 2))
                text = T.target_text(row["prompt"])
                sample_ids.append(sid)
                prompt_ids.append(row["prompt_id"])
                rewards.append(acc)
                T.append_jsonl(out / "samples.jsonl", {
                    "epoch": epoch, "sample_index": i, "sample_id": sid, "prompt_id": row["prompt_id"],
                    "prompt": row["prompt"], "target": text, "member": g, "seed": seed, "ocr_text": text,
                    "ocr_error": None, "ocr_acc": acc, "reward": acc, "attempt": attempt, "utc": T.utc_now()})
            adv = tracker.update(prompt_ids, rewards)
            n = len(plan)
            mask = adv != 0
            true_count = int(mask.sum())
            reincluded = []
            if true_count % nb != 0:
                false_idx = np.where(~mask)[0]
                need = nb - (true_count % nb)
                if len(false_idx) >= need:
                    pick = false_idx[np.random.default_rng(epoch).permutation(len(false_idx))[:need]]
                    mask[pick] = True
                    reincluded = sorted(int(x) for x in pick)
            kept = list(np.where(mask)[0])
            n_kept = len(kept)
            self.assertEqual(n_kept % nb, 0)
            n_steps_epoch = 2 * s["num_inner_epochs"] if n_kept else 0
            T.append_jsonl(out / "epoch_plan.jsonl", {
                "epoch": epoch, "kept": [sample_ids[i] for i in kept],
                "filtered": [sample_ids[i] for i in range(n) if not bool(mask[i])],
                "reincluded": [sample_ids[i] for i in reincluded], "filter_reason": "zero_advantage",
                "steps": [first_step, global_step + n_steps_epoch] if n_steps_epoch else [], "utc": T.utc_now()})
            if n_kept:
                m = n_kept // nb
                for inner in range(s["num_inner_epochs"]):
                    order = [int(kept[k]) for k in np.random.default_rng(100 + epoch).permutation(n_kept)]
                    virtual = [order[v * m:(v + 1) * m] for v in range(nb)]
                    consumed = []
                    for v, vidx in enumerate(virtual):
                        for c0 in range(0, m, s["train_batch"]):
                            consumed.extend(sample_ids[i] for i in vidx[c0:c0 + s["train_batch"]])
                        if (v + 1) % s["gradient_accumulation_steps"] == 0:
                            global_step += 1
                            T.append_jsonl(out / "train_steps.jsonl", {"step": global_step, "epoch": epoch,
                                                                       "inner": inner, "sample_ids": consumed,
                                                                       "utc": T.utc_now()})
                            consumed = []
                consumed_total += n_kept
            filtered_total += n - n_kept
            if epoch % s["save_every"] == 0 or epoch == E_:
                T.save_cut(FakeTransformer(epoch), out, epoch, global_step, consumed_total, filtered_total, ref, {}, {},
                           {"seed": s["seed"], "profile": "full", "ema": False})
        return out, filtered_total

    def test_logs_pass_the_adapters_consumption_check(self):
        out, filtered = self.run_logs()
        self.assertGreater(filtered, 0)
        ad = A.FlowGrpoSd35Adapter()
        cuts, incomplete = ad.find_cuts(out)
        self.assertEqual(([c.name for _, c in cuts], incomplete), (["checkpoint-2", "checkpoint-3"], []))
        self.assertEqual(ad.consumption_problems(out, self.data, cuts), [])
        plan, bad = ad._ledger_plan(out, self.data, cuts)
        self.assertEqual(bad, [])
        self.assertEqual(sum(1 for _, step in plan if step is None), filtered)
        reinc = [p["reincluded"] for p in read_jsonl(out / "epoch_plan.jsonl")]
        self.assertTrue(any(reinc), "the rewards should make the padding re-include a zero-advantage sample")


if __name__ == "__main__":
    unittest.main()
