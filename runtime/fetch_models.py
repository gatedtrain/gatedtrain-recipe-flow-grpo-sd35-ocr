#!/usr/bin/env python
"""Fetch the ungated OCR model files of area "SD3.5" and record every model file's SHA-256 (host step, no GPU).

- SD3.5 Medium is gated: it is NOT downloaded here. Accept its license on the Hub and download it beforehand to
  `models/stable-diffusion-3.5-medium/` at revision b940f670f0eda2d07fbb75229e779da1ad11eb80. This script only reads
  it: it hashes every file outside `.cache/`, checks that each LFS file's SHA-256 equals the hash in huggingface_hub's
  download metadata (`.cache/huggingface/download/<file>.metadata`, line 1 the commit, line 2 the etag), and records
  repo, revision and hashes in `MODELS.json`. Nothing under `models/` is written, moved or deleted.
- OCR engine A (training reward, as flow_grpo/ocr.py): PaddleOCR PP-OCR inference models in Paddle 3 format from
  the PaddlePaddle organization on the Hugging Face Hub (ungated, Apache-2.0): `en_PP-OCRv4_mobile_rec` (the English
  recognizer flow_grpo's paddleocr 2.9.1 downloads as en_PP-OCRv4_rec) and `PP-OCRv4_mobile_det` (the detector that
  PaddleOCR 3.7.0 pairs with it for lang="en", paddleocr/_pipelines/ocr.py:394-397; flow_grpo's 2.9.1 default
  detector en_PP-OCRv3_det is only on paddleocr.bj.bcebos.com, outside the allowed download sources). Downloaded with
  huggingface_hub (`token=False`, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, fresh `HF_HOME` outside the area, ../area5-ref/hf-home) to
  `ocr-models/paddleocr/<name>/`; the download metadata moves to `ocr-models/hf-meta/<name>/` (plain directories).
- OCR engine B (deciding metric): docTR db_resnet50 (detection) and crnn_vgg16_bn (recognition), the files of
  python-doctr 1.1.0 `default_cfgs`, from the mindee/doctr GitHub release assets (v0.7.0, v0.12.0), each checked
  against its full pinned SHA-256 before it is loaded. The pickled `.pt`
  state dicts stay outside the area directory (`--download-dir`, default ../area5-ref/downloads) and are converted
  once with `torch.load(weights_only=True)` to safetensors in `ocr-models/doctr/`, so no runtime unpickles them.

No credential is used or read; the script refuses to run when HF_TOKEN is set (it never contacts the Hub anyway).

Usage (host):
  .venv/bin/python fetch_models.py             # fetch what is missing, write MODELS.json
  .venv/bin/python fetch_models.py --verify    # re-hash every recorded file against MODELS.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"                 # read only: holds only the separately downloaded SD3.5 directory
OCR = HERE / "ocr-models"                # written by this script
MODELS_JSON = HERE / "MODELS.json"
SD35 = "stable-diffusion-3.5-medium"
SD35_REPO = "stabilityai/stable-diffusion-3.5-medium"
SD35_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"

# name -> (repo, revision); revisions resolved 2026-11-13 from the Hub API (`sha` of `main`, `gated: false`), pinned
# as 40-hex commits.
PADDLE = {
    "PP-OCRv4_mobile_det": ("PaddlePaddle/PP-OCRv4_mobile_det", "3cc09f3a5b424e8e010abc7a4271aea12999c2f7"),
    "en_PP-OCRv4_mobile_rec": ("PaddlePaddle/en_PP-OCRv4_mobile_rec", "f97b62fdc0eb71c689393a19a4b21baa1795c9ab"),
}
PADDLE_FILES = ["config.json", "inference.json", "inference.pdiparams", "inference.yml", "README.md"]
# arch -> (release asset URL, full SHA-256 of the asset as downloaded 2026-11-13).
DOCTR = {
    "db_resnet50": ("https://github.com/mindee/doctr/releases/download/v0.7.0/db_resnet50-79bd7d70.pt",
                    "79bd7d702506703b89cf11afa42d12aebf5cf25c3618e6ffd5f85772240ca483"),
    "crnn_vgg16_bn": ("https://github.com/mindee/doctr/releases/download/v0.12.0/crnn_vgg16_bn-0417f351.pt",
                      "0417f3511f5c75369bdcbe06539853b9c2d170aa3baceb2acc454c775ddac087"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "gatedtrain-recipe-flow-grpo-sd35-ocr-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dest)


def files_record(root: Path, skip_prefix: str | None = None) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if skip_prefix and (rel == skip_prefix or rel.startswith(skip_prefix + "/")):
            continue
        if p.is_symlink():
            raise SystemExit(f"{p} is a symbolic link; model directories must be plain (MM-20)")
        if p.is_file():
            out[rel] = {"sha256": "sha256:" + sha256_file(p), "bytes": p.stat().st_size}
    return out


def record_sd35() -> dict:
    root = MODELS / SD35
    files = files_record(root, skip_prefix=".cache")
    meta_dir = root / ".cache" / "huggingface" / "download"
    duplicates, mismatches = {}, []
    for rel, rec in files.items():
        meta = meta_dir / (rel + ".metadata")
        if not meta.exists():
            mismatches.append(f"{rel}: no download metadata")
            continue
        lines = meta.read_text(encoding="utf-8").splitlines()
        if lines[0] != SD35_REVISION:
            mismatches.append(f"{rel}: metadata commit {lines[0]}")
        etag = lines[1]
        if len(etag) == 64 and "sha256:" + etag != rec["sha256"]:      # LFS files: etag is the SHA-256
            mismatches.append(f"{rel}: SHA-256 differs from the LFS etag")
        duplicates.setdefault(rec["sha256"], []).append(rel)
    if mismatches:
        raise SystemExit("SD3.5 directory check failed: " + "; ".join(mismatches))
    dup = {h: sorted(v) for h, v in duplicates.items() if len(v) > 1 and files[v[0]]["bytes"] > 1 << 20}
    return {"repo": SD35_REPO, "revision": SD35_REVISION, "gated": True,
            "downloaded_by": "the operator, with their own Hub login (not by this script)",
            "license": "Stability AI Community License (LICENSE.md in the directory)",
            "files": files, "bytes": sum(r["bytes"] for r in files.values()),
            "excluded_from_record": [".cache/ (huggingface_hub download metadata)"],
            "byte_identical_duplicates": dup}


def fetch_paddle() -> dict:
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HOME"] = str(HERE.parent / "area5-ref" / "hf-home")
    from huggingface_hub import HfApi, snapshot_download

    out = {}
    for name, (repo, revision) in PADDLE.items():
        info = HfApi().model_info(repo, revision=revision, token=False)
        if info.sha != revision:
            raise SystemExit(f"{repo}: the Hub resolved revision {revision} to {info.sha}")
        if info.gated:
            raise SystemExit(f"{repo} is gated; this area downloads ungated repositories only")
        dest = OCR / "paddleocr" / name
        if not dest.exists():
            snapshot_download(repo, revision=info.sha, local_dir=str(dest), allow_patterns=PADDLE_FILES, token=False)
            meta = dest / ".cache"
            if meta.exists():
                (OCR / "hf-meta").mkdir(parents=True, exist_ok=True)
                os.replace(meta, OCR / "hf-meta" / name)
        out[name] = {"repo": repo, "revision": info.sha, "license": "apache-2.0", "files": files_record(dest)}
    return out


def fetch_doctr(download_dir: Path) -> dict:
    root = OCR / "doctr"
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    for arch, (url, want) in DOCTR.items():
        fname = url.split("/")[-1]
        pt = download_dir / fname
        download(url, pt)
        digest = sha256_file(pt)
        if digest != want:                     # stricter than docTR's own 8-hex prefix rule (utils/data.py)
            raise SystemExit(f"{pt}: SHA-256 {digest} differs from the pinned {want}")
        sources[arch] = {"url": url, "pt_sha256": "sha256:" + digest}
        out = root / f"{arch}.safetensors"
        if not out.exists():
            import torch
            from safetensors.torch import save_file

            state = torch.load(pt, map_location="cpu", weights_only=True)
            save_file({k: v.contiguous() for k, v in sorted(state.items())}, str(out),
                      metadata={"source_url": url, "source_sha256": digest})
    return {"repo": "mindee/doctr GitHub release assets (python-doctr 1.1.0 default_cfgs)", "revision": None, "license": "Apache-2.0",
            "sources": sources, "files": files_record(root)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verify", action="store_true")
    p.add_argument("--download-dir", type=Path, default=HERE.parent / "area5-ref" / "downloads")
    args = p.parse_args(argv)
    if os.environ.get("HF_TOKEN") is not None:
        raise SystemExit("HF_TOKEN is set; run with `env -u HF_TOKEN` (this script needs no credential)")
    if args.verify:
        doc = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
        bad = []
        for name, rec in doc["models"].items():
            for rel, want in rec["files"].items():
                base = MODELS / name if name == SD35 else OCR / name
                if "sha256:" + sha256_file(base / rel) != want["sha256"]:
                    bad.append(f"{name}/{rel}")
        print(json.dumps({"verified": not bad, "mismatches": bad}, indent=2))
        return 1 if bad else 0
    args.download_dir.mkdir(parents=True, exist_ok=True)
    models = {SD35: record_sd35(), **{f"paddleocr/{k}": v for k, v in fetch_paddle().items()},
              "doctr": fetch_doctr(args.download_dir)}
    doc = {"schema": "ptc_mm.models", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
           "models": models}
    tmp = MODELS_JSON.with_name(MODELS_JSON.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, MODELS_JSON)
    print(json.dumps({k: {"files": len(v["files"]), "bytes": sum(f["bytes"] for f in v["files"].values())}
                      for k, v in models.items()}, indent=2))
    print("byte-identical duplicates in the SD3.5 directory:",
          json.dumps(models[SD35]["byte_identical_duplicates"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
