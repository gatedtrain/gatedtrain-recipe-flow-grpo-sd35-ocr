#!/usr/bin/env python
"""Check that rapidfuzz's Levenshtein distance returns the values python-Levenshtein returned (RESULT.md).

The published 2026-11-14 run computed the OCR reward (`ocr_accuracy`) and eval_images.py's `ocr_a_acc` with
`Levenshtein.distance` from python-Levenshtein 0.27.5 (GPL-2.0-or-later), a thin wrapper of rapidfuzz; the scripts now
import `rapidfuzz.distance.Levenshtein.distance` (rapidfuzz 3.14.6, MIT). This check compares the two functions on
20,006 string pairs (six fixed edge cases, then 20,000 random pairs of 0-40 characters over lower-case letters,
digits, spaces, punctuation and non-ASCII characters, `random.Random(0)`), and `ocr_accuracy` (flow_grpo ocr.py:46-60:
spaces removed, lower case, 1 when the target is a substring, else 1 - min(distance, len(target)) / len(target)) on
every pair whose target is not empty after removing spaces. Expected output: 0 mismatches in both.

It needs both packages, so it runs in an environment that still has python-Levenshtein (the published run's venv,
built from that run's lock), not in this runtime's current one:
  python check_edit_distance.py
"""

from __future__ import annotations

import random
import sys


def ocr_accuracy(recognized: str, target: str, distance) -> float:
    recognized = recognized.replace(" ", "").lower()
    target = target.replace(" ", "").lower()
    dist = 0 if target in recognized else distance(recognized, target)
    dist = min(dist, len(target))
    return 1 - dist / len(target)


def main() -> int:
    import Levenshtein
    import rapidfuzz
    from Levenshtein import distance as old
    from rapidfuzz.distance.Levenshtein import distance as new

    print("python-Levenshtein", Levenshtein.__version__, "rapidfuzz", rapidfuzz.__version__)
    rng = random.Random(0)
    alpha = "abcdefghijklmnopqrstuvwxyz0123456789 .,'!?-éß中"
    pairs = [("", ""), ("", "abc"), ("hello", "hallo"), ("kitten", "sitting"), ("openai", "open ai"), ("ß", "ss")]
    for _ in range(20000):
        a = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 40)))
        b = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 40)))
        pairs.append((a, b))
    bad = sum(old(a, b) != new(a, b) for a, b in pairs)
    print("pairs", len(pairs), "distance mismatches", bad)
    ocr = [(a, b) for a, b in pairs if b.replace(" ", "")]
    bad_acc = sum(ocr_accuracy(a, b, old) != ocr_accuracy(a, b, new) for a, b in ocr)
    print("ocr_accuracy pairs", len(ocr), "mismatches", bad_acc)
    return 1 if bad or bad_acc else 0


if __name__ == "__main__":
    sys.exit(main())
