#!/usr/bin/env python3
"""Score completed forced-choice human labels against continuous R key.

Usage:
  python3 score_forced_choice_labels.py annotatorA.csv [annotatorB.csv]

Accepts CSV from pairs_labels_template.csv or the HTML annotator export.
Writes results/human_criterion_forced_choice.json and prints a paper-ready
LaTeX snippet for §5 when labels are complete.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parents[1]
KEY_PATH = ROOT / "pairs_analysis_key.json"
OUT_JSON = PROJ / "results" / "human_criterion_forced_choice.json"
SEED = 20260731

key = {p["pair_id"]: p for p in json.loads(KEY_PATH.read_text())["pairs"]}


def _norm_choice(c: str) -> str:
    c = (c or "").strip().lower().strip('"')
    if c in ("l", "left"):
        return "left"
    if c in ("r", "right"):
        return "right"
    return ""


def load(path: str | Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # HTML export may quote fields
            pid = (r.get("pair_id") or "").strip().strip('"')
            c = _norm_choice(r.get("choice_left_or_right") or "")
            if pid and c:
                rows.append(
                    {
                        "pair_id": pid,
                        "choice": c,
                        "annotator": (r.get("annotator") or "").strip().strip('"'),
                        "confidence": (r.get("confidence_1_to_3") or "").strip().strip('"'),
                    }
                )
    return rows


def validate(rows: list[dict], path: str | Path) -> None:
    pair_ids = [r["pair_id"] for r in rows]
    duplicate_ids = sorted(pid for pid, n in Counter(pair_ids).items() if n > 1)
    unknown_ids = sorted(set(pair_ids) - set(key))
    missing_ids = sorted(set(key) - set(pair_ids))
    annotators = sorted({r["annotator"] for r in rows})
    if duplicate_ids or unknown_ids or missing_ids or len(rows) != len(key):
        raise ValueError(
            f"invalid completed labels in {path}: rows={len(rows)} "
            f"duplicates={duplicate_ids} unknown={unknown_ids} missing={missing_ids}"
        )
    if len(annotators) != 1 or not annotators[0]:
        raise ValueError(f"expected one non-empty annotator ID in {path}; got {annotators}")


def file_record(path: str | Path) -> dict:
    p = Path(path)
    return {
        "released_path": f"data/human_criterion_instrument/received/{p.name}",
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }


def acc(rows: list[dict]) -> float | None:
    if not rows:
        return None
    ok = sum(1 for r in rows if r["choice"] == key[r["pair_id"]]["higher_R_side"])
    return ok / len(rows)


def boot_ci(rows: list[dict], n_boot: int = 5000) -> list[float]:
    rng = random.Random(SEED)
    boots = []
    for _ in range(n_boot):
        samp = [rows[rng.randrange(len(rows))] for __ in range(len(rows))]
        boots.append(acc(samp))
    boots.sort()
    return [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]]


def by_stratum(rows: list[dict]) -> dict:
    out: dict[str, list] = {}
    for r in rows:
        st = key[r["pair_id"]].get("stratum", "?")
        out.setdefault(st, []).append(r)
    return {k: {"n": len(v), "acc": acc(v)} for k, v in out.items()}


def binomial_upper_tail(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main() -> None:
    paths = [a for a in sys.argv[1:] if a not in ("-h", "--help")]
    if not paths or any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        print("\nExample:")
        print("  python3 score_forced_choice_labels.py annotatorA.csv annotatorB.csv")
        print(f"Key: {KEY_PATH}")
        print(f"Output JSON: {OUT_JSON}")
        sys.exit(0 if any(a in ("-h", "--help") for a in sys.argv[1:]) else 1)

    A = load(paths[0])
    if not A:
        print(f"No labeled rows in {paths[0]}")
        sys.exit(2)
    validate(A, paths[0])

    a_acc = acc(A)
    a_ci = boot_ci(A)
    print(f"Annotator A: n={len(A)} acc_vs_R={a_acc:.3f} CI95=[{a_ci[0]:.3f},{a_ci[1]:.3f}]")
    print("By stratum:", by_stratum(A))

    result: dict = {
        "protocol": "human forced-choice reconstructibility; frozen seed 20260731",
        "seed": SEED,
        "annotator_A": {
            **file_record(paths[0]),
            "n": len(A),
            "acc_vs_higher_R": a_acc,
            "ci95": a_ci,
            "binomial_p_upper_vs_chance": binomial_upper_tail(round(a_acc * len(A)), len(A)),
            "by_stratum": by_stratum(A),
        },
        "note": "Human criterion result for main §5; not model proxy.",
    }

    if len(paths) > 1:
        B = load(paths[1])
        if not B:
            print(f"No labeled rows in {paths[1]}")
            sys.exit(2)
        validate(B, paths[1])
        if A[0]["annotator"] == B[0]["annotator"]:
            raise ValueError("annotator IDs must be distinct")
        b_acc = acc(B)
        b_ci = boot_ci(B)
        print(f"Annotator B: n={len(B)} acc_vs_R={b_acc:.3f} CI95=[{b_ci[0]:.3f},{b_ci[1]:.3f}]")
        ba = {r["pair_id"]: r["choice"] for r in A}
        bb = {r["pair_id"]: r["choice"] for r in B}
        common = set(ba) & set(bb)
        agree = sum(1 for pid in common if ba[pid] == bb[pid])
        n = len(common)
        po = agree / n if n else float("nan")
        ca = Counter(ba[pid] for pid in common)
        cb = Counter(bb[pid] for pid in common)
        pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb)) if n else float("nan")
        kappa = (po - pe) / (1 - pe) if n and abs(1 - pe) > 1e-9 else float("nan")
        print(f"Overlap n={n} exact_agree={po:.3f} Cohen_kappa={kappa:.3f}")
        result["annotator_B"] = {
            **file_record(paths[1]),
            "n": len(B),
            "acc_vs_R": b_acc,
            "ci95": b_ci,
            "binomial_p_upper_vs_chance": binomial_upper_tail(round(b_acc * len(B)), len(B)),
            "by_stratum": by_stratum(B),
        }
        result["inter_annotator"] = {
            "n_overlap": n,
            "exact_agree": po,
            "cohen_kappa": kappa,
        }
        # pooled: majority or A primary
        result["primary_report"] = {
            "n_pairs": len(A),
            "acc": a_acc,
            "ci95": a_ci,
            "kappa": kappa,
            "n_overlap": n,
        }
    else:
        result["primary_report"] = {
            "n_pairs": len(A),
            "acc": a_acc,
            "ci95": a_ci,
            "kappa": None,
            "n_overlap": None,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {OUT_JSON}")

    pr = result["primary_report"]
    if pr.get("kappa") is not None:
        kappa_s = (
            "Cohen $\\kappa{=}"
            + f"{pr['kappa']:.2f}$ on $n{{=}}{pr['n_overlap']}$ overlap"
        )
    else:
        kappa_s = r"single annotator (no $\kappa$)"
    acc_pct = 100.0 * float(pr["acc"])
    ci0, ci1 = pr["ci95"]
    snippet = (
        "\\paragraph{Human criterion (forced-choice).}\n"
        "On the preferred $60$-pair instrument, annotators choose which of two "
        "endpoints is more reconstructible from its S3 input. "
        f"Annotator A agrees with the continuous higher-$R$ side on ${acc_pct:.1f}\\%$ "
        f"of pairs (bootstrap $95\\%$ CI $[{ci0:.3f},{ci1:.3f}]$); "
        f"annotator B agrees on ${100.0 * result['annotator_B']['acc_vs_R']:.1f}\\%$ "
        f"(CI $[{result['annotator_B']['ci95'][0]:.3f},{result['annotator_B']['ci95'][1]:.3f}]$; "
        f"$n{{=}}{pr['n_pairs']}$ each; {kappa_s}; "
        "ledger \\texttt{results/human\\_criterion\\_forced\\_choice.json}).\n"
    )
    snip_path = PROJ / "results" / "human_criterion_section_snippet.tex"
    snip_path.write_text(snippet)
    print("--- paper-ready snippet ---")
    print(snippet)
    print(f"Wrote {snip_path}")


if __name__ == "__main__":
    main()
