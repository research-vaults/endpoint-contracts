#!/usr/bin/env python3
"""Refresh the compact robustness ledger from the canonical measurement-contracts-0.6 authorities."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "metric_robustness.json"


def main() -> None:
    prior = json.loads(OUT.read_text())
    canonical = json.loads((RESULTS / "canonical_disclosure_shuffle.json").read_text())
    pilot = json.loads((RESULTS / "pilot_controls.json").read_text())
    report = {
        "protocol": "measurement-contracts-0.6 deterministic length-nearest derangement v1",
        "n": canonical["n"],
        "shuffle_mean": canonical["mean"],
        "ci": canonical["ci95"],
        "frac": canonical["frac_pos"],
        "length_match": canonical["length_match"],
        "loo_spearman_min": prior["loo_spearman_min"],
        "loo_spearman_mean": prior["loo_spearman_mean"],
        "full_means": {
            stage: values["R_availability"]["mean"]
            for stage, values in pilot["stage_summary"].items()
        },
        "topic_only_vs_S3_delta": (
            pilot["stage_summary"]["S3_methods_mid"]["R_availability"]["mean"]
            - pilot["stage_summary"]["S0_topic"]["R_availability"]["mean"]
        ),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
