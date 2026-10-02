#!/usr/bin/env python3
"""Bound disclosure-frame attrition under the frozen observed-pair contrast.

This is a worst-case completion, not an imputation. It holds the 86 observed
true-minus-length-nearest-deranged item contrasts fixed and lets each of the 11
items without full ar5iv HTML take any value in the metric-implied [-1, 1]
range. It does not claim what a newly matched 97-item derangement would yield.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "evidence_interventions.json"
OUT = RESULTS / "missing_item_tipping_bound.json"

SAMPLING_FRAME_N = 97
MISSING_HTML_N = 11
DELTA_MIN = -1.0
DELTA_MAX = 1.0


def main() -> None:
    interventions = json.loads(SOURCE.read_text(encoding="utf-8"))
    deltas = [float(row["delta_shuffle"]) for row in interventions["rows"]]
    observed_n = len(deltas)
    if observed_n != interventions["n"]:
        raise ValueError("Intervention row count does not match declared n")
    if observed_n + MISSING_HTML_N != SAMPLING_FRAME_N:
        raise ValueError("Observed and missing counts do not reconstruct the frame")
    if not all(DELTA_MIN <= delta <= DELTA_MAX for delta in deltas):
        raise ValueError("Observed item contrast lies outside the declared bounds")

    observed_sum = sum(deltas)
    observed_mean = observed_sum / observed_n
    lower = (observed_sum + MISSING_HTML_N * DELTA_MIN) / SAMPLING_FRAME_N
    upper = (observed_sum + MISSING_HTML_N * DELTA_MAX) / SAMPLING_FRAME_N
    required_missing_mean_for_zero = -observed_sum / MISSING_HTML_N
    minimum_all_worst_case_missing_n_for_nonpositive = math.ceil(observed_sum)

    report = {
        "protocol": "measurement-contracts-0.9 fixed-observed-pair worst-case attrition completion",
        "estimand": (
            "mean true-minus-length-nearest-deranged R_avail contrast over the "
            "97-item convenience sampling frame, holding the 86 observed pair "
            "contrasts fixed"
        ),
        "population": {
            "sampling_frame_n": SAMPLING_FRAME_N,
            "observed_full_html_n": observed_n,
            "missing_full_html_n": MISSING_HTML_N,
            "missing_reason": "full ar5iv HTML unavailable",
        },
        "metric_contract": {
            "R_avail_range": [0.0, 1.0],
            "item_contrast": "R_avail(true S3) - R_avail(frozen length-nearest deranged S3)",
            "item_contrast_range": [DELTA_MIN, DELTA_MAX],
        },
        "observed": {
            "sum_item_contrasts": observed_sum,
            "mean_item_contrast": observed_mean,
            "fraction_positive": sum(delta > 0 for delta in deltas) / observed_n,
        },
        "worst_case_completion": {
            "full_frame_mean_identification_region": [lower, upper],
            "lower_bound_assumption": "all 11 missing item contrasts equal -1",
            "upper_bound_assumption": "all 11 missing item contrasts equal +1",
            "required_missing_mean_for_full_frame_zero": required_missing_mean_for_zero,
            "required_missing_mean_is_feasible": (
                DELTA_MIN <= required_missing_mean_for_zero <= DELTA_MAX
            ),
            "minimum_missing_items_at_delta_minus_one_for_nonpositive_mean": (
                minimum_all_worst_case_missing_n_for_nonpositive
            ),
            "actual_missing_items": MISSING_HTML_N,
            "sign_identified_positive_under_fixed_pair_completion": lower > 0,
        },
        "interpretation": (
            "Under a fixed-observed-pair completion, the 11 unavailable items "
            "cannot reverse the positive mean even if every missing contrast is "
            "set to its mathematical minimum of -1."
        ),
        "limitations": [
            (
                "This bound holds the 86 observed donor assignments and item "
                "contrasts fixed; it does not rerun length-nearest matching on "
                "a hypothetical 97-item complete corpus."
            ),
            (
                "It bounds attrition inside the author-constructed convenience "
                "frame only; it does not transport to official ProjectionBench "
                "or another target population."
            ),
            "The identification region is a deterministic bound, not a confidence interval.",
        ],
        "source": "results/evidence_interventions.json",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
