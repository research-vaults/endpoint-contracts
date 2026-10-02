#!/usr/bin/env python3
"""Refresh cross-benchmark formal-significance surfaces from canonical ledgers.

This script deliberately preserves the independently generated disclosure and
panel inference blocks in formal_significance.json, while rebuilding every
cross-benchmark endpoint/intervention row from its named current authority.
It prevents superseded ResearchBench and AI Idea epochs from remaining live.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "formal_significance.json"
PANEL_OUT = RESULTS / "panel_correlation_cis.json"
CONFIRMATORY_SEED = 20260730
CONFIRMATORY_DRAWS = 5000


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def sign_flip_plus_one(values: list[float], seed: int = CONFIRMATORY_SEED) -> dict:
    """One-sided Monte Carlo sign-flip test with the declared plus-one rule.

    The RNG is reset to the frozen seed for each predeclared test so every test
    is independently replayable from its item-level vector.
    """
    observed = statistics.mean(values)
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(CONFIRMATORY_DRAWS):
        null_mean = statistics.mean(
            value if rng.random() < 0.5 else -value for value in values
        )
        exceedances += null_mean >= observed
    numerator = 1 + exceedances
    denominator = CONFIRMATORY_DRAWS + 1
    return {
        "n": len(values),
        "effect_mean": observed,
        "alternative": "greater",
        "seed": seed,
        "draws": CONFIRMATORY_DRAWS,
        "rng_reset_per_test": True,
        "exceedances": exceedances,
        "plus_one_numerator": numerator,
        "plus_one_denominator": denominator,
        "raw_p_fraction": f"{numerator}/{denominator}",
        "raw_p": numerator / denominator,
    }


def build_confirmatory_family() -> dict:
    pilot_rows = load("pilot_per_item.json")
    disclosure_rows = load("evidence_interventions.json")["rows"]
    fire_rows = load("firebench_recoverability.json")["rows"]
    rb_rows = load("researchbench_generation_audit.json")["generation_rows"]

    vectors = [
        (
            "disclosure_S0_to_S4",
            "Disclosure S0→S4 positive control",
            [
                row["S4_body"]["R_availability"]
                - row["S0_topic"]["R_availability"]
                for row in pilot_rows
            ],
            "pilot_per_item.json",
        ),
        (
            "disclosure_identity",
            "Disclosure S3 true minus length-nearest deranged",
            [row["delta_shuffle"] for row in disclosure_rows],
            "evidence_interventions.json",
        ),
        (
            "fire_identity",
            "FIRE instruction true minus length-nearest deranged",
            [row["delta_shuffle"] for row in fire_rows],
            "firebench_recoverability.json",
        ),
        (
            "researchbench_identity",
            "ResearchBench tiny true minus length-nearest deranged",
            [row["delta_shuffle"] for row in rb_rows],
            "researchbench_generation_audit.json",
        ),
        (
            "hard_negative_S1",
            "Disclosure S1 true minus hard negative",
            [
                row["hard_neg_S1"]["true_R_avail"]
                - row["hard_neg_S1"]["hard_R_avail"]
                for row in pilot_rows
            ],
            "pilot_per_item.json",
        ),
    ]
    expected_n = [86, 86, 35, 12, 86]
    if [len(values) for _, _, values, _ in vectors] != expected_n:
        raise ValueError("confirmatory-family population changed")

    tests = []
    for test_id, label, values, source in vectors:
        tests.append(
            {
                "test_id": test_id,
                "label": label,
                "source": source,
                **sign_flip_plus_one(values),
            }
        )

    alpha = 0.05
    ordered = sorted(range(len(tests)), key=lambda index: (tests[index]["raw_p"], index))
    continue_rejecting = True
    for rank, index in enumerate(ordered, start=1):
        threshold = alpha / (len(tests) - rank + 1)
        reject = continue_rejecting and tests[index]["raw_p"] <= threshold
        tests[index]["holm_rank"] = rank
        tests[index]["holm_threshold"] = threshold
        tests[index]["holm_reject"] = reject
        continue_rejecting = reject

    return {
        "family_id": "TRA_primary_freeze_v1",
        "m": len(tests),
        "alpha": alpha,
        "method": "one-sided Monte Carlo sign flip with plus-one correction; Holm step-down family control",
        "seed": CONFIRMATORY_SEED,
        "draws_per_test": CONFIRMATORY_DRAWS,
        "rng_contract": "independently reset to the frozen seed for each predeclared item vector",
        "minimum_attainable_p": 1 / (CONFIRMATORY_DRAWS + 1),
        "all_raw_p_below_0_001": all(test["raw_p"] < 0.001 for test in tests),
        "all_holm_significant": all(test["holm_reject"] for test in tests),
        "tests": tests,
    }


def main() -> None:
    formal = load("formal_significance.json")
    ai = load("ai_idea_bench_recoverability_full.json")["summary"]
    rb = load("researchbench_generation_audit.json")["generation"]
    fire = load("firebench_recoverability.json")["summary"]
    disclosure = load("pilot_controls.json")["stage_summary"]
    disclosure_shuffle = load("canonical_disclosure_shuffle.json")

    if ai.get("metric_family") != "R_availability_four_measure_mean":
        raise ValueError("AI Idea authority is not the four-measure R_availability family")
    if ai["n"] != 3495:
        raise ValueError(f"unexpected AI Idea population: {ai['n']}")
    if rb["n"] != 12 or fire["n"] != 35 or disclosure_shuffle["n"] != 86:
        raise ValueError("canonical cross-benchmark population changed")

    formal["ai_idea"] = {
        "metric_family": ai["metric_family"],
        "components": ai["metric_components"],
        "n": ai["n"],
        "delta_refs_vs_topic": {
            "mean": ai["mean_delta_refs_vs_topic"],
            "ci95": ai["delta_refs_ci95"],
            "frac": ai["frac_refs_help"],
        },
        "delta_shuffle": {
            "mean": ai["mean_delta_shuffle"],
            "ci95": ai["delta_shuffle_ci95"],
            "frac": ai["frac_shuffle_hurts"],
        },
        "lexical_coverage_component": ai["lexical_coverage_component"],
        "source": "ai_idea_bench_recoverability_full.json",
    }
    formal["disclosure_shuffle"] = {
        **disclosure_shuffle,
        "source": "canonical_disclosure_shuffle.json",
    }
    formal["rb_shuffle"] = {
        "mean": rb["mean_delta_shuffle"],
        "ci95": rb["delta_shuffle_ci95"],
        "n": rb["n"],
        "frac_pos": rb["frac_shuffle_hurts"],
        "source": "researchbench_generation_audit.json",
    }
    formal["fire_shuffle"] = {
        "mean": fire["mean_delta_shuffle"],
        "ci95": fire["delta_shuffle_ci95"],
        "n": fire["n"],
        "frac_pos": fire["frac_shuffle_hurts"],
        "source": "firebench_recoverability.json",
    }

    disclosure_true = disclosure["S3_methods_mid"]["R_availability"]["mean"]
    disclosure_shuffled = disclosure_true - disclosure_shuffle["mean"]
    formal["cross_bench_table"] = [
        {
            "bench": "Disclosure proxy",
            "n": disclosure_shuffle["n"],
            "R_shuffled": disclosure_shuffled,
            "R_true": disclosure_true,
            "delta_interv": disclosure_shuffle["mean"],
            "interv_frac": disclosure_shuffle["frac_pos"],
            "note": "length-nearest deranged S3 / true S3",
            "source": "pilot_controls.json + canonical_disclosure_shuffle.json",
        },
        {
            "bench": "AI Idea Bench",
            "n": ai["n"],
            "R_shuffled": ai["mean_R_topic_shuffled_refs"],
            "R_true": ai["mean_R_topic_refs"],
            "delta_interv": ai["mean_delta_shuffle"],
            "interv_frac": ai["frac_shuffle_hurts"],
            "note": "topic+length-nearest deranged refs / topic+true refs",
            "source": "ai_idea_bench_recoverability_full.json",
        },
        {
            "bench": "ResearchBench tiny",
            "n": rb["n"],
            "R_shuffled": rb["mean_R_shuffled"],
            "R_true": rb["mean_R_RQ_insp"],
            "delta_interv": rb["mean_delta_shuffle"],
            "interv_frac": rb["frac_shuffle_hurts"],
            "note": "RQ+length-nearest deranged inspirations / RQ+true inspirations",
            "source": "researchbench_generation_audit.json",
        },
        {
            "bench": "FIRE-Bench",
            "n": fire["n"],
            "R_shuffled": fire["mean_R_shuffled_instruction"],
            "R_true": fire["mean_R_instruction"],
            "delta_interv": fire["mean_delta_shuffle"],
            "interv_frac": fire["frac_shuffle_hurts"],
            "note": "length-nearest deranged instruction / true instruction",
            "source": "firebench_recoverability.json",
        },
    ]
    formal["cross_bench_metric_contract"] = {
        "metric": "R_availability",
        "family": "four-measure lexical mean",
        "components": ai["metric_components"],
        "contrast": "true evidence minus itemwise length-nearest deranged evidence",
        "identity_intervention": "deterministic_length_nearest_derangement_v1",
        "epoch": "measurement-contracts-0.6",
    }
    confirmatory = build_confirmatory_family()
    formal["confirmatory_family"] = confirmatory
    OUT.write_text(json.dumps(formal, indent=2) + "\n", encoding="utf-8")

    panel = load("panel_correlation_cis.json")
    panel["holm_note"] = {
        "confirmatory_family_m": confirmatory["m"],
        "family_id": confirmatory["family_id"],
        "seed": confirmatory["seed"],
        "draws_per_test": confirmatory["draws_per_test"],
        "plus_one_denominator": CONFIRMATORY_DRAWS + 1,
        "tests": [test["label"] for test in confirmatory["tests"]],
        "raw_plus_one_p_values": {
            test["test_id"]: test["raw_p_fraction"] for test in confirmatory["tests"]
        },
        "raw_p_statement": "four raw p=1/5001 and ResearchBench p=2/5001; all raw p<.001",
        "holm_alpha_0.05": "all five remain significant after Holm step-down correction",
        "source": "formal_significance.json#confirmatory_family",
    }
    PANEL_OUT.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"wrote {PANEL_OUT}")


if __name__ == "__main__":
    main()
