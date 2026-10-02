#!/usr/bin/env python3
"""Regenerate the joint availability × identifiability ledger.

Every R_avail value is sourced as the operative four-measure mean. Lexical
coverage is retained only under an explicitly component-typed field.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "dual_axis_rid_ravail.json"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [round(center - half, 2), round(center + half, 2)]


def main() -> None:
    pilot = load("pilot_controls.json")
    ai = load("ai_idea_bench_recoverability_full.json")["summary"]
    rb = load("researchbench_generation_audit.json")
    retrieve = load("researchbench_retrieve_recoverability.json")
    fire = load("firebench_recoverability.json")

    if ai["metric_family"] != "R_availability_four_measure_mean":
        raise ValueError("AI Idea authority is not the operative four-measure family")
    fire_instruction_rows = [row for row in fire["rows"] if row.get("instruction")]
    fire_rq_rows = [row for row in fire["rows"] if row.get("RQ")]
    if len(fire_instruction_rows) != 35 or len(fire_rq_rows) != 30:
        raise ValueError("FIRE dual-axis populations changed")

    rb_rank = rb["ranking_identifiability"]
    settings = [
        {
            "setting": "RB ranking (RQ alone)",
            "n": rb_rank["n"],
            "R_avail": rb_rank["mean_R_RQ_to_gold"],
            "R_id_MRR": rb_rank["mean_mrr"],
            "gold_at_1_k": 4,
            "gold_at_1_n": rb_rank["n"],
            "gold_at_1_wilson": wilson(4, rb_rank["n"]),
            "mean_rank": rb_rank["mean_rank_gold"],
            "margin_vs_neg": rb_rank["mean_margin_gold_vs_neg"],
            "source": "researchbench_generation_audit.json#ranking_identifiability",
        },
        {
            "setting": "FIRE instruction→conclusion",
            "n": len(fire_instruction_rows),
            "R_avail": sum(row["instruction"]["R_avail"] for row in fire_instruction_rows)
            / len(fire_instruction_rows),
            "R_id_MRR": sum(row["instruction"]["mrr"] for row in fire_instruction_rows)
            / len(fire_instruction_rows),
            "gold_at_1_k": 31,
            "gold_at_1_n": len(fire_instruction_rows),
            "gold_at_1_wilson": wilson(31, len(fire_instruction_rows)),
            "mean_rank": sum(row["instruction"]["rank"] for row in fire_instruction_rows)
            / len(fire_instruction_rows),
            "source": "firebench_recoverability.json#rows.instruction",
        },
        {
            "setting": "FIRE RQ→conclusion",
            "n": len(fire_rq_rows),
            "R_avail": sum(row["RQ"]["R_avail"] for row in fire_rq_rows)
            / len(fire_rq_rows),
            "R_id_MRR": sum(row["RQ"]["mrr"] for row in fire_rq_rows)
            / len(fire_rq_rows),
            "gold_at_1_k": 29,
            "gold_at_1_n": len(fire_rq_rows),
            "gold_at_1_wilson": wilson(29, len(fire_rq_rows)),
            "source": "firebench_recoverability.json#rows.RQ",
        },
        {
            "setting": "RB retrieve (RQ→gold insp)",
            "n": retrieve["n"],
            "R_avail": retrieve["mean_R_RQ"],
            "R_id_MRR": retrieve["mean_mrr"],
            "gold_in_top3": retrieve["frac_gold_in_top3"],
            "mean_best_gold_rank": retrieve["mean_best_gold_rank"],
            "source": "researchbench_retrieve_recoverability.json",
        },
        {
            "setting": "Disclosure proxy S3",
            "n": pilot["n_items"],
            "R_avail": pilot["stage_summary"]["S3_methods_mid"]["R_availability"]["mean"],
            "R_id": "sc",
            "note": "no multi-candidate endpoint pool; R_id out of protocol scope",
            "source": "pilot_controls.json#stage_summary.S3_methods_mid",
        },
        {
            "setting": "AI Idea topic+refs",
            "n": ai["n"],
            "R_avail": ai["mean_R_topic_refs"],
            "lexical_coverage_component": ai["lexical_coverage_component"]["mean_topic_refs"],
            "R_id": "sc",
            "note": "structured summary gold; no ranked alternate-endpoint pool in public release",
            "source": "ai_idea_bench_recoverability_full.json#summary",
        },
    ]

    ledger = {
        "protocol": "measurement-contracts-dual-axis-R_avail-x-R_id-v2",
        "R_avail_metric_contract": {
            "family": ai["metric_family"],
            "components": ai["metric_components"],
            "aggregation": "unweighted mean of four components",
            "lexical_coverage_policy": "component only; never stored under R_avail",
        },
        "R_id_metric_contract": "MRR/gold@1 where candidate pools exist; sc = no candidate pool",
        "note": "Joint report of over-determination (R_avail) and under-determination (R_id) where candidate pools exist.",
        "settings": settings,
    }
    OUT.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
