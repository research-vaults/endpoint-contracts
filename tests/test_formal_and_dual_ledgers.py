import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_result(name: str):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def test_confirmatory_plus_one_values_and_holm_decisions_are_exact():
    family = load_result("formal_significance.json")["confirmatory_family"]
    tests = family["tests"]
    assert family["seed"] == 20260730
    assert family["draws_per_test"] == 5000
    assert [test["n"] for test in tests] == [86, 86, 35, 12, 86]
    assert [test["plus_one_numerator"] for test in tests] == [1, 1, 1, 2, 1]
    assert {test["plus_one_denominator"] for test in tests} == {5001}
    assert all(test["raw_p"] < 0.001 for test in tests)
    assert all(test["holm_reject"] for test in tests)


def test_dual_axis_ai_idea_keeps_composite_and_component_typed():
    dual = load_result("dual_axis_rid_ravail.json")
    ai_authority = load_result("ai_idea_bench_recoverability_full.json")["summary"]
    ai_row = next(row for row in dual["settings"] if row["setting"] == "AI Idea topic+refs")
    assert dual["R_avail_metric_contract"]["family"] == "R_availability_four_measure_mean"
    assert ai_row["R_avail"] == ai_authority["mean_R_topic_refs"]
    assert ai_row["lexical_coverage_component"] == ai_authority["lexical_coverage_component"]["mean_topic_refs"]
    assert ai_row["R_avail"] != ai_row["lexical_coverage_component"]


def test_disclosure_hurt_thresholds_are_typed_and_reconcile_to_rows():
    interventions = load_result("evidence_interventions.json")
    deltas = [row["delta_shuffle"] for row in interventions["rows"]]
    n = interventions["n"]
    assert n == len(deltas) == 86
    assert sum(delta > 0 for delta in deltas) == 84
    assert sum(delta > 0.01 for delta in deltas) == 83
    assert interventions["fraction_shuffle_hurts"] == 84 / 86
    assert interventions["fraction_shuffle_hurts_gt_0_01"] == 83 / 86


def test_missing_item_tipping_bound_reconciles_and_preserves_scope():
    interventions = load_result("evidence_interventions.json")
    bound = load_result("missing_item_tipping_bound.json")
    population = bound["population"]
    completion = bound["worst_case_completion"]
    observed_sum = sum(row["delta_shuffle"] for row in interventions["rows"])

    assert population == {
        "sampling_frame_n": 97,
        "observed_full_html_n": 86,
        "missing_full_html_n": 11,
        "missing_reason": "full ar5iv HTML unavailable",
    }
    assert bound["observed"]["sum_item_contrasts"] == observed_sum
    assert completion["full_frame_mean_identification_region"] == [
        (observed_sum - 11) / 97,
        (observed_sum + 11) / 97,
    ]
    assert completion["required_missing_mean_for_full_frame_zero"] == -observed_sum / 11
    assert completion["required_missing_mean_is_feasible"] is False
    assert completion["minimum_missing_items_at_delta_minus_one_for_nonpositive_mean"] == 12
    assert completion["actual_missing_items"] == 11
    assert completion["sign_identified_positive_under_fixed_pair_completion"] is True
    assert "holding the 86 observed pair contrasts fixed" in bound["estimand"]


def test_adjusted_panel_association_is_clustered_and_scope_cautious():
    adjusted = load_result("panel_adjusted_association.json")
    population = adjusted["population"]
    base = adjusted["models"]["base_surface_temporal"]
    conservative = adjusted["models"]["plus_generation_length_conservative"]
    within = adjusted["within_item_fixed_effects"]

    assert population["panel_rows"] == 228
    assert population["disclosure_items"] == 32
    assert population["publication_years"] == ["2023", "2024", "2025"]
    assert base["full_column_rank"] is True
    assert conservative["full_column_rank"] is True
    assert base["beta_R_avail"] > 0
    assert conservative["beta_R_avail"] > 0
    assert base["item_cluster_bootstrap"]["replicates"] == 5000
    assert base["item_cluster_bootstrap"]["seed"] == 20260802
    assert base["item_cluster_bootstrap"]["ci95_percentile"][0] > 0
    assert conservative["item_cluster_bootstrap"]["ci95_percentile"][0] > 0
    assert adjusted["protocol"].startswith("measurement-contracts-1.1")
    assert within["base"]["full_column_rank"] is True
    assert within["plus_generation_length_conservative"]["full_column_rank"] is True
    assert within["base"]["item_fixed_effect_count"] == 32
    assert within["base"]["beta_R_avail"] > 0
    assert within["base"]["item_cluster_bootstrap"]["ci95_percentile"][0] > 0
    assert (
        within["plus_generation_length_conservative"]["item_cluster_bootstrap"]
        ["ci95_percentile"][0]
        > 0
    )
    assert within["bootstrap_contract"] == {
        "unit": "disclosure item (all model-stage rows retained together)",
        "seed": 20260802,
        "replicates": 3000,
    }
    assert within["leave_one_item_out"]["folds"] == 32
    assert within["leave_one_item_out"]["all_positive"] is True
    assert within["leave_one_item_out"]["beta_R_avail_range"][0] > 0
    stage = within["stage_interactions"]
    assert stage["stage_specific_beta_R_avail"]["S1_abstract"] > 0
    assert stage["stage_specific_beta_R_avail"]["S3_methods_mid"] > 0
    assert (
        stage["item_cluster_bootstrap_by_stage"]["S1_abstract"]
        ["ci95_percentile"][0]
        > 0
    )
    assert (
        stage["item_cluster_bootstrap_by_stage"]["S3_methods_mid"]
        ["ci95_percentile"][0]
        > 0
    )
    assert (
        stage["item_cluster_bootstrap_by_stage"]["S0_topic"]
        ["ci95_percentile"][0]
        < 0
        < stage["item_cluster_bootstrap_by_stage"]["S0_topic"]
        ["ci95_percentile"][1]
    )
    assert any("not a causal" in text for text in adjusted["limitations"])
    assert any("does not establish" in text for text in adjusted["limitations"])
    assert any("stage-varying" in text for text in adjusted["limitations"])
