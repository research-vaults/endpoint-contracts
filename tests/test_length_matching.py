from recoverability.metrics import (
    length_match_diagnostics,
    length_nearest_derangement,
)


def test_even_derangement_is_deterministic_unique_and_local():
    texts = ["a " * n for n in (5, 6, 20, 21)]
    mapping = length_nearest_derangement(texts)
    assert mapping == length_nearest_derangement(texts)
    assert sorted(mapping) == list(range(len(texts)))
    assert all(index != donor for index, donor in enumerate(mapping))
    assert mapping == [1, 0, 3, 2]


def test_odd_derangement_rotates_final_three_without_losing_a_donor():
    texts = ["a " * n for n in (5, 6, 20, 21, 22)]
    mapping = length_nearest_derangement(texts)
    assert sorted(mapping) == list(range(len(texts)))
    assert all(index != donor for index, donor in enumerate(mapping))
    assert mapping == [1, 0, 3, 4, 2]


def test_diagnostics_validate_contract_and_preserve_population_mean():
    texts = ["a " * n for n in (10, 10, 11, 11, 12, 12)]
    mapping = length_nearest_derangement(texts)
    diagnostics = length_match_diagnostics(texts, mapping)
    assert diagnostics["algorithm"] == "deterministic_length_nearest_derangement_v1"
    assert diagnostics["donor_unique_count"] == len(texts)
    assert diagnostics["self_pair_count"] == 0
    assert diagnostics["mean_source_tokens"] == diagnostics["mean_donor_tokens"]
    assert diagnostics["fraction_within_10_percent_source_length"] == 1.0
