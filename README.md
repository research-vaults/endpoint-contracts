# Endpoint Measurement Contracts

Reproducibility materials for **Is the Endpoint Well-Posed? Measurement
Contracts for AI-Assisted Scientific Discovery**.

Historical-answer agreement can reflect information already available in the
input. This project measures textual endpoint availability under disclosure and
evidence swaps, and separates retrieval within a declared candidate pool from
scientific correctness or uniqueness. It provides measurement diagnostics, not
a causal test of which strategy a model actually used.

## Release scope

This is the **2026-10-02 reproducibility snapshot**: original metric code,
offline analysis/plotting scripts, numerical result ledgers, retained generated
model responses, embedding vectors, and anonymous forced-choice judgments.
No manuscript PDF is included. This public repository is **not certified
anonymous for conference review**; use a separate reviewer-facing snapshot.

| Path | Purpose |
|---|---|
| `src/recoverability/` | Text-similarity metrics, retrieval diagnostics and length-nearest derangements |
| `scripts/` | Ledger reconstruction, statistical tests, shared-item rankings and plots |
| `results/` | Original derived measurements and retained model response/vector artifacts |
| `results/contract_checks/` | Retrospective closer-donor, candidate-pool, ranking and extraction checks |
| `data/human_criterion_instrument/` | Anonymous labels, numerical comparison key and scoring script |
| `tests/` | Metric and released-ledger consistency tests |
| `DATA_SOURCES.md` | Input/licensing boundaries and conditional full-input replay |
| `SHA256SUMS`, `RELEASE_FILES.json` | Exact release integrity and file allowlist |

## Setup

Python 3.11 was tested; Python 3.10+ is expected. From the checkout root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt pytest
shasum -a 256 -c SHA256SUMS
PYTHONPATH=src python -m pytest -q -p no:cacheprovider
```

No API account, credential, paid model call or external dataset is needed for
the commands below. `requirements-tested.txt` records the exact packages used
for release testing; use it instead of the minimum requirements to match that
environment. CPU execution is sufficient.

## Offline reproduction

Run these in a disposable clean checkout: they regenerate the named outputs,
so integrity checks should be run **before** replay.

```sh
python scripts/regenerate_dual_axis_ledger.py
python scripts/regenerate_formal_significance.py
python scripts/regenerate_missing_item_tipping_bound.py
python scripts/regenerate_metric_robustness.py
python scripts/run_contract_checks.py --panel-only
python scripts/regenerate_main_figures.py
python scripts/regenerate_sampling_flow_figure.py
python scripts/regenerate_supplement_figures.py
python scripts/render_contract_figures.py
python scripts/render_candidate_figure.py
python data/human_criterion_instrument/score_forced_choice_labels.py \
  data/human_criterion_instrument/received/annotatorA_pairs_labels.csv \
  data/human_criterion_instrument/received/annotatorB_pairs_labels.csv
```

Expected outputs are JSON ledgers under `results/`, a refreshed
`results/contract_checks/panel_checks.json`, human agreement summaries in
`results/human_criterion_forced_choice.json`, and PDF/PNG plots in
`results/figures/`. Plotting reads existing measurements; it does not rerun
models. Formal-significance reconstruction reruns the stated sign-flip tests
and rebuilds cross-benchmark summaries but preserves separately computed
disclosure/panel inference blocks. Metric-robustness reconstruction preserves
the stored leave-one-metric-out correlations. These are deliberately partial
reconstruction entry points, not claims of new independent replication.

`results/openrouter_panel/raw_cache.jsonl` and
`results/firebench_panel/raw_cache.jsonl` contain retained generated answer text,
not complete provider response envelopes or hidden reasoning traces. Model
outputs may be inaccurate; they are evidence of model behavior, not endorsed
scientific findings. Numerical rows and original frozen scores are retained
separately. No provider secrets or live API scripts are included.
The disclosure panel contains 228 numerical score rows but only 72 retained
answer texts; FIRE contains 260 score rows and 260 retained answers. Missing
disclosure answers are not reconstructed or represented as available.

## Interpretation and limits

The disclosure population is an author-constructed 86-item proxy, not official
ProjectionBench data. Evidence swaps change relevance as well as identity.
The two-reader pilot supports coarse relative recoverability, not scientific
admissibility. Candidate-pool MRR/gold@1 measures retrieval within the declared
pool, not uniqueness among all scientifically valid answers. Same-family and
scorer-disjoint associations do not identify causal shortcut use. Common-item
aggregate ranking changes remain uncertain.

The original text epoch is not fully recoverable; later retained-text checks
are distinct from historical estimates. The manifest in
`results/contract_checks/run_manifest.json` describes the historical computation,
not a hash assertion for the renamed public scripts. Public file integrity is
defined by `SHA256SUMS`. No scientific numeric values were changed for release.

## Included, excluded and licensing

Original code and derived measurements use the existing MIT license. Generated
responses are supplied as experimental records subject to any underlying
third-party rights; they are not a license to third-party publications.
Upstream paper/benchmark titles and identifiers are preserved for attribution.
See `DATA_SOURCES.md` for sources and the exact reproduction boundary.

Excluded: full scholarly-text corpora without verified redistribution rights,
private datasets, manuscripts/old drafts, reviews, comparator archives, planning
documents, correspondence, credentials, local paths, unrelated repositories,
large checkpoints and unneeded temporary outputs. Full text-level reconstruction
requires separately obtained inputs; it is not represented as one-command
reproducibility from this public repository.
