# Is the Endpoint Well-Posed?

Reproducibility materials for **Is the Endpoint Well-Posed? Measurement
Contracts for AI-Assisted Scientific Discovery**.

[Paper: revised draft, 1 October 2026](https://odysseus-personal-website.vercel.app/materials/papers/endpoint-well-posed.pdf)
| Code and evidence snapshot: **2 October 2026**

Historical-answer agreement can reflect information already available in the
input. This project measures textual endpoint availability under disclosure and
evidence swaps, and separates retrieval within a declared candidate pool from
scientific correctness or uniqueness. These are measurement diagnostics, not
a causal test of which strategy a model actually used.

The linked paper is a revision after workshop submission, not a certified copy
of the accepted upload or a final camera-ready. It includes the closer-donor,
candidate-pool and common-item checks supplied here. See [version and input
details](DATA_SOURCES.md). The repository contains code, numerical ledgers,
retained generated answers and embeddings, and anonymous reader judgments;
source-paper corpora must be obtained separately.

## Quick start

Python 3.11 was tested; Python 3.10+ is expected. From the checkout root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-tested.txt
shasum -a 256 -c SHA256SUMS
PYTHONPATH=src python -m pytest -q -p no:cacheprovider
python scripts/run_contract_checks.py --panel-only
```

Expect all checksums to pass, **8 tests passed**, and a printed common-item
ranking summary with `results/contract_checks/panel_checks.json` regenerated.
Installation needs internet access; the checks and replay need only a CPU and
the included files, with no API account, model download or paid calls. Allow
approximately 300 MiB for the checkout and tested environment, plus package-manager
cache space. On an Apple M1 Pro with Python 3.11.1, tests took 0.7 seconds and
the panel replay 3.7 seconds after installation; other machines will vary.
`requirements.txt` gives minimum dependencies if an exact tested environment
is not needed.

## Claims and commands

Run regeneration commands in a disposable checkout: they write result files.
Check integrity **before** replay. Paths below are relative to the repository root.

| Evidence or question | Command / supplied record | Reproduction boundary |
|---|---|---|
| Disclosure and cross-benchmark recoverability; paired swap evidence | `python scripts/regenerate_dual_axis_ledger.py`; `python scripts/regenerate_formal_significance.py` | Rebuild summaries from stored item measurements; the latter reruns sign-flip tests but preserves separately computed disclosure/panel inference blocks |
| Excluded-item sensitivity and metric-family robustness | `python scripts/regenerate_missing_item_tipping_bound.py`; `python scripts/regenerate_metric_robustness.py` | Reconstruct the tipping bound and robustness summaries; stored leave-one-metric-out correlations are preserved |
| Model rankings on common items | `python scripts/run_contract_checks.py --panel-only` | Replays saved scores, not model inference; aggregate ranking changes remain uncertain |
| Closer donors, candidate-pool difficulty and sentence extraction | `results/contract_checks/{donor,pool,extraction}_checks.json`; `python scripts/run_contract_checks.py` | Records are included; the full command requires separately obtained text inputs described in [DATA_SOURCES.md](DATA_SOURCES.md) |
| Two-reader agreement on 60 frozen pairs | Scoring command below; `results/human_criterion_forced_choice.json` | Replays the retained labels; supports relative recoverability, not scientific admissibility |
| Main and supplementary plots | Five plotting commands below | Render supplied measurements without rerunning models or acquiring source corpora |

The four ledger commands above each took under 2 seconds in the tested
environment. The five plotting commands together took about 7 seconds, and
human-label scoring took 0.3 seconds. These are single-run observations, not
performance benchmarks; installation and full text-input replay are excluded.
Full text-input replay was not timed in this public checkout.

```sh
python scripts/regenerate_main_figures.py
python scripts/regenerate_sampling_flow_figure.py
python scripts/regenerate_supplement_figures.py
python scripts/render_contract_figures.py
python scripts/render_candidate_figure.py
python data/human_criterion_instrument/score_forced_choice_labels.py \
  data/human_criterion_instrument/received/annotatorA_pairs_labels.csv \
  data/human_criterion_instrument/received/annotatorB_pairs_labels.csv
```

Plots appear in `results/figures/`. Human scoring refreshes
`results/human_criterion_forced_choice.json` and writes a companion TeX summary.
Metrics live in `src/recoverability/`; tests in `tests/`; release-file hashes
and inventory in `SHA256SUMS` and `RELEASE_FILES.json`.

## Evidence and interpretation

The disclosure panel has **228 numerical score rows but only 72 retained answer
texts**; FIRE has **260 score rows and all 260 answer texts**. The answer caches
are `results/openrouter_panel/raw_cache.jsonl` and
`results/firebench_panel/raw_cache.jsonl`. They contain generated responses,
not complete provider envelopes or hidden reasoning. Missing answers are not
reconstructed. Generated prose may be scientifically incorrect.

The disclosure population is an author-constructed 86-item proxy, not official
ProjectionBench data. Evidence swaps change relevance as well as identity.
The two-reader study is a pilot. Candidate-pool MRR/gold@1 measures retrieval,
not uniqueness among valid scientific answers; score associations do not
identify causal shortcut use. The original text epoch is not fully recoverable,
and later retained-text checks are distinct from historical estimates.
Full text-level reconstruction is conditional on separately obtained inputs,
not one-command reproduction from this repository.

## Citation and rights

Paper reference: *Is the Endpoint Well-Posed? Measurement Contracts for
AI-Assisted Scientific Discovery* (2026), revised manuscript dated 1 October
2026, [available here](https://odysseus-personal-website.vercel.app/materials/papers/endpoint-well-posed.pdf).
Use the author information accompanying the paper when preparing a formal
bibliographic citation. For this code and evidence, also cite
`https://github.com/research-vaults/endpoint-contracts`, the 2 October 2026
snapshot, and the commit returned by `git rev-parse HEAD`.

Original code and derived measurements retain the existing [MIT license](LICENSE).
The original copyright attribution is unchanged; its review-era wording does
not imply that this public repository is anonymous. Generated outputs remain
subject to underlying third-party rights, and the license does not cover
upstream publications or corpora. [DATA_SOURCES.md](DATA_SOURCES.md) records
source attribution, input schemas and redistribution boundaries.
