# Data sources, permissions and reproduction boundary

Public accessibility is not a redistribution license. This release therefore
ships original derived measurements, not the upstream scholarly-text corpora.
It preserves source paper/task identifiers and scientific attribution.

| Material | Shipped | Source / boundary |
|---|---|---|
| Disclosure proxy | Item IDs, metrics, embedding vectors and generated predictions; no source excerpts | arXiv/ar5iv papers identified in `results/disclosure_item_ids.json` and numerical rows; per-paper text redistribution permission not established |
| FIRE-Bench | Task IDs, derived metrics and generated panel answers; no task corpus | [FIRE-Bench project](https://firebench.github.io/), [paper](https://arxiv.org/abs/2602.02905); retained subset has 35 conclusions, not necessarily the current upstream release |
| AI Idea Bench | Derived numerical measurements and title identifiers; no source corpus | [AI Idea Bench paper](https://arxiv.org/abs/2504.14191), underlying source collection remains separately licensed |
| ResearchBench | Derived pilot measurements; no corpus | [ResearchBench](https://ankitala.github.io/ResearchBench/), [paper](https://aclanthology.org/2026.findings-acl.644/); tiny pilot is not the full benchmark |
| Human pilot | Generic A/B choices, confidence and numerical comparison key | Both readers labeled 60 pairs; no names, backgrounds, recruitment/compensation records, free-text notes or source excerpts |
| Model observations | Retained generated answer text, scores and embeddings | No credentials, request routing, full provider envelopes or hidden reasoning; unretained traces cannot be reconstructed |

The two response caches are generated experimental outputs, not supplied source
documents. Preserve original outputs when inspecting model errors. No claims
are made about factual correctness of the generated prose. Third-party source
licenses continue to apply and are not replaced by the project MIT license.
No third-party code is vendored or stripped of attribution.

## Conditional full-input replay

`python scripts/run_contract_checks.py --panel-only` works using shipped scores.
The full `python scripts/run_contract_checks.py` additionally requires:

* `data/disclosure_pilot/items.jsonl`: 86 retained full-HTML items with `item_id`,
  `has_html_body` and `stages` (`S0_topic`, `S1_abstract`, `S2_intro`,
  `S3_methods_mid`, `S4_body`, `target`).
* `data/firebench/tasks.jsonl`: the 35 retained task records, with `task_id`,
  `instruction`, `research_question` and `conclusion`.

Obtain material only under the appropriate upstream permissions. Reconstructing
these processed files from a newer upstream release need not reproduce the
historical bytes or donor assignments. The historical run manifest records
input hashes for comparison; it does not guarantee those versions remain
available. There is no turnkey exact recovery of the original disclosure epoch.
The code fails with an explicit missing-input explanation instead of making
network requests or fabricating replacements. No paid generation is performed.

## Public export transformations

Internal analysis directory/script labels were replaced by descriptive
`contract_checks` names. Legacy protocol prefixes were consistently renamed to
`measurement-contracts` while retaining their version suffixes. Figure output
paths now point to `results/figures/`. A `--panel-only` entry point exposes the
unchanged common-item analysis without requiring withheld corpora. Human-label
CSVs retain only scoring-relevant anonymous fields; the numerical key no longer
contains an internal review note. These are packaging/interface changes, not
new experiments or altered scientific observations. Original source and all
private evidence remain preserved separately.

The extraction-quality ledger retains numeric flags and item IDs but omits
source-text snippets. Human-result provenance hashes refer to the reduced
public label files; choices, scores and uncertainty estimates are unchanged.
