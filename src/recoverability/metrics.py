"""Scorer-disjoint recoverability metrics (lexical + retrieval-rank over bag-of-words).

Primary family intentionally avoids embedding models so it remains disjoint from
common embedding-similarity outcome scorers used by idea-generation benches.
A secondary embedding family (R_embed) is computed in
scripts/run_embedding_recoverability.py when outcomes are lexical; do not mix
families into the same predictor/outcome pair without a scorer-disjointness argument.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.I)

# Minimal scientific/stopword set; keep domain terms.
_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "from", "as", "is", "are", "was", "were", "be", "been", "being", "that",
    "this", "these", "those", "it", "its", "we", "our", "their", "they", "at",
    "into", "via", "using", "use", "used", "can", "may", "also", "than", "such",
    "which", "who", "whom", "what", "when", "where", "how", "not", "no", "but",
    "if", "then", "than", "so", "do", "does", "did", "have", "has", "had",
    "will", "would", "should", "could", "between", "among", "over", "under",
    "about", "more", "most", "other", "only", "both", "each", "few", "many",
    "much", "some", "any", "all", "while", "during", "after", "before",
    "paper", "study", "result", "results", "method", "methods", "based",
    "show", "shows", "shown", "propose", "proposed", "approach", "work",
}


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    if not text:
        return []
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    if drop_stop:
        toks = [t for t in toks if t not in _STOP and len(t) > 1]
    return toks


def text_token_length(text: str) -> int:
    """Token length used only for intervention matching, before stopword removal."""
    return len(tokenize(text, drop_stop=False))


def length_nearest_derangement(texts: Sequence[str]) -> list[int]:
    """Return a deterministic, donor-unique, length-nearest derangement.

    Items are sorted by raw token length (then original index). Adjacent items are
    swapped; for an odd population, the final three are rotated. Consequently
    every source receives another source's text, every donor is used exactly once,
    and local length distance is minimized without stochastic search.
    """
    n = len(texts)
    if n < 2:
        raise ValueError("a derangement requires at least two texts")
    order = sorted(range(n), key=lambda i: (text_token_length(texts[i]), i))
    donor_for = [-1] * n
    pair_limit = n if n % 2 == 0 else n - 3
    for pos in range(0, pair_limit, 2):
        left, right = order[pos], order[pos + 1]
        donor_for[left] = right
        donor_for[right] = left
    if n % 2:
        first, second, third = order[-3:]
        donor_for[first] = second
        donor_for[second] = third
        donor_for[third] = first
    if sorted(donor_for) != list(range(n)):
        raise AssertionError("length-nearest mapping is not a donor permutation")
    if any(i == donor for i, donor in enumerate(donor_for)):
        raise AssertionError("length-nearest mapping contains a self-pair")
    return donor_for


def length_match_diagnostics(texts: Sequence[str], donor_for: Sequence[int]) -> dict:
    """Summarize and validate an itemwise length-matching assignment."""
    n = len(texts)
    if len(donor_for) != n or sorted(donor_for) != list(range(n)):
        raise ValueError("donor_for must be a permutation over texts")
    if any(i == donor for i, donor in enumerate(donor_for)):
        raise ValueError("donor_for must be a derangement")
    lengths = [text_token_length(text) for text in texts]
    donor_lengths = [lengths[donor] for donor in donor_for]
    absolute_log_ratios = [
        abs(math.log((donor_length + 1) / (source_length + 1)))
        for source_length, donor_length in zip(lengths, donor_lengths)
    ]
    relative_gaps = [
        abs(donor_length - source_length) / max(source_length, 1)
        for source_length, donor_length in zip(lengths, donor_lengths)
    ]
    ordered_log = sorted(absolute_log_ratios)

    def quantile(values: Sequence[float], fraction: float) -> float:
        return values[int(fraction * (len(values) - 1))]

    return {
        "algorithm": "deterministic_length_nearest_derangement_v1",
        "length_unit": "regex_tokens_before_stopword_removal",
        "n": n,
        "donor_unique_count": len(set(donor_for)),
        "self_pair_count": sum(i == donor for i, donor in enumerate(donor_for)),
        "mean_source_tokens": sum(lengths) / n,
        "mean_donor_tokens": sum(donor_lengths) / n,
        "median_absolute_log_length_ratio": quantile(ordered_log, 0.5),
        "p90_absolute_log_length_ratio": quantile(ordered_log, 0.9),
        "max_absolute_log_length_ratio": max(absolute_log_ratios),
        "fraction_within_10_percent_source_length": sum(gap <= 0.10 for gap in relative_gaps) / n,
        "fraction_within_20_percent_source_length": sum(gap <= 0.20 for gap in relative_gaps) / n,
    }


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def lexical_coverage(input_text: str, target_text: str) -> float:
    """Fraction of target content tokens that appear in the input (availability)."""
    tgt = set(tokenize(target_text))
    if not tgt:
        return 0.0
    inp = set(tokenize(input_text))
    return len(tgt & inp) / len(tgt)


def rouge_l_f1(input_text: str, target_text: str) -> float:
    """Character-level-ish token LCS F1 (ROUGE-L style) on tokens."""
    x, y = tokenize(input_text), tokenize(target_text)
    if not x or not y:
        return 0.0
    # LCS length DP
    m, n = len(x), len(y)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    lcs = dp[n]
    prec = lcs / m
    rec = lcs / n
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def bow_cosine(a: str, b: str) -> float:
    ca, cb = Counter(tokenize(a)), Counter(tokenize(b))
    if not ca or not cb:
        return 0.0
    keys = set(ca) | set(cb)
    dot = sum(ca[k] * cb[k] for k in keys)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def target_rank_among_candidates(
    input_text: str,
    true_target: str,
    candidate_targets: Sequence[str],
    score_fn=bow_cosine,
) -> dict:
    """Identifiability: rank of true target among candidates by score_fn(input, cand).

    Rank is 1-based (1 = best). If true_target not in candidates, it is appended.
    """
    cands = list(candidate_targets)
    if true_target not in cands:
        cands.append(true_target)
    scored = [(score_fn(input_text, c), i, c) for i, c in enumerate(cands)]
    # higher score better; stable by index
    scored.sort(key=lambda t: (-t[0], t[1]))
    rank = None
    true_score = score_fn(input_text, true_target)
    for r, (s, i, c) in enumerate(scored, start=1):
        if c == true_target:
            rank = r
            break
    assert rank is not None
    # margin: true score - second best if true is best, else true - best
    scores_only = [s for s, _, _ in scored]
    if rank == 1 and len(scores_only) > 1:
        margin = scores_only[0] - scores_only[1]
    else:
        margin = true_score - scores_only[0]
    return {
        "rank": rank,
        "n_candidates": len(cands),
        "reciprocal_rank": 1.0 / rank,
        "true_score": true_score,
        "margin": margin,
        "top_score": scores_only[0],
    }


def recoverability_bundle(
    input_text: str,
    target_text: str,
    candidate_targets: Sequence[str] | None = None,
) -> dict:
    """Composite recoverability using scorer-disjoint lexical/BoW measures."""
    out = {
        "lexical_coverage": lexical_coverage(input_text, target_text),
        "token_jaccard": token_jaccard(input_text, target_text),
        "rouge_l_f1": rouge_l_f1(input_text, target_text),
        "bow_cosine": bow_cosine(input_text, target_text),
    }
    if candidate_targets is not None:
        rank_info = target_rank_among_candidates(
            input_text, target_text, candidate_targets
        )
        out.update({f"id_{k}": v for k, v in rank_info.items()})
    # Simple mean of availability measures (not including rank)
    out["R_availability"] = (
        out["lexical_coverage"] + out["token_jaccard"] + out["rouge_l_f1"] + out["bow_cosine"]
    ) / 4.0
    if candidate_targets is not None:
        out["R_identifiability"] = out["id_reciprocal_rank"]
        out["R_composite"] = 0.5 * out["R_availability"] + 0.5 * out["R_identifiability"]
    else:
        out["R_composite"] = out["R_availability"]
    return out
