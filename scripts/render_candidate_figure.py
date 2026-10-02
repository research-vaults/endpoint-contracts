#!/usr/bin/env python3
"""Regenerate the AI for Meta-Science candidate-discrimination figure from frozen item-level ledgers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
OUTPUTS = [PROJECT / "results/figures"]

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 6.5,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

RENDER_WIDTH_IN = 148.42 / 72.0
PDF_METADATA = {"CreationDate": None, "Producer": None, "Creator": None}


def sem(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0


def main() -> None:
    dual = json.loads((PROJECT / "results/dual_axis_rid_ravail.json").read_text())
    fire = json.loads((PROJECT / "results/firebench_recoverability.json").read_text())
    rank = json.loads((PROJECT / "results/researchbench_generation_audit.json").read_text())
    retrieve = json.loads(
        (PROJECT / "results/researchbench_retrieve_recoverability.json").read_text()
    )

    fire_rows = fire["rows"]
    inst_r = [row["instruction"]["R_avail"] for row in fire_rows if row.get("instruction")]
    inst_m = [row["instruction"]["mrr"] for row in fire_rows if row.get("instruction")]
    rq_r = [row["RQ"]["R_avail"] for row in fire_rows if row.get("RQ")]
    rq_m = [row["RQ"]["mrr"] for row in fire_rows if row.get("RQ")]
    rank_r = [row["R_avail_RQ_to_gold"] for row in rank["ranking_rows"]]
    rank_m = [row["mrr"] for row in rank["ranking_rows"]]
    retrieve_r = [row["R_RQ_to_gold_abs"] for row in retrieve["rows"]]
    retrieve_m = [row["mrr_proxy"] for row in retrieve["rows"]]

    points = [
        ("FIRE instr", inst_r, inst_m, "#1f4e79", (-9, 4), "right", "bottom"),
        # The short label and nine-point gap keep its right edge clear of the
        # seven-point green marker after placement in the compiled paper.
        ("FIRE RQ", rq_r, rq_m, "#2ca02c", (-9, -2), "right", "center"),
        ("RB retrieve", retrieve_r, retrieve_m, "#c55a11", (7, 4), "left", "bottom"),
        ("RB rank", rank_r, rank_m, "#d62728", (8, 4), "left", "bottom"),
    ]
    authority_names = {
        "FIRE instr": "FIRE instruction→conclusion",
        "FIRE RQ": "FIRE RQ→conclusion",
        "RB retrieve": "RB retrieve (RQ→gold insp)",
        "RB rank": "RB ranking (RQ alone)",
    }
    authority = {
        row["setting"]: row
        for row in dual["settings"]
        if row.get("R_id_MRR") is not None
    }

    fig, ax = plt.subplots(
        figsize=(RENDER_WIDTH_IN, 99.9 / 72.0), layout="constrained"
    )
    for label, x_values, y_values, color, offset, horizontal, vertical in points:
        x = float(np.mean(x_values))
        y = float(np.mean(y_values))
        expected = authority[authority_names[label]]
        if not np.isclose(x, expected["R_avail"], atol=1e-12):
            raise ValueError(f"R_avail mismatch for {label}")
        if not np.isclose(y, expected["R_id_MRR"], atol=1e-12):
            raise ValueError(f"R_id mismatch for {label}")
        ax.errorbar(
            x,
            y,
            xerr=sem(x_values),
            yerr=sem(y_values),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=3,
            markersize=7,
            alpha=0.95,
        )
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=offset,
            ha=horizontal,
            va=vertical,
            fontsize=6.0,
            color=color,
        )

    ax.set_xlabel("Endpoint availability")
    ax.set_ylabel("Gold retrieval (MRR)")
    ax.set_title("Pool-conditional endpoint retrieval")
    ax.set_xlim(0.0, 0.62)
    ax.set_ylim(0.0, 1.08)
    ax.text(
        0.98,
        0.03,
        "SEM error bars (item-level)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
        color="0.35",
    )

    for output in OUTPUTS:
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / "fig6_dual_axis.pdf", metadata=PDF_METADATA)
        fig.savefig(output / "fig6_dual_axis.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
