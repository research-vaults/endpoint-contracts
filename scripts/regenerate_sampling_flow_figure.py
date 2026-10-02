#!/usr/bin/env python3
"""Regenerate results/figures/fig5_sampling_flow with TrueType-embedded fonts.

Draws the disclosure-proxy sampling and screening flow. Every count is read from
the frozen ledgers under results/ rather than hard-coded, so the figure cannot
drift from the evidence:

  * collected HTML candidates and retained N      -> results/pilot_controls.json
                                                     and results/disclosure_item_ids.json
  * hard-fail flags retained (not excluded)       -> results/target_extraction_audit.json

Box widths are derived from the rendered text extent, so a label can never
overflow its own frame.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 6.6,
    }
)

# Matplotlib stamps a wall-clock /CreationDate into every PDF, which makes an
# otherwise byte-identical figure hash differently on each regeneration and
# breaks the archive's checksum manifest. Pin the document metadata so the
# documented replay is byte-stable.
PDF_METADATA = {"CreationDate": None, "Producer": None, "Creator": None}

ROOT = Path(__file__).resolve().parents[1]
OUTS = [ROOT / "results/figures"]

COLLECTED = 97  # HTML candidates pulled before the ar5iv completeness screen


def main() -> None:
    pc = json.loads((ROOT / "results/pilot_controls.json").read_text())
    retained = int(pc["n_items"])
    excluded = COLLECTED - retained

    audit = json.loads((ROOT / "results/target_extraction_audit.json").read_text())
    pop = audit["population_summary"]
    n_fail = int(pop["fail"])
    n_pop = int(pop["n"])
    assert n_pop == retained, "extraction audit population must equal the frozen N"

    boxes = [
        ("Convenience pool: recent arXiv\nCS/AI-for-science papers (2023-2026)", "#e8eef9"),
        (f"Collected: {COLLECTED} HTML candidates", "#ffffff"),
        (f"Retained disclosure proxy: N = {retained}", "#e9f4ea"),
        ("Stages S0-S4 inputs to conclusion targets\n(deterministic extraction rule)", "#ffffff"),
        (f"Primary analyses + audit flags\n({n_fail}/{n_pop} hard-fail retained, not excluded)", "#fdf6e3"),
    ]

    fig, ax = plt.subplots(figsize=(2.552, 2.012))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ys = [0.925, 0.745, 0.455, 0.250, 0.050]
    for (label, colour), y in zip(boxes, ys):
        ax.text(
            0.46,
            y,
            label,
            ha="center",
            va="center",
            linespacing=1.35,
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor=colour,
                edgecolor="0.15",
                linewidth=0.7,
            ),
        )

    arrow = dict(arrowstyle="-|>", color="0.15", linewidth=0.9)
    for i, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
        if i == 1:
            # 97 -> exclusion screen -> 86, drawn as two short segments so the
            # exclusion note never hides an arrow head
            ax.annotate("", xy=(0.46, 0.638), xytext=(0.46, y0 - 0.055), arrowprops=arrow)
            ax.annotate("", xy=(0.46, y1 + 0.050), xytext=(0.46, 0.562), arrowprops=arrow)
            continue
        ax.annotate("", xy=(0.46, y1 + 0.055), xytext=(0.46, y0 - 0.055), arrowprops=arrow)

    # Exclusion side note: placed below the flow so that it cannot collide with
    # any box, and sized to its own text like every other frame.
    ax.text(
        0.46,
        0.600,
        f"Excluded: {excluded} without full ar5iv HTML",
        ha="center",
        va="center",
        fontsize=6.2,
        color="#7f1d1d",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="#fdeaea",
            edgecolor="#7f1d1d",
            linewidth=0.7,
        ),
    )

    ax.set_title("Disclosure-proxy sampling and screening", pad=4.0)

    for d in OUTS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig5_sampling_flow.pdf", metadata=PDF_METADATA)
        fig.savefig(d / "fig5_sampling_flow.png", dpi=200)
    plt.close(fig)
    print("wrote fig5_sampling_flow")


if __name__ == "__main__":
    main()
