#!/usr/bin/env python3
"""Regenerate supplement figures fig6 (dual-axis with SEM) and fig7 (S3 tercile curves).

Numbers drawn only from frozen results ledgers. TrueType fonts (pdf.fonttype=42).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
        "legend.fontsize": 6.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

ROOT = Path(__file__).resolve().parents[1]
OUTS = [ROOT / "results/figures"]

# Both supplement plots are included at 0.68\linewidth = 148.42pt.  Authoring
# the canvas at exactly that width and saving without a tight bounding box
# holds the native-to-rendered scale factor at 1.0, so no glyph is shrunk below
# its nominal point size on the page.
RENDER_WIDTH_IN = 148.42 / 72.0


def canvas(height_pt: float):
    """A constrained-layout figure whose native width equals the rendered width."""
    return plt.subplots(
        figsize=(RENDER_WIDTH_IN, height_pt / 72.0), layout="constrained"
    )


# Matplotlib stamps a wall-clock /CreationDate into every PDF, which makes an
# otherwise byte-identical figure hash differently on each regeneration and
# breaks the archive's checksum manifest. Pin the document metadata so the
# documented replay is byte-stable.
PDF_METADATA = {"CreationDate": None, "Producer": None, "Creator": None}


def save(fig, name: str) -> None:
    for d in OUTS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf", metadata=PDF_METADATA)
        fig.savefig(d / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def sem(vals) -> float:
    v = np.asarray(list(vals), float)
    if len(v) < 2:
        return 0.0
    return float(v.std(ddof=1) / np.sqrt(len(v)))


def fig6_dual_axis() -> None:
    """Joint R_avail × R_id with SEM error bars from item-level ledgers."""
    dual = json.loads((ROOT / "results/dual_axis_rid_ravail.json").read_text())
    fb = json.loads((ROOT / "results/firebench_recoverability.json").read_text())
    rb = json.loads((ROOT / "results/researchbench_generation_audit.json").read_text())
    ret = json.loads(
        (ROOT / "results/researchbench_retrieve_recoverability.json").read_text()
    )

    fire_rows = fb["rows"]
    inst_r = [r["instruction"]["R_avail"] for r in fire_rows if r.get("instruction")]
    inst_m = [r["instruction"]["mrr"] for r in fire_rows if r.get("instruction")]
    rq_r = [r["RQ"]["R_avail"] for r in fire_rows if r.get("RQ")]
    rq_m = [r["RQ"]["mrr"] for r in fire_rows if r.get("RQ")]

    rank_rows = rb["ranking_rows"]
    rb_rank_r = [r["R_avail_RQ_to_gold"] for r in rank_rows]
    rb_rank_m = [r["mrr"] for r in rank_rows]

    ret_rows = ret["rows"]
    rb_ret_r = [r["R_RQ_to_gold_abs"] for r in ret_rows]
    rb_ret_m = [r["mrr_proxy"] for r in ret_rows]

    # Means match dual_axis_rid_ravail.json; SEM from item rows.
    points = [
        {
            "label": "FIRE instr",
            "x": float(np.mean(inst_r)),
            "y": float(np.mean(inst_m)),
            "xerr": sem(inst_r),
            "yerr": sem(inst_m),
            "color": "#1f4e79",
        },
        {
            "label": "FIRE RQ→con",
            "x": float(np.mean(rq_r)),
            "y": float(np.mean(rq_m)),
            "xerr": sem(rq_r),
            "yerr": sem(rq_m),
            "color": "#2ca02c",
        },
        {
            "label": "RB retrieve",
            "x": float(np.mean(rb_ret_r)),
            "y": float(np.mean(rb_ret_m)),
            "xerr": sem(rb_ret_r),
            "yerr": sem(rb_ret_m),
            "color": "#c55a11",
        },
        {
            "label": "RB rank",
            "x": float(np.mean(rb_rank_r)),
            "y": float(np.mean(rb_rank_m)),
            "xerr": sem(rb_rank_r),
            "yerr": sem(rb_rank_m),
            "color": "#d62728",
        },
    ]

    ledger_points = {
        row["setting"]: row
        for row in dual["settings"]
        if row.get("R_id_MRR") is not None
    }
    point_to_setting = {
        "FIRE instr": "FIRE instruction→conclusion",
        "FIRE RQ→con": "FIRE RQ→conclusion",
        "RB retrieve": "RB retrieve (RQ→gold insp)",
        "RB rank": "RB ranking (RQ alone)",
    }
    for point in points:
        authority = ledger_points[point_to_setting[point["label"]]]
        if not np.isclose(point["x"], authority["R_avail"], atol=1e-12):
            raise ValueError(f"dual-axis R_avail mismatch for {point['label']}")
        if not np.isclose(point["y"], authority["R_id_MRR"], atol=1e-12):
            raise ValueError(f"dual-axis R_id mismatch for {point['label']}")

    fig, ax = canvas(99.9)
    for p in points:
        ax.errorbar(
            p["x"],
            p["y"],
            xerr=p["xerr"],
            yerr=p["yerr"],
            fmt="o",
            color=p["color"],
            ecolor=p["color"],
            elinewidth=1.1,
            capsize=3,
            markersize=7,
            alpha=0.95,
            label=p["label"],
        )
        # Per-point placement so that no two labels share a lane and no label
        # runs past the plot edge: the rightmost point is labelled to its left,
        # and the point directly beneath it is labelled underneath.
        placement = {
            "FIRE RQ→con": ((-4, -2), "right", "center"),
            "FIRE instr": ((-9, 4), "right", "bottom"),
            "RB retrieve": ((7, 4), "left", "bottom"),
            "RB rank": ((8, 4), "left", "bottom"),
        }
        offset, halign, valign = placement[p["label"]]
        ax.annotate(
            p["label"],
            (p["x"], p["y"]),
            textcoords="offset points",
            xytext=offset,
            ha=halign,
            va=valign,
            fontsize=6.0,
            color=p["color"],
        )

    # Plain text with no mathtext subscripts: subscripts render at 0.7x and
    # dropped below the 6pt floor.  The y-label is also short enough to fit the
    # canvas height, which the previous long form did not - it was clipped
    # mid-word and collided with the in-plot note.  The axis meanings are
    # spelled out in the caption and in Appendix D.
    ax.set_xlabel("R availability (over-determined →)")
    ax.set_ylabel("R identifiability = MRR")
    ax.set_title("Dual-axis endpoint snapshot")
    ax.set_xlim(0.0, 0.62)
    ax.set_ylim(0.0, 1.08)
    ax.axhline(0.5, color="0.85", linewidth=0.8, linestyle="--")
    ax.axvline(0.25, color="0.85", linewidth=0.8, linestyle="--")
    # Compact note, kept inside the axes and clear of the y-axis label lane.
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
    save(fig, "fig6_dual_axis")


def fig7_tercile_curves() -> None:
    """Mean panel score by S3 R_avail tercile (Table 23 companion)."""
    t = json.loads((ROOT / "results/s3_tertile_scores.json").read_text())
    order = ["Low", "Mid", "High"]
    by = {row["tertile"]: row for row in t["tertiles"]}
    means = [by[k]["mean_score"] for k in order]
    mean_r = [by[k]["mean_R"] for k in order]
    ns = [by[k]["n"] for k in order]

    # Approximate SEM from panel cells if available
    panel_path = ROOT / "results/openrouter_panel/panel_rows.json"
    yerr = [0.0, 0.0, 0.0]
    if panel_path.exists():
        panel = json.loads(panel_path.read_text())
        s3 = [r for r in panel if r.get("stage") in ("S3", "S3_methods_mid") or "S3" in str(r.get("stage", ""))]
        if not s3:
            # stage field may be numeric or named differently
            stages = set(r.get("stage") for r in panel)
            # try methods_mid / S3_methods_mid
            for cand in stages:
                if cand and ("S3" in str(cand) or "methods" in str(cand).lower()):
                    s3 = [r for r in panel if r.get("stage") == cand]
                    if s3:
                        break
        if s3:
            rs = np.array([float(r["R_availability"]) for r in s3])
            ss = np.array([float(r["score_lex_target_in_pred"]) for r in s3])
            order_idx = np.argsort(rs)
            rs, ss = rs[order_idx], ss[order_idx]
            n = len(ss)
            # equal-size tertiles matching ledger rule
            cuts = [0, n // 3, 2 * n // 3, n]
            yerr = []
            for i in range(3):
                chunk = ss[cuts[i] : cuts[i + 1]]
                yerr.append(sem(chunk) if len(chunk) > 1 else 0.0)

    x = np.arange(3)
    fig, ax = canvas(105.7)
    ax.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="-o",
        color="#1f4e79",
        ecolor="#6c8ebf",
        capsize=3,
        linewidth=1.6,
        markersize=6,
    )
    # Left-aligned on the first tercile and right-aligned on the last, so the
    # annotation cannot cross the y-axis spine or the right-hand plot edge.
    aligns = ["left", "center", "right"]
    offsets = [(0, 9), (0, 9), (0, 9)]
    for i, (m, r, n) in enumerate(zip(means, mean_r, ns)):
        ax.annotate(
            f"mean $R$={r:.2f}\n$n$={n}",
            (i, m),
            textcoords="offset points",
            xytext=offsets[i],
            ha=aligns[i],
            fontsize=6.0,
            color="0.3",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Low $R$", "Mid $R$", "High $R$"])
    ax.set_xlabel(r"S3 $R$ availability tercile (panel cells)")
    ax.set_ylabel(r"Mean $s$ (lexical)")
    ax.set_title("Panel score by R tercile")
    ax.set_ylim(0.0, max(means) * 1.45 + 0.02)
    save(fig, "fig7_tercile_curves")


def main() -> None:
    fig6_dual_axis()
    fig7_tercile_curves()


if __name__ == "__main__":
    main()
