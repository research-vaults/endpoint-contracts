#!/usr/bin/env python3
"""Regenerate result figures 1–4 with TrueType-embedded fonts (no Type 3).

Uses pdf.fonttype=42. Numbers drawn from frozen results ledgers.
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
        "lines.linewidth": 1.1,
    }
)

ROOT = Path(__file__).resolve().parents[1]
OUTS = [ROOT / "results/figures"]

# The manuscript includes every plot at 0.68\linewidth in a two-column ACL
# layout, i.e. 0.68 * 7.7cm = 148.42pt = 2.0614in.  Authoring each canvas at
# exactly that width and saving without a tight bounding box keeps the
# native-to-rendered scale factor at 1.0, so every glyph reaches the page at
# its nominal point size instead of being shrunk by the includegraphics scale.
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


def se(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def fig1() -> None:
    pc = json.loads((ROOT / "results/pilot_controls.json").read_text())
    ss = pc["stage_summary"]
    order = [
        ("S0", "S0_topic"),
        ("S1", "S1_abstract"),
        ("S2", "S2_intro"),
        ("S3", "S3_methods_mid"),
        ("S4", "S4_body"),
    ]
    labels = [a for a, _ in order]
    means = [ss[k]["R_availability"]["mean"] for _, k in order]
    sds = [ss[k]["R_availability"]["stdev"] for _, k in order]
    x = np.arange(len(labels))
    fig, ax = canvas(108.4)
    ax.errorbar(
        x,
        means,
        yerr=sds,
        fmt="-o",
        color="#1f4e79",
        ecolor="#6c8ebf",
        capsize=3,
        linewidth=1.5,
        markersize=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Disclosure stage")
    ax.set_ylabel(r"Mean $R$ (availability)")
    ax.set_title(r"Disclosure recoverability ($N{=}86$)")
    save(fig, "fig1_disclosure_R")


def fig3() -> None:
    ei = json.loads((ROOT / "results/evidence_interventions.json").read_text())
    rows = ei["rows"]

    def ms(key):
        v = np.array([r[key] for r in rows], float)
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))

    vals_sems = [ms("R_orig"), ms("R_shuffled"), ms("R_topic"), ms("R_masked")]
    # Wrapped onto two lines to remain legible at the manuscript's column width.
    cats = ["True\nS3", "Deranged\nS3", "Topic\nonly", "Full\nmask"]
    vals = [a for a, _ in vals_sems]
    sems = [b for _, b in vals_sems]
    fig, ax = canvas(100.3)
    ax.bar(
        cats,
        vals,
        yerr=sems,
        color=["#1f4e79", "#c55a11", "#7f7f7f", "#bfbfbf"],
        capsize=3,
        width=0.65,
    )
    ax.set_ylabel(r"Mean $R$ (availability)")
    ax.set_title(r"Evidence interventions ($N{=}86$)")
    # The full mask is exactly zero for every item; mark its zero-height bar.
    ax.plot(3, 0, marker="_", color="#595959", markersize=8, zorder=4)
    save(fig, "fig3_interventions")


def fig2() -> None:
    panel = json.loads((ROOT / "results/openrouter_panel/panel_rows.json").read_text())

    def short(m: str) -> str:
        if "gemma" in m:
            return "gemma-3-4b-it"
        if "llama" in m:
            return "llama-3.2-3b"
        if "gpt" in m:
            return "gpt-4o-mini"
        if "qwen" in m:
            return "qwen-2.5-7b"
        return m.split("/")[-1][:12]

    colors = {
        "gemma-3-4b-it": "#2ca02c",
        "llama-3.2-3b": "#1f77b4",
        "gpt-4o-mini": "#ff7f0e",
        "qwen-2.5-7b": "#d62728",
    }
    xs = np.array([r["R_availability"] for r in panel], float)
    ys = np.array([r["score_lex_target_in_pred"] for r in panel], float)
    mods = [short(r["model"]) for r in panel]
    fig, ax = canvas(109.2)
    for m, c in colors.items():
        mask = np.array([mm == m for mm in mods])
        if mask.any():
            ax.scatter(
                xs[mask],
                ys[mask],
                s=12,
                alpha=0.55,
                label=m,
                c=c,
                edgecolors="none",
            )
    coef = np.polyfit(xs, ys, 1)
    xline = np.linspace(xs.min(), xs.max(), 100)
    yline = coef[0] * xline + coef[1]
    rng = np.random.default_rng(20260731)
    boots = []
    n = len(xs)
    for _ in range(800):
        idx = rng.integers(0, n, n)
        c = np.polyfit(xs[idx], ys[idx], 1)
        boots.append(c[0] * xline + c[1])
    boots = np.array(boots)
    ax.fill_between(
        xline,
        np.quantile(boots, 0.025, axis=0),
        np.quantile(boots, 0.975, axis=0),
        color="0.75",
        alpha=0.5,
    )
    ax.plot(xline, yline, "k-", linewidth=1.4)
    ax.set_xlabel(r"Input $R$ (availability)")
    ax.set_ylabel(r"Endpoint score $s$ (lexical)")
    ax.set_title("Score vs recoverability (228 cells)", fontsize=6.5)
    # Opaque frame, model series only: with a transparent legend the scatter
    # points and the OLS band/fit drew straight through the entries that label
    # them, and a six-entry legend was wider than the axes.  The black fit line
    # and the grey bootstrap band are identified in the caption instead.
    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        framealpha=1.0,
        edgecolor="0.8",
        fancybox=False,
        borderpad=0.25,
        labelspacing=0.2,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.6,
        markerscale=0.8,
    ).set_zorder(10)
    save(fig, "fig2_R_vs_score")


def fig4() -> None:
    formal = json.loads((ROOT / "results/formal_significance.json").read_text())
    ai = json.loads(
        (ROOT / "results/ai_idea_bench_recoverability_full.json").read_text()
    )["summary"]
    disclosure = json.loads((ROOT / "results/evidence_interventions.json").read_text())
    rb = json.loads((ROOT / "results/researchbench_generation_audit.json").read_text())
    fb = json.loads((ROOT / "results/firebench_recoverability.json").read_text())

    table = {row["bench"]: row for row in formal["cross_bench_table"]}
    disc_true = [r["R_orig"] for r in disclosure["rows"]]
    disc_shuf = [r["R_shuffled"] for r in disclosure["rows"]]
    rb_true = [r["RQ_insp"]["R_avail"] for r in rb["generation_rows"]]
    rb_shuf = [r["RQ_shuffled_insp"]["R_avail"] for r in rb["generation_rows"]]
    fire_true = [r["instruction"]["R_avail"] for r in fb["rows"]]
    fire_shuf = [r["shuffled_instruction"]["R_avail"] for r in fb["rows"]]

    items = [
        ("Disclosure", table["Disclosure proxy"], se(disc_true)[1], se(disc_shuf)[1]),
        (
            "AI Idea",
            table["AI Idea Bench"],
            ai["sem_R_topic_refs"],
            ai["sem_R_topic_shuffled_refs"],
        ),
        ("RB tiny", table["ResearchBench tiny"], se(rb_true)[1], se(rb_shuf)[1]),
        ("FIRE", table["FIRE-Bench"], se(fire_true)[1], se(fire_shuf)[1]),
    ]
    labels = [item[0] for item in items]
    true_means = np.array([item[1]["R_true"] for item in items])
    shuf_means = np.array([item[1]["R_shuffled"] for item in items])
    true_sems = np.array([item[2] for item in items])
    shuf_sems = np.array([item[3] for item in items])
    deltas = [item[1]["delta_interv"] for item in items]
    fig, ax = canvas(81.5)
    y = np.arange(len(labels))
    for yi, shuf, true in zip(y, shuf_means, true_means):
        ax.plot([shuf, true], [yi, yi], color="0.70", linewidth=1.4, zorder=1)
    ax.errorbar(
        shuf_means,
        y,
        xerr=1.96 * shuf_sems,
        fmt="o",
        color="#c55a11",
        capsize=2,
        markersize=4.5,
        label="Length-nearest donor",
        zorder=2,
    )
    ax.errorbar(
        true_means,
        y,
        xerr=1.96 * true_sems,
        fmt="o",
        color="#1f4e79",
        capsize=2,
        markersize=4.5,
        label="True evidence",
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Mean four-measure $R$ (availability)", fontsize=6.0)
    ax.set_title("Evidence-identity interventions", fontsize=6.5)
    xmax = float(max(true_means + 1.96 * true_sems)) + 0.105
    ax.set_xlim(0, xmax)
    # Right-aligned in a reserved gutter: anchoring each delta to its own marker
    # placed the text on top of the true-evidence error-bar cap.
    for yi, delta in zip(y, deltas):
        ax.text(
            xmax * 0.995,
            yi,
            rf"$\Delta={delta:.3f}$",
            va="center",
            ha="right",
            fontsize=6.0,
        )
    save(fig, "fig4_cross_bench")


def main() -> None:
    fig1()
    fig2()
    fig3()
    fig4()


if __name__ == "__main__":
    main()
