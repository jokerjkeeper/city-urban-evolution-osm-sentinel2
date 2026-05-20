"""
Fig 3 — Structural Break Timeline (Dual City)
==============================================
Reads structural_break.csv for Taichung and Taipei and produces a
horizontal Gantt-style timeline showing the detected break year per
indicator per city, sorted by break year.

Output: output/viz/paper/fig3_structural_break.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"
VIZ_DIR      = OUTPUT_DIR / "viz" / "paper"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

DPI        = 300
YEAR_START = 2018
YEAR_END   = 2025

# ─── Display config ───────────────────────────────────────────────────────────
INDICATOR_LABELS = {
    "amenity_count":      "Amenity Count (N_am)",
    "building_count":     "Building Count (N_bld)",
    "building_coverage":  "Building Coverage (rho_b)",
    "building_area_mean": "Mean Bldg Area (A_bld)",
    "edge_density":       "Edge Density (rho_e)",
    "poi_diversity":      "POI Diversity (H_POI)",
    "shop_count":         "Shop Count (N_sh)",
    "road_length_total":  "Road Length (L_road)",
    "texture_entropy":    "Texture Entropy (H_tex)",
    "leisure_count":      "Leisure Count (N_lei)",
    "ssim":               "SSIM",
    "resnet_cosine_dist": "ResNet Cosine Dist",
}

CITY_COLORS = {
    "Taichung": "#2563EB",
    "Taipei":   "#EA580C",
}
SIG_ALPHA = {True: 1.0, False: 0.4}   # p<0.05 = solid, else faded


def load_break_data(city: str) -> pd.DataFrame:
    path = OUTPUT_DIR / city.lower() / "structural_break.csv"
    df = pd.read_csv(path)
    df["city"] = city
    df["significant"] = df["p_value"].astype(float) < 0.05
    df["best_break_year"] = df["best_break_year"].astype(int)
    df["F_stat"] = df["F_stat"].astype(float)
    df["p_value"] = df["p_value"].astype(float)
    return df


def fig3_structural_break():
    tc = load_break_data("Taichung")
    tp = load_break_data("Taipei")
    combined = pd.concat([tc, tp], ignore_index=True)

    # ── Choose indicators present in both cities ──────────────────────────────
    tc_inds = set(tc["indicator"])
    tp_inds = set(tp["indicator"])
    tc_break = tc.drop_duplicates("indicator").set_index("indicator")["best_break_year"]
    shared   = sorted(tc_inds & tp_inds, key=lambda x: tc_break.get(x, 9999))

    n_inds = len(shared)
    fig, ax = plt.subplots(figsize=(13, max(5, n_inds * 0.75 + 2)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")

    years = list(range(YEAR_START, YEAR_END + 1))
    city_offset = {"Taichung": -0.18, "Taipei": +0.18}   # vertical jitter

    for y_idx, ind in enumerate(shared):
        label = INDICATOR_LABELS.get(ind, ind)
        y_base = n_inds - y_idx - 1   # top-down ordering

        # ── Background bar spanning full time range ───────────────────────────
        ax.barh(y_base, YEAR_END - YEAR_START, left=YEAR_START,
                height=0.55, color="#E2E8F0", alpha=0.6, zorder=1)

        for city in ["Taichung", "Taipei"]:
            row = combined[(combined["indicator"] == ind) &
                           (combined["city"] == city)]
            if row.empty:
                continue
            row = row.iloc[0]
            brk  = row["best_break_year"]
            sig  = row["significant"]
            fval = row["F_stat"]
            pval = row["p_value"]
            color = CITY_COLORS[city]
            alpha = SIG_ALPHA[sig]
            y_pos = y_base + city_offset[city]

            # Before-break segment
            ax.barh(y_pos, brk - YEAR_START, left=YEAR_START,
                    height=0.28, color=color, alpha=alpha * 0.35,
                    zorder=2)
            # After-break segment
            ax.barh(y_pos, YEAR_END - brk, left=brk,
                    height=0.28, color=color, alpha=alpha * 0.85,
                    zorder=2)

            # Break point marker
            ax.scatter(brk, y_pos, s=90, color=color, alpha=alpha,
                       zorder=5, edgecolors="white" if sig else color,
                       linewidths=1.2)

            # F-stat annotation (significant only)
            if sig:
                ax.text(brk + 0.12, y_pos + 0.04,
                        f"F={fval:.1f}", fontsize=7.5, color=color,
                        va="bottom", fontweight="bold", zorder=6)

    # ── Year grid lines ───────────────────────────────────────────────────────
    for yr in years:
        ax.axvline(yr, color="#CBD5E1", linewidth=0.6, zorder=0)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(YEAR_START - 0.5, YEAR_END + 0.8)
    ax.set_ylim(-0.6, n_inds - 0.4)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=10)
    ax.set_yticks(range(n_inds))
    ax.set_yticklabels(
        [INDICATOR_LABELS.get(ind, ind) for ind in reversed(shared)],
        fontsize=10
    )
    ax.set_xlabel("Year", fontsize=12, labelpad=6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=CITY_COLORS["Taichung"], label="Taichung"),
        mpatches.Patch(color=CITY_COLORS["Taipei"],   label="Taipei"),
        plt.scatter([], [], s=80, color="gray", edgecolors="white",
                    linewidths=1.2, label="Break point (p<0.05)"),
        mpatches.Patch(color="gray", alpha=0.25,
                       label="Break point (p≥0.05, faded)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right",
              framealpha=0.9, edgecolor="#CBD5E1", ncol=2)

    ax.set_title(
        "Fig 3 — Chow Structural Break Detection: Taichung vs Taipei (2018–2025)\n"
        "Marker = detected break year; darker segment = post-break phase; "
        "F-statistic shown for p < 0.05",
        fontsize=12, fontweight="bold", pad=12
    )

    plt.tight_layout()
    out = VIZ_DIR / "fig3_structural_break.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"[Fig 3] Saved: {out}")
    plt.close()


if __name__ == "__main__":
    fig3_structural_break()
