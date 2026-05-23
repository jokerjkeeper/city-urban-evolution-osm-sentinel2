"""
論文圖表生成模組（paper_figures.py）
=====================================
生成投稿（CITIES v2 — 雙城市比較研究）所需的 3 張高品質圖表：

  Fig 1 — 研究區域地圖（雙城市並排，各 195 採樣點空間分佈）
  Fig 2 — Moran's I 年度變化折線圖（雙城市對比）
  Fig 4 — PCA 碎石圖（雙城市解釋方差 + 累積曲線）

輸出目錄：output/viz/paper/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib import font_manager, rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.spatial_autocorr import run_spatial_autocorr

# ─── 路徑 ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
VIZ_DIR = OUTPUT_DIR / "viz" / "paper"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# 雙城市資料目錄
CITY_DIRS = {
    "Taichung": OUTPUT_DIR / "taichung",
    "Taipei":   OUTPUT_DIR / "taipei",
}

# ─── 字型設定（SimHei / Noto CJK 備選） ─────────────────────────────────────
def _setup_font():
    """嘗試載入中文字型，若不可用則使用 fallback。"""
    candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msjh.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            fp = font_manager.FontProperties(fname=p)
            rcParams["font.family"] = fp.get_name()
            rcParams["axes.unicode_minus"] = False
            return
    # fallback: DejaVu + minus fix
    rcParams["axes.unicode_minus"] = False

_setup_font()

# ─── 全局美化設定 ─────────────────────────────────────────────────────────────
PALETTE = {
    "blue":   "#2563EB",
    "orange": "#EA580C",
    "green":  "#16A34A",
    "purple": "#7C3AED",
    "red":    "#DC2626",
    "teal":   "#0D9488",
    "amber":  "#D97706",
    "gray":   "#6B7280",
}
CLUSTER_COLORS = ["#2563EB", "#EA580C", "#16A34A", "#7C3AED"]
CLUSTER_LABELS = ["Stable Core", "Rapid Expander", "Functional Diversifier", "Sparse/Rural"]

DPI = 300
FIGSIZE_WIDE  = (14, 7)
FIGSIZE_SQUARE = (10, 9)

# 各城市地理邊界與格線設定
CITY_BOUNDS = {
    "Taichung": {
        "lon": (120.615, 120.755),
        "lat": (24.085, 24.215),
        "bbox_lon": (120.610, 120.760),
        "bbox_lat": (24.080, 24.220),
        "grid_lon_range": (120.62, 120.76, 0.03),
        "grid_lat_range": (24.09, 24.22, 0.03),
    },
    "Taipei": {
        "lon": (121.450, 121.590),
        "lat": (24.985, 25.115),
        "bbox_lon": (121.445, 121.595),
        "bbox_lat": (24.980, 25.120),
        "grid_lon_range": (121.46, 121.60, 0.03),
        "grid_lat_range": (24.99, 25.12, 0.03),
    },
}


def _load_city_data(city_name: str):
    """載入指定城市的 panel_data 與 trajectory_clusters。"""
    city_dir = CITY_DIRS[city_name]
    panel_path = city_dir / "panel_data.csv"
    cluster_path = city_dir / "trajectory_clusters.csv"
    autocorr_path = city_dir / "spatial_autocorr.csv"

    panel = pd.read_csv(panel_path) if panel_path.exists() else None
    cluster_df = pd.read_csv(cluster_path) if cluster_path.exists() else None
    autocorr_df = pd.read_csv(autocorr_path) if autocorr_path.exists() else None

    return panel, cluster_df, autocorr_df


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — 研究區域地圖（雙城市並排）
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_study_area(city_data: dict):
    """
    繪製雙城市（Taichung / Taipei）各 195 個採樣點的空間分佈圖。
    左：台中；右：台北。以聚類著色，氣泡大小 = 平均建築數量。
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("#F8FAFC")

    for ax, (city_name, (panel, cluster_df, _)) in zip(axes, city_data.items()):
        ax.set_facecolor("#EFF6FF")
        bounds = CITY_BOUNDS[city_name]

        # 合併聚類標籤
        merged = panel.merge(cluster_df[["lon", "lat", "cluster"]], on=["lon", "lat"])
        pts = merged.drop_duplicates(subset=["lon", "lat"])

        # 年度建築數量（用於氣泡大小）
        bld_mean = panel.groupby(["lon", "lat"])["building_count"].mean().reset_index()
        bld_mean.columns = ["lon", "lat", "bld_mean"]
        pts = pts.merge(bld_mean, on=["lon", "lat"])

        # 城市邊界框（虛線）
        rect = mpatches.FancyBboxPatch(
            (bounds["bbox_lon"][0], bounds["bbox_lat"][0]),
            bounds["bbox_lon"][1] - bounds["bbox_lon"][0],
            bounds["bbox_lat"][1] - bounds["bbox_lat"][0],
            boxstyle="round,pad=0.002",
            linewidth=1.5, edgecolor="#94A3B8", facecolor="none",
            linestyle="--", zorder=1
        )
        ax.add_patch(rect)

        # 格線
        lon_start, lon_end, lon_step = bounds["grid_lon_range"]
        lat_start, lat_end, lat_step = bounds["grid_lat_range"]
        for lo in np.arange(lon_start, lon_end, lon_step):
            ax.axvline(lo, color="#CBD5E1", linewidth=0.5, zorder=1)
        for la in np.arange(lat_start, lat_end, lat_step):
            ax.axhline(la, color="#CBD5E1", linewidth=0.5, zorder=1)

        # 採樣點（按聚類著色，氣泡大小 = 平均建築密度）
        max_bld = pts["bld_mean"].max()
        size_scale = 1200 / max_bld if max_bld > 0 else 1
        for cid, color, label in zip(range(4), CLUSTER_COLORS, CLUSTER_LABELS):
            sub = pts[pts["cluster"] == cid]
            if len(sub) == 0:
                continue
            sizes = np.clip(sub["bld_mean"] * size_scale, 60, 400)
            ax.scatter(
                sub["lon"], sub["lat"],
                s=sizes, c=color, alpha=0.85, edgecolors="white",
                linewidths=1.2, zorder=5,
                label=f"C{cid}: {label} (n={len(sub)})"
            )

        # 高密度點標註（前 5 個最大建築量）
        for _, row in pts.sort_values("bld_mean", ascending=False).head(5).iterrows():
            ax.annotate(
                f"{row['bld_mean']:.0f}",
                (row["lon"], row["lat"]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=8, color="#374151", zorder=6
            )

        # 研究區域文字標注
        label_x = bounds["bbox_lon"][0] + 0.005
        label_y = bounds["bbox_lat"][1] - 0.008
        ax.text(label_x, label_y, f"{city_name} City\nStudy Area",
                fontsize=10, color="#1E3A5F", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="#94A3B8"))

        # 比例尺（~5km = 0.045 deg at lat ~24-25）
        scale_lon = bounds["bbox_lon"][0] + 0.015
        scale_lat = bounds["bbox_lat"][0] + 0.007
        ax.plot([scale_lon, scale_lon + 0.045], [scale_lat, scale_lat],
                "k-", linewidth=3, solid_capstyle="butt", zorder=6)
        ax.plot([scale_lon, scale_lon],
                [scale_lat - 0.003, scale_lat + 0.003], "k-", linewidth=2)
        ax.plot([scale_lon + 0.045, scale_lon + 0.045],
                [scale_lat - 0.003, scale_lat + 0.003], "k-", linewidth=2)
        ax.text(scale_lon + 0.0225, scale_lat - 0.008, "5 km",
                ha="center", fontsize=9, fontweight="bold", color="#111827")

        # 北方指示
        arrow_x = bounds["bbox_lon"][1] - 0.020
        arrow_y_base = bounds["bbox_lat"][1] - 0.020
        ax.annotate("", xy=(arrow_x, arrow_y_base + 0.014),
                    xytext=(arrow_x, arrow_y_base),
                    arrowprops=dict(arrowstyle="-|>", color="#1E3A5F", lw=2.5))
        ax.text(arrow_x + 0.0005, arrow_y_base + 0.016, "N",
                fontsize=12, fontweight="bold", color="#1E3A5F", ha="center")

        # 軸設定
        ax.set_xlim(bounds["bbox_lon"][0] - 0.003, bounds["bbox_lon"][1] + 0.003)
        ax.set_ylim(bounds["bbox_lat"][0] - 0.003, bounds["bbox_lat"][1] + 0.003)
        ax.set_xlabel("Longitude (\u00b0E)", fontsize=12, labelpad=6)
        ax.set_ylabel("Latitude (\u00b0N)", fontsize=12, labelpad=6)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f\u00b0"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f\u00b0"))
        ax.tick_params(labelsize=9)

        # 子圖標題
        subplot_label = "(a)" if city_name == "Taichung" else "(b)"
        ax.set_title(
            f"{subplot_label} {city_name} — 195 Grid Points",
            fontsize=13, pad=8, fontweight="bold"
        )

        # 圖例
        legend = ax.legend(loc="lower right", fontsize=9, framealpha=0.9,
                           edgecolor="#CBD5E1", ncol=1)
        legend.get_frame().set_facecolor("#F8FAFC")

    fig.suptitle(
        "Fig 1 \u2014 Study Area: Taichung and Taipei (195 Grid Points Each)\n"
        "(Coloured by development trajectory cluster; bubble size \u221d mean building count)",
        fontsize=14, y=1.02, fontweight="bold"
    )
    plt.tight_layout()
    out = VIZ_DIR / "fig1_study_area.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[Fig 1] Saved: {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Moran's I 年度變化折線圖（雙城市對比）
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_morans_i(city_data: dict):
    """
    繪製 Moran's I 年度變化折線圖，顯示關鍵指標在雙城市中的空間自相關趨勢。
    三指標：building_count, edge_density, road_length_total。
    實線 = Taichung，虛線 = Taipei。
    """
    target_indicators = ["building_count", "edge_density", "road_length_total"]
    label_map = {
        "building_count":    "Building Count",
        "edge_density":      "Edge Density",
        "road_length_total": "Road Length Total",
    }
    colors = [PALETTE["blue"], PALETTE["red"], PALETTE["teal"]]
    markers = ["o", "s", "^"]
    city_linestyles = {"Taichung": "-", "Taipei": "--"}

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    fig.patch.set_facecolor("white")
    ax.axhline(0, color="#9CA3AF", linewidth=0.8, linestyle="--")

    all_years = set()

    for city_name, (panel, _, autocorr_df) in city_data.items():
        ls = city_linestyles[city_name]

        # 使用預計算的 spatial_autocorr.csv 若存在，否則即時計算
        if autocorr_df is not None:
            mi_df = autocorr_df
        else:
            existing = [c for c in target_indicators if c in panel.columns]
            mi_df = run_spatial_autocorr(panel, indicators=existing, permutation=False)

        years = sorted(mi_df["year"].unique())
        all_years.update(years)

        for ind, col, mk in zip(target_indicators, colors, markers):
            sub = mi_df[mi_df["indicator"] == ind].sort_values("year")
            if sub.empty:
                continue
            I_vals = sub.set_index("year")["I"].reindex(years)
            p_vals = sub.set_index("year")["p_value"].reindex(years)

            ax.plot(
                years, I_vals, linestyle=ls, color=col, linewidth=2.2,
                markersize=7, marker=mk, zorder=4,
                label=f"{label_map.get(ind, ind)} ({city_name})"
            )
            ax.fill_between(years, I_vals - 0.03, I_vals + 0.03,
                            alpha=0.05, color=col)

            # 顯著性標記（* p<0.05, ** p<0.01）
            for yr, i_val, p_val in zip(years, I_vals, p_vals):
                if pd.isna(i_val) or pd.isna(p_val):
                    continue
                sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                if sig:
                    ax.text(yr, i_val + 0.018, sig, ha="center", va="bottom",
                            fontsize=9, color=col, fontweight="bold", zorder=5)

    years_sorted = sorted(all_years)
    ax.set_xlim(years_sorted[0] - 0.3, years_sorted[-1] + 0.3)
    ax.set_ylim(-0.05, 0.95)
    ax.set_xticks(years_sorted)
    ax.set_xticklabels([str(y) for y in years_sorted], fontsize=10)
    ax.set_yticks(np.arange(0, 1.0, 0.1))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Year", fontsize=12, labelpad=5)
    ax.set_ylabel("Moran's I", fontsize=12, labelpad=5)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # 城市線型說明附加到圖例
    city_legend = [
        Line2D([0], [0], color="gray", linewidth=2, linestyle="-",
               label="Taichung (solid)"),
        Line2D([0], [0], color="gray", linewidth=2, linestyle="--",
               label="Taipei (dashed)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + city_legend, fontsize=9, loc="lower left",
              framealpha=0.9, edgecolor="#E2E8F0", ncol=2)

    ax.set_title(
        "Fig 2 \u2014 Global Moran's I Spatial Autocorrelation by Year\n"
        "(*p<0.05, **p<0.01; inverse-distance weights; "
        "solid=Taichung, dashed=Taipei)",
        fontsize=13, fontweight="bold", pad=10
    )
    plt.tight_layout()
    out = VIZ_DIR / "fig2_morans_i.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"[Fig 2] Saved: {out}")
    plt.close()



# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═
# Fig 4 — PCA 碼石圖 + K-Means 驗證（雙城市，2×2 layout）
# ═

def _build_trajectory_matrix(panel):
    """把 panel 轉成寬格式軌跡矩陣並標準化，返回 X_scaled, wide。"""
    feature_cols = [
        "edge_density", "building_coverage", "texture_entropy",
        "building_count", "building_area_mean", "amenity_count",
        "shop_count", "leisure_count", "poi_diversity", "road_length_total",
    ]
    existing = [c for c in feature_cols if c in panel.columns]
    pivot_frames = []
    for col in existing:
        pv = panel.pivot_table(index=["lon", "lat"], columns="year", values=col)
        pv.columns = [f"{col}_{y}" for y in pv.columns]
        pivot_frames.append(pv)
    wide = pd.concat(pivot_frames, axis=1).fillna(0)
    X_scaled = StandardScaler().fit_transform(wide)
    return X_scaled, wide


def fig4_pca_scree(city_data: dict):
    """
    2x2 Fig 4:
      (a)(b) PCA scree plots (Taichung / Taipei)
      (c)    K-Means Elbow plot (WCSS, dual city)
      (d)    K-Means Silhouette plot (dual city)
    """
    K_RANGE = range(2, 9)
    city_colors = {"Taichung": PALETTE["blue"], "Taipei": PALETTE["orange"]}

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 2, hspace=0.42, wspace=0.40)

    # Pre-compute trajectory matrices
    traj_data = {}
    for city_name, (panel, _, _) in city_data.items():
        X_scaled, wide = _build_trajectory_matrix(panel)
        traj_data[city_name] = (X_scaled, wide)

    # (a)(b) PCA scree
    for col_idx, (city_name, (X_scaled, wide)) in enumerate(traj_data.items()):
        n_comp = min(wide.shape[0] - 1, wide.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(X_scaled)
        ev_ratio = pca.explained_variance_ratio_
        cum_var  = np.cumsum(ev_ratio)
        n_show   = min(20, n_comp)
        x_pos    = np.arange(1, n_show + 1)

        ax1 = fig.add_subplot(gs[0, col_idx])
        bar_colors = [PALETTE["blue"] if i < 5 else PALETTE["gray"]
                      for i in range(n_show)]
        ax1.bar(x_pos, ev_ratio[:n_show] * 100, color=bar_colors, alpha=0.85,
                width=0.6, edgecolor="white", linewidth=0.8, zorder=3)

        for i, (x, v) in enumerate(zip(x_pos[:6], ev_ratio[:6] * 100)):
            ax1.text(x, v + 0.5, f"{v:.1f}%", ha="center", va="bottom",
                     fontsize=9, color=PALETTE["blue"], fontweight="bold")

        ax2_twin = ax1.twinx()
        ax2_twin.plot(x_pos, cum_var[:n_show] * 100, "o-",
                      color=PALETTE["orange"], linewidth=2.2, markersize=5, zorder=4)
        ax2_twin.fill_between(x_pos, 0, cum_var[:n_show] * 100,
                              alpha=0.08, color=PALETTE["orange"])

        for thresh, thresh_col in [(80, "#059669"), (90, "#7C3AED")]:
            ax2_twin.axhline(thresh, color=thresh_col, linewidth=1.5,
                             linestyle="--", alpha=0.7, zorder=2)
            ax2_twin.text(n_show + 0.2, thresh, f"{thresh}%",
                          fontsize=10, color=thresh_col, va="center")
            n_thresh = int(np.searchsorted(cum_var, thresh / 100) + 1)
            if n_thresh <= n_show:
                ax1.axvline(n_thresh + 0.5, color=thresh_col, linewidth=1.2,
                            linestyle="--", alpha=0.5, zorder=2)
                ax1.text(n_thresh + 0.6, ev_ratio[0] * 95, f"d={n_thresh}",
                         fontsize=9, color=thresh_col, va="top", fontweight="bold")

        ax1.set_xlim(0.4, n_show + 0.6)
        ax1.set_ylim(0, ev_ratio[0] * 110)
        ax2_twin.set_ylim(0, 105)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"PC{i}" for i in x_pos], fontsize=9)
        ax1.set_xlabel("Principal Component", fontsize=11, labelpad=5)
        ax1.set_ylabel("Explained Variance (%)", fontsize=11,
                       color=PALETTE["blue"], labelpad=5)
        ax2_twin.set_ylabel("Cumulative Variance (%)", fontsize=11,
                            color=PALETTE["orange"], labelpad=5)
        ax1.tick_params(axis="y", colors=PALETTE["blue"])
        ax2_twin.tick_params(axis="y", colors=PALETTE["orange"])
        ax1.spines[["top", "right"]].set_visible(False)

        subplot_label = "(a)" if city_name == "Taichung" else "(b)"
        ax1.set_title(
            f"{subplot_label} {city_name} — PCA Scree "
            f"({wide.shape[0]} points × {wide.shape[1]} features)",
            fontsize=11, pad=8
        )

        if col_idx == 0:
            handles = [
                mpatches.Patch(color=PALETTE["blue"], alpha=0.85,
                               label="Individual variance (top PCs)"),
                mpatches.Patch(color=PALETTE["gray"], alpha=0.5,
                               label="Individual variance (remaining)"),
                Line2D([0], [0], color=PALETTE["orange"], linewidth=2.2,
                       marker="o", markersize=5, label="Cumulative variance"),
            ]
            ax1.legend(handles=handles, fontsize=9, loc="upper right",
                       framealpha=0.9, edgecolor="#E2E8F0")

    # (c) Elbow plot
    ax_elbow = fig.add_subplot(gs[1, 0])
    ax_elbow.set_facecolor("#F8FAFC")

    for city_name, (X_scaled, _) in traj_data.items():
        wcss = []
        for k in K_RANGE:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            wcss.append(km.inertia_)
        color = city_colors[city_name]
        ls = "-" if city_name == "Taichung" else "--"
        ax_elbow.plot(list(K_RANGE), wcss, marker="o", color=color,
                      linewidth=2.2, markersize=7, linestyle=ls,
                      label=city_name, zorder=4)
        idx4 = list(K_RANGE).index(4)
        ax_elbow.scatter([4], [wcss[idx4]], s=140, color=color,
                         edgecolors="white", linewidths=2, zorder=5)

    ax_elbow.axvline(4, color="#6B7280", linewidth=1.2, linestyle=":",
                     alpha=0.7, zorder=2)
    ymax = ax_elbow.get_ylim()[1]
    ax_elbow.text(4.1, ymax * 0.97, "K=4 selected",
                  fontsize=9, color="#6B7280", va="top")
    ax_elbow.set_xlabel("Number of Clusters (K)", fontsize=11, labelpad=5)
    ax_elbow.set_ylabel("Within-Cluster Sum of Squares", fontsize=11, labelpad=5)
    ax_elbow.set_xticks(list(K_RANGE))
    ax_elbow.spines[["top", "right"]].set_visible(False)
    ax_elbow.legend(fontsize=10, framealpha=0.9, edgecolor="#E2E8F0")
    ax_elbow.set_title("(c) K-Means Elbow Plot", fontsize=11, pad=8)

    # (d) Silhouette plot
    ax_sil = fig.add_subplot(gs[1, 1])
    ax_sil.set_facecolor("#F8FAFC")

    for city_name, (X_scaled, _) in traj_data.items():
        sil_scores = []
        for k in K_RANGE:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil_scores.append(silhouette_score(X_scaled, labels))
        color = city_colors[city_name]
        ls = "-" if city_name == "Taichung" else "--"
        ax_sil.plot(list(K_RANGE), sil_scores, marker="s", color=color,
                    linewidth=2.2, markersize=7, linestyle=ls,
                    label=city_name, zorder=4)
        idx4 = list(K_RANGE).index(4)
        ax_sil.scatter([4], [sil_scores[idx4]], s=140, color=color,
                       edgecolors="white", linewidths=2, zorder=5)
        ax_sil.text(4.12, sil_scores[idx4], f"s={sil_scores[idx4]:.3f}",
                    fontsize=8.5, color=color, va="center")

    ax_sil.axvline(4, color="#6B7280", linewidth=1.2, linestyle=":",
                   alpha=0.7, zorder=2)
    ax_sil.set_xlabel("Number of Clusters (K)", fontsize=11, labelpad=5)
    ax_sil.set_ylabel("Silhouette Score", fontsize=11, labelpad=5)
    ax_sil.set_xticks(list(K_RANGE))
    ax_sil.spines[["top", "right"]].set_visible(False)
    ax_sil.legend(fontsize=10, framealpha=0.9, edgecolor="#E2E8F0")
    ax_sil.set_title("(d) K-Means Silhouette Scores", fontsize=11, pad=8)

    fig.suptitle(
        "Fig 4 — PCA Dimensionality Reduction and K-Means Cluster Validation "
        "(Taichung vs Taipei)",
        fontsize=14, fontweight="bold", y=1.01
    )
    out = VIZ_DIR / "fig4_pca_scree.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"[Fig 4] Saved: {out}")
    plt.close()



# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """載入雙城市資料並依序生成所有論文圖表。"""
    city_data = {}
    for city_name in ["Taichung", "Taipei"]:
        panel, cluster_df, autocorr_df = _load_city_data(city_name)
        if panel is None:
            print(f"[ERROR] {city_name} panel_data.csv not found, "
                  f"please run features/build_features.py first.")
            return
        city_data[city_name] = (panel, cluster_df, autocorr_df)

    print(f"\nOutput directory: {VIZ_DIR}\n")
    print("=" * 55)

    # Fig 1 — Study area map (dual city)
    print("[1/3] Study area map (Taichung + Taipei)...")
    all_have_clusters = all(
        data[1] is not None for data in city_data.values()
    )
    if all_have_clusters:
        fig1_study_area(city_data)
    else:
        print("  [SKIP] Missing trajectory_clusters.csv for one or both cities")

    # Fig 2 — Moran's I trends (dual city)
    print("[2/3] Moran's I yearly trends (dual city)...")
    fig2_morans_i(city_data)

    # Fig 4 — PCA scree plot (dual city)
    print("[3/3] PCA scree plot (dual city)...")
    fig4_pca_scree(city_data)

    print("\n" + "=" * 55)
    print(f"All 3 figures saved to: {VIZ_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
