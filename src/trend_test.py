"""
趨勢顯著性與結構突變點分析模組
================================
1. Mann–Kendall 非參數趨勢檢定
   - 不要求數據常態分佈
   - 適用於小樣本時序（n=8 年）
   - 支持 Sen's slope 估計（趨勢斜率）

2. Chow 結構突變點檢定
   - 對每個候選斷點執行兩段線性迴歸比較
   - 選取最顯著的斷點年份
   - 同時輸出 F 統計量與 p 值

輸出：
- output/trend_test.csv        — Mann–Kendall 各指標結果
- output/structural_break.csv  — Chow 各指標最佳斷點結果
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


# ─── Mann–Kendall ────────────────────────────────────────────────────────────

def mann_kendall(x: np.ndarray) -> dict:
    """
    Mann–Kendall 趨勢檢定（單一時序）。

    Parameters
    ----------
    x : 時序觀測值（按時間排列），長度 n ≥ 4

    Returns
    -------
    dict:
        S         – Mann–Kendall S 統計量（正=上升，負=下降）
        Var_S     – S 的方差（含連結校正）
        Z_MK      – 標準化 Z 分數
        p_value   – 雙尾 p 值
        trend     – "increasing" / "decreasing" / "no trend"
        Sen_slope – Sen's slope 斜率估計（每單位時間的中位數變化量）
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return {"S": np.nan, "Var_S": np.nan, "Z_MK": np.nan,
                "p_value": np.nan, "trend": "insufficient data", "Sen_slope": np.nan}

    # ─ S 統計量 ──────────────────────────────────────────────────────────────
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if diff > 0:
                S += 1
            elif diff < 0:
                S -= 1

    # ─ 方差（含連結校正） ────────────────────────────────────────────────────
    # 統計每個唯一值的出現次數（連結值）
    unique_vals, tie_counts = np.unique(x, return_counts=True)
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in tie_counts if t > 1)
    Var_S = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    # ─ Z 分數 ────────────────────────────────────────────────────────────────
    if Var_S <= 0:
        Z_MK = 0.0
    elif S > 0:
        Z_MK = (S - 1) / np.sqrt(Var_S)
    elif S < 0:
        Z_MK = (S + 1) / np.sqrt(Var_S)
    else:
        Z_MK = 0.0

    p_value = float(2 * (1 - stats.norm.cdf(abs(Z_MK))))

    # ─ 趨勢判斷 ──────────────────────────────────────────────────────────────
    alpha = 0.05
    if p_value < alpha and S > 0:
        trend = "increasing"
    elif p_value < alpha and S < 0:
        trend = "decreasing"
    else:
        trend = "no trend"

    # ─ Sen's slope ───────────────────────────────────────────────────────────
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if j != i:
                slopes.append((x[j] - x[i]) / (j - i))
    Sen_slope = float(np.median(slopes)) if slopes else np.nan

    return {
        "S": int(S),
        "Var_S": round(Var_S, 4),
        "Z_MK": round(Z_MK, 3),
        "p_value": round(p_value, 4),
        "trend": trend,
        "Sen_slope": round(Sen_slope, 6) if not np.isnan(Sen_slope) else np.nan,
    }


def run_mann_kendall(panel: pd.DataFrame, indicators: list[str] | None = None) -> pd.DataFrame:
    """
    對面板數據各指標計算年度橫截面均值的 Mann–Kendall 趨勢檢定。

    Parameters
    ----------
    panel      : 含 year 欄位的面板 DataFrame
    indicators : 要分析的指標欄位（None 則自動選取）

    Returns
    -------
    DataFrame: indicator, S, Z_MK, p_value, trend, Sen_slope, ...
    """
    if indicators is None:
        exclude = {"year", "lon", "lat"}
        indicators = [
            c for c in panel.select_dtypes(include="number").columns
            if c not in exclude and not c.startswith("d_")
        ]

    years = sorted(panel["year"].unique())
    results = []

    for ind in indicators:
        if ind not in panel.columns:
            continue
        # 每年的橫截面均值時序
        yearly_mean = (
            panel.groupby("year")[ind].mean().reindex(years).values
        )
        valid = ~np.isnan(yearly_mean)
        if valid.sum() < 4:
            continue

        mk = mann_kendall(yearly_mean[valid])
        results.append({
            "indicator": ind,
            "n_years": int(valid.sum()),
            "series_mean": round(float(np.nanmean(yearly_mean)), 6),
            "series_std": round(float(np.nanstd(yearly_mean)), 6),
            **mk,
        })

    return pd.DataFrame(results).sort_values("p_value")


# ─── Chow 結構突變點 ──────────────────────────────────────────────────────────

def chow_test(y: np.ndarray, tau: int) -> dict:
    """
    對時序 y 在位置 tau 執行 Chow 結構突變點檢定。

    Parameters
    ----------
    y   : 時序數據（長度 n）
    tau : 斷點索引（0-based），子序列分為 y[:tau] 和 y[tau:]

    Returns
    -------
    dict: F_stat, p_value, RSS_full, RSS1, RSS2, tau_index
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    t = np.arange(n, dtype=float)

    if tau < 2 or tau > n - 2:
        return {"F_stat": np.nan, "p_value": np.nan,
                "RSS_full": np.nan, "RSS1": np.nan, "RSS2": np.nan,
                "tau_index": tau}

    def ols_rss(t_sub, y_sub):
        X = np.column_stack([np.ones(len(t_sub)), t_sub])
        beta, res, _, _ = np.linalg.lstsq(X, y_sub, rcond=None)
        y_hat = X @ beta
        return float(((y_sub - y_hat) ** 2).sum())

    RSS_full = ols_rss(t, y)
    RSS1 = ols_rss(t[:tau], y[:tau])
    RSS2 = ols_rss(t[tau:], y[tau:])

    k = 2  # 參數數量（截距 + 斜率）
    numerator = (RSS_full - RSS1 - RSS2) / k
    denominator = (RSS1 + RSS2) / max(n - 2 * k, 1)

    if denominator <= 0:
        return {"F_stat": np.nan, "p_value": np.nan,
                "RSS_full": RSS_full, "RSS1": RSS1, "RSS2": RSS2, "tau_index": tau}

    F_stat = numerator / denominator
    p_value = float(1 - stats.f.cdf(F_stat, k, n - 2 * k))

    return {
        "F_stat": round(F_stat, 4),
        "p_value": round(p_value, 4),
        "RSS_full": round(RSS_full, 6),
        "RSS1": round(RSS1, 6),
        "RSS2": round(RSS2, 6),
        "tau_index": tau,
    }


def find_best_breakpoint(y: np.ndarray, years: list[int]) -> dict:
    """
    掃描所有候選斷點，找出使 Chow F 統計量最大的斷點年份。

    Parameters
    ----------
    y     : 時序數據（長度 = len(years)）
    years : 對應年份列表

    Returns
    -------
    dict: best_year, best_tau, F_stat, p_value, all_results
    """
    n = len(y)
    # 候選斷點：排除最前 2 和最後 2 個時間點（確保兩段都有足夠樣本）
    candidates = list(range(2, n - 1))

    best = {"F_stat": -np.inf, "p_value": 1.0, "tau_index": -1}
    all_results = []

    for tau in candidates:
        res = chow_test(y, tau)
        res["year_break"] = years[tau] if tau < len(years) else None
        all_results.append(res)

        if not np.isnan(res["F_stat"]) and res["F_stat"] > best["F_stat"]:
            best = res.copy()

    return {**best, "all_candidates": all_results}


def run_structural_break(
    panel: pd.DataFrame,
    indicators: list[str] | None = None,
) -> pd.DataFrame:
    """
    對面板數據各指標（年度均值時序）執行 Chow 結構突變點掃描。

    Returns
    -------
    DataFrame: indicator, best_break_year, F_stat, p_value, ...
    """
    if indicators is None:
        exclude = {"year", "lon", "lat"}
        indicators = [
            c for c in panel.select_dtypes(include="number").columns
            if c not in exclude and not c.startswith("d_")
        ]

    years = sorted(panel["year"].unique())
    results = []

    for ind in indicators:
        if ind not in panel.columns:
            continue
        yearly_mean = panel.groupby("year")[ind].mean().reindex(years).values
        valid_years = [y for y, v in zip(years, yearly_mean) if not np.isnan(v)]
        valid_vals = yearly_mean[~np.isnan(yearly_mean)]

        if len(valid_vals) < 5:
            continue

        best = find_best_breakpoint(valid_vals, valid_years)
        results.append({
            "indicator": ind,
            "n_years": len(valid_vals),
            "best_break_year": best.get("year_break"),
            "F_stat": best.get("F_stat"),
            "p_value": best.get("p_value"),
            "tau_index": best.get("tau_index"),
            "RSS_full": best.get("RSS_full"),
            "RSS1": best.get("RSS1"),
            "RSS2": best.get("RSS2"),
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("p_value")
    return df


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    panel_path = OUTPUT_DIR / "panel_data.csv"
    if not panel_path.exists():
        print("[ERROR] 找不到 panel_data.csv，請先執行 features/build_features.py")
        return

    panel = pd.read_csv(panel_path)
    print(f"面板數據: {panel.shape[0]} 筆")

    target_indicators = [
        "edge_density", "building_coverage", "texture_entropy",
        "building_count", "amenity_count", "shop_count",
        "poi_diversity", "building_area_mean",
        "ssim", "resnet_cosine_dist",
    ]
    existing = [c for c in target_indicators if c in panel.columns]

    # ── 1. Mann–Kendall ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Mann–Kendall 趨勢顯著性檢定")
    print("=" * 60)
    mk_df = run_mann_kendall(panel, indicators=existing)

    if not mk_df.empty:
        print(f"\n{'指標':<28} {'S':>6} {'Z_MK':>7} {'p值':>7} {'趨勢':<14} {'Sen斜率':>12}")
        print("─" * 80)
        for _, row in mk_df.iterrows():
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 \
                  else "*" if row["p_value"] < 0.05 else ""
            print(f"  {row['indicator']:<26} {int(row['S']):>6} {row['Z_MK']:>7.3f} "
                  f"{row['p_value']:>7.4f}{sig:<3} {row['trend']:<14} {row['Sen_slope']:>12.6f}")

        out_mk = OUTPUT_DIR / "trend_test.csv"
        mk_df.to_csv(out_mk, index=False)
        print(f"\n已儲存: {out_mk}")

    # ── 2. Chow 結構突變點 ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Chow 結構突變點檢定")
    print("=" * 60)
    chow_df = run_structural_break(panel, indicators=existing)

    if not chow_df.empty:
        print(f"\n{'指標':<28} {'最佳斷點年':>10} {'F統計量':>10} {'p值':>8}")
        print("─" * 60)
        for _, row in chow_df.iterrows():
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 \
                  else "*" if row["p_value"] < 0.05 else ""
            year_str = str(int(row["best_break_year"])) if pd.notna(row["best_break_year"]) else "N/A"
            print(f"  {row['indicator']:<26} {year_str:>10} {row['F_stat']:>10.3f} "
                  f"{row['p_value']:>8.4f}{sig}")

        out_chow = OUTPUT_DIR / "structural_break.csv"
        chow_df.to_csv(out_chow, index=False)
        print(f"\n已儲存: {out_chow}")

    print("\n趨勢分析完成！")


if __name__ == "__main__":
    main()
