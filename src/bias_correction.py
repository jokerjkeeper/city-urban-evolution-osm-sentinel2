"""
OSM 志願者貢獻偏差修正模組
============================
OpenStreetMap 的數據量隨時間增長，部分源於城市實際建設，
部分源於 OSM 社群志願者持續數位化既有建築（回填效應）。

本模組提出志願者偏差修正指數：

    G_t_corr = (N_bld_t / N_bld_t0) × (C_total_t0 / C_total_t)^λ

其中：
- N_bld_t  : 第 t 年的建築數量（或其他指標）
- C_total_t: 第 t 年 OSM 快照的總元素數（建築+設施+商店+休閒+道路）
             作為社群貢獻量的代理變數
- t0       : 基準年（預設 2018）
- λ ∈ [0,1]: 修正強度（0=不修正, 1=完全比例修正, 建議 0.5）

輸出：
- output/osm_bias_corrected.csv  — 修正後的建築數量及增長指數
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# ─── 常數 ────────────────────────────────────────────────────────────────────

BASE_YEAR = 2018
DEFAULT_LAMBDA = 0.5   # 中度修正


# ─── 核心函數 ────────────────────────────────────────────────────────────────

def compute_total_elements(osm_metrics: pd.DataFrame) -> pd.Series:
    """
    計算每年 OSM 快照的「總元素數」（社群貢獻量代理）。
    使用各年份各點的建築+設施+商店+休閒數量加總，
    以全部採樣點的年度均值代表全域貢獻量。

    Returns
    -------
    pd.Series: index=year, value=mean_total_elements
    """
    count_cols = [c for c in ["building_count", "amenity_count", "shop_count", "leisure_count"]
                  if c in osm_metrics.columns]
    if not count_cols:
        raise ValueError("osm_metrics 缺少計數欄位")

    osm_metrics = osm_metrics.copy()
    osm_metrics["total_elements"] = osm_metrics[count_cols].sum(axis=1)
    return osm_metrics.groupby("year")["total_elements"].mean()


def bias_correction_index(
    series: pd.Series,
    community_proxy: pd.Series,
    base_year: int = BASE_YEAR,
    lambda_: float = DEFAULT_LAMBDA,
) -> pd.DataFrame:
    """
    計算單一指標的偏差修正增長指數。

    Parameters
    ----------
    series          : pd.Series，index=year，value=指標年度均值
    community_proxy : pd.Series，index=year，value=社群貢獻量代理（總元素數）
    base_year       : 基準年
    lambda_         : 修正強度 ∈ [0, 1]

    Returns
    -------
    DataFrame: year, raw_value, raw_growth_index, corrected_growth_index,
               community_proxy, correction_factor
    """
    if base_year not in series.index:
        raise ValueError(f"基準年 {base_year} 不在數據中")

    v0 = series[base_year]
    c0 = community_proxy[base_year]

    rows = []
    for year in sorted(series.index):
        v_t = series[year]
        c_t = community_proxy.get(year, c0)

        raw_growth = v_t / v0 if v0 > 0 else np.nan

        # 社群貢獻量修正因子 = (C_t0 / C_t)^λ
        if c0 > 0 and c_t > 0:
            correction_factor = (c0 / c_t) ** lambda_
        else:
            correction_factor = 1.0

        corrected_growth = raw_growth * correction_factor if not np.isnan(raw_growth) else np.nan

        rows.append({
            "year": year,
            "raw_value": round(v_t, 4),
            "raw_growth_index": round(raw_growth, 4) if not np.isnan(raw_growth) else np.nan,
            "correction_factor": round(correction_factor, 4),
            "corrected_growth_index": round(corrected_growth, 4) if not np.isnan(corrected_growth) else np.nan,
            "community_proxy": round(c_t, 2),
        })

    return pd.DataFrame(rows)


def sensitivity_analysis(
    series: pd.Series,
    community_proxy: pd.Series,
    base_year: int = BASE_YEAR,
    lambdas: list[float] | None = None,
) -> pd.DataFrame:
    """
    對不同 λ 值執行靈敏度分析，比較修正後的最終年增長指數。

    Returns
    -------
    DataFrame: lambda, final_year, corrected_growth_index, implied_real_growth_pct
    """
    if lambdas is None:
        lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]

    final_year = sorted(series.index)[-1]
    rows = []

    for lam in lambdas:
        df = bias_correction_index(series, community_proxy, base_year, lam)
        final_row = df[df["year"] == final_year].iloc[0]
        rows.append({
            "lambda": lam,
            "final_year": final_year,
            "raw_growth_index": final_row["raw_growth_index"],
            "corrected_growth_index": final_row["corrected_growth_index"],
            "implied_real_growth_pct": round(
                (final_row["corrected_growth_index"] - 1) * 100, 1
            ) if not np.isnan(final_row["corrected_growth_index"]) else np.nan,
        })

    return pd.DataFrame(rows)


def run_bias_correction(
    osm_metrics: pd.DataFrame,
    target_cols: list[str] | None = None,
    base_year: int = BASE_YEAR,
    lambda_: float = DEFAULT_LAMBDA,
) -> pd.DataFrame:
    """
    對多個 OSM 指標批次執行偏差修正，返回完整修正結果。

    Returns
    -------
    DataFrame: year, indicator, raw_value, corrected_growth_index,
               correction_factor, community_proxy
    """
    if target_cols is None:
        target_cols = ["building_count", "amenity_count", "shop_count", "poi_diversity"]

    existing_cols = [c for c in target_cols if c in osm_metrics.columns]
    if not existing_cols:
        raise ValueError("目標欄位不存在於 osm_metrics")

    community_proxy = compute_total_elements(osm_metrics)
    yearly_means = osm_metrics.groupby("year")[existing_cols].mean()

    all_results = []
    for col in existing_cols:
        series = yearly_means[col].dropna()
        if base_year not in series.index or len(series) < 2:
            continue
        df = bias_correction_index(series, community_proxy, base_year, lambda_)
        df.insert(0, "indicator", col)
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    osm_path = OUTPUT_DIR / "osm_metrics.csv"
    if not osm_path.exists():
        print("[ERROR] 找不到 osm_metrics.csv，請先執行 osm/fetch_osm.py")
        return

    osm = pd.read_csv(osm_path)
    print(f"OSM 數據: {osm.shape[0]} 筆，年份: {sorted(osm['year'].unique())}")

    # ─ 1. 社群貢獻量代理 ────────────────────────────────────────────────────
    community_proxy = compute_total_elements(osm)
    print("\n年度平均總元素數（社群貢獻量代理）：")
    for yr, val in community_proxy.items():
        print(f"  {yr}: {val:.1f}")

    # ─ 2. 建築數量偏差修正（核心指標） ──────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"建築數量偏差修正（λ={DEFAULT_LAMBDA}，基準年={BASE_YEAR}）")
    print("=" * 65)

    bld_series = osm.groupby("year")["building_count"].mean()
    result_df = bias_correction_index(bld_series, community_proxy, BASE_YEAR, DEFAULT_LAMBDA)

    print(f"\n  {'年份':>6}  {'原始均值':>10}  {'原始增長':>10}  {'修正因子':>10}  {'修正增長':>10}")
    print("  " + "─" * 55)
    for _, row in result_df.iterrows():
        raw_pct = f"{(row['raw_growth_index']-1)*100:+.1f}%" if pd.notna(row['raw_growth_index']) else "N/A"
        cor_pct = f"{(row['corrected_growth_index']-1)*100:+.1f}%" if pd.notna(row['corrected_growth_index']) else "N/A"
        print(f"  {int(row['year']):>6}  {row['raw_value']:>10.2f}  {raw_pct:>10}  "
              f"{row['correction_factor']:>10.4f}  {cor_pct:>10}")

    # ─ 3. 靈敏度分析 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("λ 靈敏度分析（建築數量，最終年相對基準年）")
    print("=" * 55)
    sens_df = sensitivity_analysis(bld_series, community_proxy, BASE_YEAR)
    print(f"\n  {'λ':>6}  {'原始增長%':>12}  {'修正後增長%':>12}")
    print("  " + "─" * 35)
    for _, row in sens_df.iterrows():
        raw_pct = f"{(row['raw_growth_index']-1)*100:.1f}%" if pd.notna(row['raw_growth_index']) else "N/A"
        cor_pct = f"{row['implied_real_growth_pct']:.1f}%" if pd.notna(row['implied_real_growth_pct']) else "N/A"
        print(f"  {row['lambda']:>6.2f}  {raw_pct:>12}  {cor_pct:>12}")

    # ─ 4. 批次修正所有指標 ───────────────────────────────────────────────────
    corrected_df = run_bias_correction(
        osm,
        target_cols=["building_count", "amenity_count", "shop_count", "poi_diversity"],
        base_year=BASE_YEAR,
        lambda_=DEFAULT_LAMBDA,
    )

    out_path = OUTPUT_DIR / "osm_bias_corrected.csv"
    corrected_df.to_csv(out_path, index=False)
    print(f"\n已儲存: {out_path}  ({len(corrected_df)} 筆)")

    # ─ 5. 靈敏度分析 CSV ─────────────────────────────────────────────────────
    sens_out = OUTPUT_DIR / "osm_bias_sensitivity.csv"
    sens_df.to_csv(sens_out, index=False)
    print(f"已儲存: {sens_out}")

    print("\nOSM 偏差修正分析完成！")


if __name__ == "__main__":
    main()
