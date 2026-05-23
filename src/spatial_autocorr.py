"""
空間自相關分析模組
==================
實現全局 Moran's I 空間自相關指數，用於檢驗城市發展是否具有空間聚集性。

方法：
- 空間權重矩陣：逆距離加權（距離閾值 ~1500m），行標準化
- 顯著性：隨機化假設下的常態近似 Z 檢定（B=999 置換檢定可選）
- 分析對象：osm_metrics.csv 與 panel_data.csv 中各年度各指標

輸出：
- output/spatial_autocorr.csv  — 各指標各年度的 I、Z、p 值
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# 距離閾值（度）：0.015° ≈ 1500m（採樣格網間距約 500m，此閾值涵蓋 1~3 階鄰居）
DEFAULT_THRESHOLD = 0.015


# ─── 空間權重矩陣 ────────────────────────────────────────────────────────────

def build_weight_matrix(
    lons: np.ndarray,
    lats: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    row_standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    構建空間權重矩陣 W（逆距離，距離帶截斷）。

    Parameters
    ----------
    lons, lats : 座標陣列，長度 n
    threshold  : 距離閾值（度）；超出閾值的點對權重設為 0
    row_standardize : 是否對 W 進行行標準化（使每行和為 1）

    Returns
    -------
    W    : (n, n) 空間權重矩陣
    dist : (n, n) 歐氏距離矩陣（對角線為 inf）
    """
    coords = np.column_stack([lons, lats])          # (n, 2)
    n = len(coords)

    # 廣播計算成對距離
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (n, n, 2)
    dist = np.sqrt((diff ** 2).sum(axis=2))                      # (n, n)
    np.fill_diagonal(dist, np.inf)

    # 距離帶逆距離權重
    with np.errstate(divide="ignore"):
        W = np.where(dist <= threshold, 1.0 / dist, 0.0)

    if row_standardize:
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0          # 孤立點不除零
        W = W / row_sums

    return W, dist


# ─── Moran's I 計算 ──────────────────────────────────────────────────────────

def morans_i(
    x: np.ndarray,
    W: np.ndarray,
    permutation: bool = False,
    n_perm: int = 999,
) -> dict:
    """
    計算全局 Moran's I。

    Parameters
    ----------
    x           : 觀測值陣列，長度 n
    W           : (n, n) 行標準化空間權重矩陣
    permutation : 是否使用置換檢定計算 p 值（較精確但較慢）
    n_perm      : 置換次數（permutation=True 時有效）

    Returns
    -------
    dict:
        I        – Moran's I 值（正=聚集，負=分散，~0=隨機）
        E_I      – 在 H₀ 下的期望值 = -1/(n-1)
        Var_I    – 在隨機化假設下的方差
        z_score  – 標準化 Z 分數
        p_value  – 雙尾 p 值（常態近似或置換）
        interpretation – 文字解釋
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    xbar = x.mean()
    z = x - xbar

    # ─ Moran's I ─────────────────────────────────────────────────────────────
    W_sum = W.sum()
    numerator = float((W * np.outer(z, z)).sum())
    denominator = float((z ** 2).sum())

    if denominator == 0 or W_sum == 0:
        return {"I": 0.0, "E_I": -1/(n-1), "Var_I": np.nan,
                "z_score": 0.0, "p_value": 1.0, "interpretation": "constant (no variation)"}

    I = (n / W_sum) * (numerator / denominator)

    # ─ 期望值 E[I] ────────────────────────────────────────────────────────────
    E_I = -1.0 / (n - 1)

    # ─ 方差（隨機化假設） ────────────────────────────────────────────────────
    S1 = 0.5 * ((W + W.T) ** 2).sum()
    S2 = ((W.sum(axis=1) + W.sum(axis=0)) ** 2).sum()

    m2 = float((z ** 2).mean())
    m4 = float((z ** 4).mean())
    b2 = m4 / (m2 ** 2) if m2 > 0 else 0.0

    A = n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * W_sum**2)
    B = b2 * ((n**2 - n) * S1 - 2*n * S2 + 6 * W_sum**2)
    C = (n - 1) * (n - 2) * (n - 3) * (W_sum**2)

    Var_I = (A - B) / C - E_I**2 if C != 0 else np.nan
    Var_I = max(Var_I, 1e-12) if not np.isnan(Var_I) else np.nan

    z_score = (I - E_I) / np.sqrt(Var_I) if Var_I and not np.isnan(Var_I) else 0.0

    # ─ p 值 ──────────────────────────────────────────────────────────────────
    if permutation:
        rng = np.random.default_rng(42)
        perm_I = np.zeros(n_perm)
        for i in range(n_perm):
            xp = rng.permutation(x)
            zp = xp - xp.mean()
            num_p = float((W * np.outer(zp, zp)).sum())
            den_p = float((zp ** 2).sum())
            perm_I[i] = (n / W_sum) * (num_p / den_p) if den_p > 0 else 0.0
        p_value = float(((perm_I >= I).sum() + (perm_I <= -I).sum() + 1) / (n_perm + 1))
    else:
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_score))))

    # ─ 文字解釋 ──────────────────────────────────────────────────────────────
    if p_value < 0.05:
        interp = "clustered (high-high / low-low)" if I > 0 else "dispersed"
    else:
        interp = "random (not significant)"

    return {
        "I": round(I, 4),
        "E_I": round(E_I, 4),
        "Var_I": round(Var_I, 6) if not np.isnan(Var_I) else np.nan,
        "z_score": round(z_score, 3),
        "p_value": round(p_value, 4),
        "interpretation": interp,
    }


# ─── 批次分析 ────────────────────────────────────────────────────────────────

def run_spatial_autocorr(
    panel: pd.DataFrame,
    indicators: list[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    permutation: bool = False,
) -> pd.DataFrame:
    """
    對 panel 數據中各指標、各年份計算 Moran's I。

    Parameters
    ----------
    panel      : 含 year, lon, lat 及各指標欄位的面板 DataFrame
    indicators : 要分析的指標欄位（None 則自動選取數值欄）
    threshold  : 空間權重距離閾值
    permutation: 是否使用置換檢定

    Returns
    -------
    DataFrame: year, indicator, n_points, I, E_I, z_score, p_value, interpretation
    """
    if indicators is None:
        exclude = {"year", "lon", "lat"}
        indicators = [
            c for c in panel.select_dtypes(include="number").columns
            if c not in exclude and not c.startswith("d_")
        ]

    years = sorted(panel["year"].unique())
    results = []

    for year in years:
        df_year = panel[panel["year"] == year].dropna(subset=["lon", "lat"]).copy()
        if len(df_year) < 4:
            continue

        lons = df_year["lon"].values
        lats = df_year["lat"].values
        W, _ = build_weight_matrix(lons, lats, threshold=threshold)

        for ind in indicators:
            if ind not in df_year.columns:
                continue
            vals = df_year[ind].dropna().values
            if len(vals) < 4 or vals.std() == 0:
                continue

            # 確保與 W 維度一致
            valid_mask = df_year[ind].notna().values
            if valid_mask.sum() != len(W):
                W_sub, _ = build_weight_matrix(
                    lons[valid_mask], lats[valid_mask], threshold=threshold
                )
            else:
                W_sub = W

            mi = morans_i(vals, W_sub, permutation=permutation)
            results.append({
                "year": year,
                "indicator": ind,
                "n_points": len(vals),
                **mi,
            })

    return pd.DataFrame(results)


# ─── 摘要輸出 ────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, top_indicators: list[str] | None = None):
    """列印關鍵指標的年度 Moran's I 摘要。"""
    if top_indicators is None:
        top_indicators = ["building_count", "edge_density", "texture_entropy", "poi_diversity"]

    for ind in top_indicators:
        sub = df[df["indicator"] == ind]
        if sub.empty:
            continue
        print(f"\n{'─'*55}")
        print(f"  {ind}")
        print(f"{'─'*55}")
        print(f"  {'Year':>6}  {'I':>7}  {'Z':>7}  {'p':>7}  {'解釋'}")
        for _, row in sub.sort_values("year").iterrows():
            sig = "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            print(f"  {int(row['year']):>6}  {row['I']:>7.4f}  {row['z_score']:>7.3f}"
                  f"  {row['p_value']:>7.4f}{sig:2s}  {row['interpretation']}")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="空間自相關分析")
    parser.add_argument("city", nargs="?", default="taichung", help="城市 key (預設 taichung)")
    args = parser.parse_args()
    city_key = args.city

    panel_path = OUTPUT_DIR / f"panel_data_{city_key}.csv"
    if not panel_path.exists():
        print(f"[ERROR] 找不到 {panel_path.name}，請先執行 src/build_features.py")
        return

    panel = pd.read_csv(panel_path)
    print(f"面板數據: {panel.shape[0]} 筆")

    # 分析主要指標（單幀指標＋OSM 結構指標）
    target_indicators = [
        "edge_density", "building_coverage", "texture_entropy",
        "building_count", "amenity_count", "shop_count", "poi_diversity",
    ]
    existing = [c for c in target_indicators if c in panel.columns]

    print("\n計算 Moran's I（逆距離空間權重，閾值 0.015°）...")
    result_df = run_spatial_autocorr(panel, indicators=existing, permutation=False)

    if result_df.empty:
        print("[WARN] 無法計算空間自相關（數據不足）")
        return

    # 輸出 CSV
    out_path = OUTPUT_DIR / f"spatial_autocorr_{city_key}.csv"
    result_df.to_csv(out_path, index=False)
    print(f"已儲存: {out_path}  ({len(result_df)} 筆)")

    # 控制台摘要
    print("\n" + "=" * 55)
    print("全局 Moran's I 結果摘要")
    print("=" * 55)
    print_summary(result_df, top_indicators=existing)

    # 空間聚集趨勢（Moran's I 年度均值）
    print("\n\n年度平均 Moran's I（所有指標）")
    yearly_avg = result_df.groupby("year")[["I", "z_score"]].mean().round(4)
    print(yearly_avg.to_string())

    print("\n空間自相關分析完成！")


if __name__ == "__main__":
    main()
