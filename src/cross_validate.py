"""
CV ↔ OSM 交叉驗證模組
驗證兩種數據源的一致性：
- CV 邊緣密度 vs OSM 建築數量 (相關性)
- SSIM vs OSM 新增 POI
- 不一致區域標記
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def correlation_analysis(panel: pd.DataFrame) -> pd.DataFrame:
    """計算 CV 指標與 OSM 指標之間的 Pearson 相關係數。"""
    cv_cols = ["edge_density", "building_coverage", "texture_entropy"]
    osm_cols = ["building_count", "building_area_mean", "amenity_count",
                "shop_count", "leisure_count", "poi_diversity", "road_length_total"]

    existing_cv = [c for c in cv_cols if c in panel.columns]
    existing_osm = [c for c in osm_cols if c in panel.columns]

    results = []
    for cv in existing_cv:
        for osm in existing_osm:
            df = panel[[cv, osm]].dropna()
            if len(df) < 5:
                continue
            r, p = stats.pearsonr(df[cv], df[osm])
            results.append({
                "cv_metric": cv,
                "osm_metric": osm,
                "pearson_r": round(r, 4),
                "p_value": round(p, 4),
                "significant": "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "",
            })

    return pd.DataFrame(results).sort_values("pearson_r", ascending=False, key=abs)


def delta_consistency(panel: pd.DataFrame) -> pd.DataFrame:
    """
    檢查變化方向一致性：
    CV 指標增加的地方，OSM 指標是否也增加？
    """
    pairs = [
        ("d_edge_density", "d_building_count"),
        ("d_building_coverage", "d_building_count"),
        ("d_texture_entropy", "d_amenity_count"),
        ("d_edge_density", "d_road_length_total"),
    ]

    results = []
    for cv_delta, osm_delta in pairs:
        if cv_delta not in panel.columns or osm_delta not in panel.columns:
            continue
        df = panel[[cv_delta, osm_delta, "lon", "lat"]].dropna()
        if len(df) < 3:
            continue

        # 同向變化比例
        same_dir = ((df[cv_delta] > 0) & (df[osm_delta] > 0)) | ((df[cv_delta] < 0) & (df[osm_delta] < 0))
        consistency_rate = same_dir.mean()

        r, p = stats.pearsonr(df[cv_delta], df[osm_delta])

        results.append({
            "cv_delta": cv_delta,
            "osm_delta": osm_delta,
            "consistency_rate": round(consistency_rate, 3),
            "pearson_r": round(r, 4),
            "p_value": round(p, 4),
        })

    return pd.DataFrame(results)


def flag_inconsistent_locations(panel: pd.DataFrame) -> pd.DataFrame:
    """
    標記 CV 和 OSM 不一致的地點（可能是 OSM 數據滯後或影像問題）。
    規則：CV 邊緣密度增加但 OSM 建築數量減少，或反之。
    """
    d_cv = "d_edge_density"
    d_osm = "d_building_count"

    if d_cv not in panel.columns or d_osm not in panel.columns:
        return pd.DataFrame()

    df = panel.dropna(subset=[d_cv, d_osm]).copy()
    df["inconsistent"] = (
        ((df[d_cv] > 0.005) & (df[d_osm] < -5)) |
        ((df[d_cv] < -0.005) & (df[d_osm] > 5))
    )

    flagged = df[df["inconsistent"]][["year", "lon", "lat", d_cv, d_osm]]
    return flagged


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OSM-CV 交叉驗證")
    parser.add_argument("city", nargs="?", default="taichung", help="城市 key (預設 taichung)")
    args = parser.parse_args()
    city_key = args.city

    panel_path = OUTPUT_DIR / f"panel_data_{city_key}.csv"
    if not panel_path.exists():
        print(f"找不到 {panel_path.name}，請先執行 src/build_features.py")
        return

    panel = pd.read_csv(panel_path)

    # 1. 相關性分析
    print("=" * 60)
    print("CV ↔ OSM 相關性分析")
    print("=" * 60)
    corr_df = correlation_analysis(panel)
    print(corr_df.to_string(index=False))
    corr_df.to_csv(OUTPUT_DIR / "cv_osm_correlation.csv", index=False)

    # 2. 變化方向一致性
    print("\n" + "=" * 60)
    print("變化方向一致性檢驗")
    print("=" * 60)
    consist_df = delta_consistency(panel)
    if len(consist_df) > 0:
        print(consist_df.to_string(index=False))
        consist_df.to_csv(OUTPUT_DIR / f"delta_consistency_{city_key}.csv", index=False)
    else:
        print("(delta 數據不足，僅兩個年份時第一年無 delta)")

    # 3. 不一致地點
    print("\n" + "=" * 60)
    print("不一致地點標記")
    print("=" * 60)
    flagged = flag_inconsistent_locations(panel)
    if len(flagged) > 0:
        print(f"共 {len(flagged)} 個不一致地點:")
        print(flagged.to_string(index=False))
    else:
        print("未發現明顯不一致的地點")

    print("\nPhase 3c (交叉驗證) 完成！")


if __name__ == "__main__":
    main()
