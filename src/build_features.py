"""
面板數據整合模組
合併 CV 指標 + OSM 指標 為統一的面板 DataFrame。
結構: 44 locations x N years = panel data
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def build_panel_from_dir(data_dir: Path = None) -> pd.DataFrame:
    """從指定目錄讀取 CSV 並合併為面板數據（支持多城市）"""
    d = data_dir or OUTPUT_DIR
    return build_panel(d / "cv_metrics.csv", d / "osm_metrics.csv", d / "cv_change_metrics.csv")


def build_panel(cv_path: Path = None, osm_path: Path = None, change_path: Path = None) -> pd.DataFrame:
    """
    讀取 cv_metrics.csv + osm_metrics.csv + cv_change_metrics.csv，
    合併為統一面板數據。
    """
    cv_path = cv_path or OUTPUT_DIR / "cv_metrics.csv"
    osm_path = osm_path or OUTPUT_DIR / "osm_metrics.csv"
    change_path = change_path or OUTPUT_DIR / "cv_change_metrics.csv"

    df_cv = pd.read_csv(cv_path)
    df_osm = pd.read_csv(osm_path)

    # 合併 CV + OSM (on year, lon, lat)
    # 使用近似匹配 (round 到 5 位) 避免浮點誤差
    for df in [df_cv, df_osm]:
        df["lon_r"] = df["lon"].round(5)
        df["lat_r"] = df["lat"].round(5)

    panel = pd.merge(
        df_cv, df_osm,
        on=["year", "lon_r", "lat_r"],
        suffixes=("_cv", "_osm"),
        how="inner",
    )

    # 清理重複欄位
    panel["lon"] = panel["lon_cv"]
    panel["lat"] = panel["lat_cv"]
    panel.drop(columns=["lon_cv", "lat_cv", "lon_osm", "lat_osm", "lon_r", "lat_r"], inplace=True)

    # 合併 change metrics (只有 year_from->year_to 的行)
    if change_path.exists():
        df_change = pd.read_csv(change_path)
        df_change["lon_r"] = df_change["lon"].round(5)
        df_change["lat_r"] = df_change["lat"].round(5)

        panel["lon_r"] = panel["lon"].round(5)
        panel["lat_r"] = panel["lat"].round(5)

        # 把 change metrics 掛到 year_to 那一年的列
        change_cols = df_change.rename(columns={"year_to": "year"})[["year", "lon_r", "lat_r", "ssim", "resnet_cosine_dist"]]
        panel = pd.merge(panel, change_cols, on=["year", "lon_r", "lat_r"], how="left")
        panel.drop(columns=["lon_r", "lat_r"], inplace=True)

    # 排序
    cols_order = [
        "year", "lon", "lat",
        # CV
        "edge_density", "building_coverage", "texture_entropy",
        # Change
        "ssim", "resnet_cosine_dist",
        # OSM
        "building_count", "building_area_mean",
        "amenity_count", "shop_count", "leisure_count",
        "poi_diversity", "road_length_total",
    ]
    existing = [c for c in cols_order if c in panel.columns]
    panel = panel[existing].sort_values(["lon", "lat", "year"]).reset_index(drop=True)

    return panel


def compute_delta_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    計算每個地點各指標的「年度變化量」(Δ)。
    """
    numeric_cols = [
        "edge_density", "building_coverage", "texture_entropy",
        "building_count", "building_area_mean",
        "amenity_count", "shop_count", "leisure_count",
        "poi_diversity", "road_length_total",
    ]
    existing = [c for c in numeric_cols if c in panel.columns]

    panel_sorted = panel.sort_values(["lon", "lat", "year"])
    deltas = panel_sorted.groupby(["lon", "lat"])[existing].diff()
    deltas.columns = [f"d_{c}" for c in deltas.columns]

    result = pd.concat([panel_sorted.reset_index(drop=True), deltas.reset_index(drop=True)], axis=1)
    return result


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("建構面板數據...")
    panel = build_panel()
    print(f"面板形狀: {panel.shape}")
    print(f"欄位: {panel.columns.tolist()}")
    print(panel.head(5))

    print("\n計算變化量 (delta)...")
    panel_delta = compute_delta_features(panel)

    out_path = OUTPUT_DIR / "panel_data.csv"
    panel_delta.to_csv(out_path, index=False)
    print(f"\n已儲存: {out_path}  ({len(panel_delta)} 筆)")

    # 摘要
    print("\n各年份指標平均:")
    numeric = panel_delta.select_dtypes(include="number").columns.tolist()
    numeric = [c for c in numeric if c not in ["lon", "lat", "year"]]
    print(panel_delta.groupby("year")[numeric].mean().round(4).T)

    print("\nPhase 3a (面板數據整合) 完成！")


if __name__ == "__main__":
    main()
