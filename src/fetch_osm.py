"""
OSM 多標籤多年份抓取模組
從 Overpass API 抓取台中市歷史 OSM 數據，計算結構化指標。

指標:
- building_count: 建築數量
- building_area_mean: 平均建築面積 (m²)
- amenity_count: 設施數量
- shop_count: 商店數量
- leisure_count: 休閒設施數量
- poi_diversity (Shannon entropy): POI 功能混合度
- road_length_total: 路網總長度 (m)
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Point
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OSM_DATA_DIR = PROJECT_ROOT / "osm_data"
CITY_NAME = "Taichung, Taiwan"
DEFAULT_CITY_KEY = "taichung"
BUFFER_DEG = 0.005  # ~500m


def get_osm_dir(city_key: str = DEFAULT_CITY_KEY) -> Path:
    """取得指定城市的 OSM 數據目錄，向下相容舊路徑"""
    return OSM_DATA_DIR / city_key


def fetch_osm_by_tags(city: str, year: int, tags: dict, label: str,
                      city_key: str = DEFAULT_CITY_KEY) -> gpd.GeoDataFrame | None:
    """
    從本地 osm_{year}.geojson 讀取數據並過濾指定的標籤。
    如果本地文件不存在，嘗試從快取文件讀取。
    """
    save_dir = get_osm_dir(city_key)
    # 1. 優先從本地 osm_{year}.geojson 讀取並過濾
    main_geojson_path = save_dir / f"osm_{year}.geojson"
    if main_geojson_path.exists() and main_geojson_path.stat().st_size > 0:
        try:
            full_gdf = gpd.read_file(main_geojson_path)
            # 根據 tags 過濾
            filtered_gdf = full_gdf.copy()
            for key, value in tags.items():
                if value is True:
                    # 標籤存在即可
                    filtered_gdf = filtered_gdf[filtered_gdf[key].notna()]
                else:
                    # 標籤等於指定值
                    filtered_gdf = filtered_gdf[filtered_gdf[key] == value]
            print(f"  [本地] {label}_{year}: 從 osm_{year}.geojson 讀取 {len(filtered_gdf)} 筆")
            return filtered_gdf
        except Exception as e:
            print(f"  [警告] 本地文件讀取失敗: {e}")

    # 2. 從快取文件讀取
    cache_path = save_dir / f"osm_{label}_{year}.geojson"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            gdf = gpd.read_file(cache_path)
            print(f"  [快取] {label}_{year}: {len(gdf)} 筆")
            return gdf
        except Exception:
            pass

    # 3. 如果本地沒有數據，返回 None
    print(f"  [警告] {label}_{year}: 本地數據不存在，跳過")
    return None


def fetch_road_network(city: str, year: int,
                       city_key: str = DEFAULT_CITY_KEY) -> gpd.GeoDataFrame | None:
    """
    從本地 osm_roads_{year}.geojson 讀取道路數據。
    優先讀獨立路網文件，其次嘗試從主 GeoJSON 過濾。
    """
    save_dir = get_osm_dir(city_key)

    # 1. 優先讀獨立路網文件
    roads_path = save_dir / f"osm_roads_{year}.geojson"
    if roads_path.exists() and roads_path.stat().st_size > 0:
        try:
            gdf = gpd.read_file(roads_path)
            print(f"  [本地] roads_{year}: {len(gdf)} 筆")
            return gdf
        except Exception as e:
            print(f"  [警告] 路網文件讀取失敗: {e}")

    # 2. 嘗試從主 GeoJSON 過濾 highway
    main_geojson_path = save_dir / f"osm_{year}.geojson"
    if main_geojson_path.exists() and main_geojson_path.stat().st_size > 0:
        try:
            full_gdf = gpd.read_file(main_geojson_path)
            if 'highway' in full_gdf.columns:
                roads_gdf = full_gdf[full_gdf['highway'].notna()]
                if len(roads_gdf) > 0:
                    print(f"  [過濾] roads_{year}: 從 osm_{year}.geojson 讀取 {len(roads_gdf)} 筆")
                    return roads_gdf
        except Exception as e:
            print(f"  [警告] 本地文件讀取失敗: {e}")

    # 3. 本地沒有數據
    print(f"  [警告] roads_{year}: 本地數據不存在，跳過")
    return None


def shannon_entropy(counts: list[int]) -> float:
    """計算 Shannon 多樣性指數。"""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log2(p) for p in probs)


def _query_nearby(gdf: gpd.GeoDataFrame | None, area) -> gpd.GeoDataFrame:
    """用空間索引快速查詢 buffer 範圍內的要素"""
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame()
    # 用 sindex 先粗篩 bounding box，再精確 intersects
    candidates_idx = list(gdf.sindex.intersection(area.bounds))
    if not candidates_idx:
        return gpd.GeoDataFrame()
    candidates = gdf.iloc[candidates_idx]
    return candidates[candidates.geometry.intersects(area)]


def compute_osm_metrics_for_point(
    lon: float, lat: float,
    buildings_gdf: gpd.GeoDataFrame | None,
    amenity_gdf: gpd.GeoDataFrame | None,
    shop_gdf: gpd.GeoDataFrame | None,
    leisure_gdf: gpd.GeoDataFrame | None,
    roads_gdf: gpd.GeoDataFrame | None,
) -> dict:
    """計算單一座標點周圍 500m 的 OSM 結構化指標。"""
    area = Point(lon, lat).buffer(BUFFER_DEG)

    result = {"lon": lon, "lat": lat}

    # 建築物
    nearby = _query_nearby(buildings_gdf, area)
    result["building_count"] = len(nearby)
    if len(nearby) > 0:
        try:
            projected = nearby.to_crs(epsg=3826)  # TWD97 / TM2
            result["building_area_mean"] = float(projected.geometry.area.mean())
        except Exception:
            result["building_area_mean"] = 0.0
    else:
        result["building_area_mean"] = 0.0

    # Amenity
    result["amenity_count"] = len(_query_nearby(amenity_gdf, area))

    # Shop
    result["shop_count"] = len(_query_nearby(shop_gdf, area))

    # Leisure
    result["leisure_count"] = len(_query_nearby(leisure_gdf, area))

    # POI 多樣性 (amenity + shop + leisure 類型的 Shannon entropy)
    type_counts = [result["amenity_count"], result["shop_count"], result["leisure_count"]]
    result["poi_diversity"] = shannon_entropy(type_counts)

    # 路網密度 (總長度公尺)
    nearby_r = _query_nearby(roads_gdf, area)
    if len(nearby_r) > 0:
        try:
            projected_r = nearby_r.to_crs(epsg=3826)
            result["road_length_total"] = float(projected_r.geometry.length.sum())
        except Exception:
            result["road_length_total"] = 0.0
    else:
        result["road_length_total"] = 0.0

    return result


def get_coord_list(raw_dir: Path, years: list[int], city_key: str = DEFAULT_CITY_KEY) -> list[tuple[float, float]]:
    """從 raw_images/{city_key}/ 取得所有不重複座標。"""
    coords = set()
    city_dir = raw_dir / city_key
    # 向下相容：如果城市子目錄不存在，嘗試舊版直接 raw_images/{year}/
    base = city_dir if city_dir.exists() else raw_dir
    for year in years:
        d = base / str(year)
        if not d.exists():
            continue
        for f in d.glob("*.png"):
            parts = f.stem.split("_")
            try:
                coords.add((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                pass
    return sorted(coords)


def build_osm_metrics(years: list[int], coords: list[tuple[float, float]],
                      city_key: str = DEFAULT_CITY_KEY, city_name: str = CITY_NAME) -> pd.DataFrame:
    """主流程：對每個年份讀取 OSM 數據，計算所有座標點的指標。"""
    all_rows = []

    for year in years:
        print(f"\n{'='*50}")
        print(f"處理 {city_key} {year} 年 OSM 數據...")
        print(f"{'='*50}")

        buildings = fetch_osm_by_tags(city_name, year, {"building": True}, "buildings", city_key)
        amenity = fetch_osm_by_tags(city_name, year, {"amenity": True}, "amenity", city_key)
        shop = fetch_osm_by_tags(city_name, year, {"shop": True}, "shop", city_key)
        leisure = fetch_osm_by_tags(city_name, year, {"leisure": True}, "leisure", city_key)
        roads = fetch_road_network(city_name, year, city_key)

        print(f"\n計算 {len(coords)} 個座標點的指標...")
        for i, (lon, lat) in enumerate(coords):
            row = compute_osm_metrics_for_point(lon, lat, buildings, amenity, shop, leisure, roads)
            row["year"] = year
            all_rows.append(row)
            if (i + 1) % 10 == 0:
                print(f"  進度: {i+1}/{len(coords)}")

    cols = ["year", "lon", "lat", "building_count", "building_area_mean",
            "amenity_count", "shop_count", "leisure_count", "poi_diversity", "road_length_total"]
    if len(all_rows) == 0:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(all_rows)
    return df[cols]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OSM 指標計算")
    parser.add_argument("city", nargs="?", default=DEFAULT_CITY_KEY, help="城市 key (預設 taichung)")
    args = parser.parse_args()

    city_key = args.city
    raw_dir = PROJECT_ROOT / "raw_images"
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    coords = get_coord_list(raw_dir, years, city_key)
    print(f"[{city_key}] 共 {len(coords)} 個不重複座標點")

    df = build_osm_metrics(years, coords, city_key)

    out_path = output_dir / f"osm_metrics_{city_key}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n已儲存: {out_path}  ({len(df)} 筆)")
    print("\n各年份指標平均:")
    print(df.groupby("year").mean(numeric_only=True).round(2))
    print("\n完成！")


if __name__ == "__main__":
    main()
