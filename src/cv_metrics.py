"""
CV 量化指標模組
對衛星影像計算可量化的城市發展指標：
- 邊緣密度 (edge_density)
- 建築覆蓋率 (building_coverage)
- 紋理熵 (texture_entropy)
- SSIM 變化偵測 (跨年份)
- ResNet 餘弦距離 (跨年份)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import disk
from skimage.metrics import structural_similarity as ssim
import sys

# 加入專案根目錄以便 import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def calc_edge_density(gray: np.ndarray) -> float:
    """邊緣密度 = 邊緣像素數 / 總像素數，反映建築結構複雜度。"""
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.sum() / 255) / edges.size


def calc_building_coverage(gray: np.ndarray) -> float:
    """建築覆蓋率 = Otsu 二值化後白色像素占比，反映人造物面積比。"""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((binary == 255).sum()) / binary.size


def calc_texture_entropy(gray: np.ndarray, radius: int = 5) -> float:
    """紋理熵 (Shannon entropy)，反映區域複雜度/開發程度。"""
    # skimage entropy 需要 uint8
    img_u8 = (gray).astype(np.uint8)
    ent = sk_entropy(img_u8, disk(radius))
    return float(ent.mean())


def calc_ssim(img1_gray: np.ndarray, img2_gray: np.ndarray) -> float:
    """結構相似度 SSIM，值越低代表兩年份變化越大。"""
    # 統一尺寸
    h = min(img1_gray.shape[0], img2_gray.shape[0])
    w = min(img1_gray.shape[1], img2_gray.shape[1])
    a = cv2.resize(img1_gray, (w, h))
    b = cv2.resize(img2_gray, (w, h))
    score, _ = ssim(a, b, full=True)
    return float(score)


def calc_resnet_cosine_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """ResNet 特徵餘弦距離，值越大代表語義變化越大。"""
    cos_sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
    return float(1.0 - cos_sim)


def extract_single_image_metrics(img_path: Path) -> dict:
    """對單張圖片計算所有單幀指標。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 解析座標
    stem = img_path.stem
    parts = stem.split("_")
    try:
        lon, lat = float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return None

    return {
        "lon": lon,
        "lat": lat,
        "edge_density": calc_edge_density(gray),
        "building_coverage": calc_building_coverage(gray),
        "texture_entropy": calc_texture_entropy(gray),
    }


def build_cv_metrics(raw_dir: Path, years: list[int]) -> pd.DataFrame:
    """
    批次計算所有年份的 CV 指標，回傳面板 DataFrame。

    columns: year, lon, lat, edge_density, building_coverage, texture_entropy
    """
    rows = []
    for year in years:
        year_dir = raw_dir / str(year)
        if not year_dir.exists():
            print(f"[WARN] 目錄不存在，跳過: {year_dir}")
            continue

        png_files = sorted(year_dir.glob("*.png"))
        print(f"[{year}] 找到 {len(png_files)} 張圖片")

        for img_path in png_files:
            metrics = extract_single_image_metrics(img_path)
            if metrics:
                metrics["year"] = year
                rows.append(metrics)

    df = pd.DataFrame(rows)
    # 欄位排序
    cols = ["year", "lon", "lat", "edge_density", "building_coverage", "texture_entropy"]
    return df[cols]


def build_change_metrics(raw_dir: Path, year_pairs: list[tuple[int, int]]) -> pd.DataFrame:
    """
    計算跨年份變化指標 (SSIM + ResNet 餘弦距離)。

    year_pairs: [(2018, 2020), ...] — 前後兩年份
    回傳 DataFrame: year_from, year_to, lon, lat, ssim, resnet_cosine_dist
    """
    # 延遲載入 ResNet (較耗時)
    from src.aiutils import CityResNet
    resnet = CityResNet()

    rows = []
    for y1, y2 in year_pairs:
        dir1 = raw_dir / str(y1)
        dir2 = raw_dir / str(y2)
        if not dir1.exists() or not dir2.exists():
            print(f"[WARN] 缺少目錄: {dir1} 或 {dir2}")
            continue

        files1 = {f.name: f for f in dir1.glob("*.png")}
        files2 = {f.name: f for f in dir2.glob("*.png")}
        common = sorted(set(files1) & set(files2))
        print(f"[{y1}->{y2}] 共同圖片 {len(common)} 張")

        for fname in common:
            p1, p2 = files1[fname], files2[fname]
            img1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(p2), cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                continue

            # 解析座標
            parts = Path(fname).stem.split("_")
            try:
                lon, lat = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue

            ssim_val = calc_ssim(img1, img2)

            feat1 = resnet.get_vector(str(p1))
            feat2 = resnet.get_vector(str(p2))
            cos_dist = calc_resnet_cosine_distance(feat1, feat2) if (feat1 is not None and feat2 is not None) else None

            rows.append({
                "year_from": y1,
                "year_to": y2,
                "lon": lon,
                "lat": lat,
                "ssim": ssim_val,
                "resnet_cosine_dist": cos_dist,
            })

    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CV 指標計算")
    parser.add_argument("city", nargs="?", default="taichung", help="城市 key (預設 taichung)")
    args = parser.parse_args()

    city_key = args.city
    raw_dir = PROJECT_ROOT / "raw_images" / city_key
    # 向下相容：如果城市子目錄不存在，嘗試舊版路徑
    if not raw_dir.exists():
        raw_dir = PROJECT_ROOT / "raw_images"
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

    # 1. 單幀指標
    out_path = output_dir / f"cv_metrics_{city_key}.csv"
    if out_path.exists():
        print(f"[SKIP] 單幀 CV 指標已存在，略過計算: {out_path}")
        df_metrics = pd.read_csv(out_path)
    else:
        print("=" * 50)
        print(f"計算 {city_key} 單幀 CV 指標...")
        print("=" * 50)
        df_metrics = build_cv_metrics(raw_dir, years)
        df_metrics.to_csv(out_path, index=False)
        print(f"\n已儲存: {out_path}  ({len(df_metrics)} 筆)")
        print(df_metrics.groupby("year")[["edge_density", "building_coverage", "texture_entropy"]].mean())

    # 2. 跨年份變化指標
    out_path2 = output_dir / f"cv_change_metrics_{city_key}.csv"
    if out_path2.exists():
        print(f"[SKIP] 跨年份變化指標已存在，略過計算: {out_path2}")
    else:
        print("\n" + "=" * 50)
        print(f"計算 {city_key} 跨年份變化指標 (SSIM + ResNet)...")
        print("=" * 50)
        year_pairs = [(2018, 2019), (2019, 2020), (2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
        df_change = build_change_metrics(raw_dir, year_pairs)
        df_change.to_csv(out_path2, index=False)
        print(f"\n已儲存: {out_path2}  ({len(df_change)} 筆)")
        if len(df_change) > 0:
            print(df_change[["ssim", "resnet_cosine_dist"]].describe())

    print("\n完成！")


if __name__ == "__main__":
    main()
