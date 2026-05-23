# city-urban-evolution-osm-sentinel2

Replication code for the paper:

> **Comparative Urban Evolution in Two Taiwanese Cities: A Multi-City OSM–Sentinel-2
> Panel Analysis of Taichung and Taipei (2018–2025)**
> Chung-Ming Chen — submitted to *Cities* (Elsevier), 2026

---

## Overview

This repository contains the analysis and figure-generation code used in the paper.
Raw satellite imagery and OSM GeoJSON files are **not included** due to size constraints
(~15 GB total). They are available for download via Google Drive:

**[Download raw data (Google Drive)](https://drive.google.com/drive/folders/1Y_TWLWPU5XBfPrSbe8zrxPgSeEYOJITR?usp=sharing)**

```
paper/
├── raw_images/     # Sentinel-2 GeoTIFF tiles (per city per year)
└── osm_data/       # OSM GeoJSON snapshots (per city per year)
```

Place the downloaded `paper/` folder in the repository root before running the pipeline.

---

## Repository Structure

```
├── src/                    # Core analysis modules
│   ├── fetch_osm.py        # OSM data loading and metric computation
│   ├── bias_correction.py  # OSM volunteer-bias correction index
│   ├── cv_metrics.py       # Computer vision feature extraction
│   ├── build_features.py   # Panel dataset construction
│   ├── spatial_autocorr.py # Moran's I spatial autocorrelation
│   ├── trend_test.py       # Mann-Kendall trend test + Chow break detection
│   └── cross_validate.py   # OSM-CV cross-validation (Pearson)
│
├── figures/                # Figure generation scripts
│   ├── paper_figures.py    # Fig 1 (study area), Fig 2 (Moran's I), Fig 4 (PCA)
│   └── fig3_structural_break.py   # Fig 3 (structural break timeline)
│
├── data/                   # Reference data (small files only)
│   └── image_dates.csv     # Satellite image acquisition dates per city/year
│
└── docs/
    └── overpass_queries.md # Exact Overpass API queries used for OSM download
```

---

## Requirements

```
python >= 3.10
osmnx >= 1.6
geopandas >= 0.14
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
matplotlib >= 3.7
scipy >= 1.11
pymannkendall >= 1.4
```

Install:
```bash
pip install osmnx geopandas pandas numpy scikit-learn matplotlib scipy pymannkendall
```

---

## Data Preparation Pipeline

Run the following steps in order. Each step reads from `paper/` and writes
intermediate results to `output/`. Download the raw data from Google Drive first
(see Overview).

```bash
# Step 1 — Fetch OSM data and compute per-point metrics
#   Reads:  paper/osm_data/osm_{year}.geojson
#   Writes: output/{city}/osm_metrics_{year}.csv
python src/fetch_osm.py

# Step 2 — Extract computer vision features from Sentinel-2 imagery
#   Reads:  paper/raw_images/{city}_{year}.tif
#   Writes: output/{city}/cv_metrics_{year}.csv
python src/cv_metrics.py

# Step 3 — Apply OSM volunteer-bias correction (lambda = 0.5)
#   Reads:  output/{city}/osm_metrics_*.csv
#   Writes: output/{city}/osm_corrected.csv
python src/bias_correction.py

# Step 4 — Assemble the full panel dataset
#   Reads:  output/{city}/osm_corrected.csv + cv_metrics_*.csv
#   Writes: output/{city}/panel_data.csv
python src/build_features.py

# Step 5 — Spatial autocorrelation (Moran's I, inverse-distance weights)
#   Reads:  output/{city}/panel_data.csv
#   Writes: output/{city}/morans_i.csv
python src/spatial_autocorr.py

# Step 6 — Mann-Kendall trend test + Chow structural break detection
#   Reads:  output/{city}/panel_data.csv
#   Writes: output/{city}/trend_results.csv
python src/trend_test.py

# Step 7 — OSM–CV cross-validation (Pearson correlation)
#   Reads:  output/{city}/panel_data.csv
#   Writes: output/{city}/cross_val_results.csv
python src/cross_validate.py
```

**fetch osm**

`python src/fetch_osm.py`

```
==================================================
處理 taichung 2025 年 OSM 數據...
==================================================
  [本地] buildings_2025: 從 osm_2025.geojson 讀取 27826 筆
  [本地] amenity_2025: 從 osm_2025.geojson 讀取 1021 筆
  [本地] shop_2025: 從 osm_2025.geojson 讀取 378 筆
  [本地] leisure_2025: 從 osm_2025.geojson 讀取 40 筆
  [過濾] roads_2025: 從 osm_2025.geojson 讀取 1 筆

計算 195 個座標點的指標...
  進度: 10/195
  進度: 20/195
  進度: 30/195
  進度: 40/195
  進度: 50/195
  進度: 60/195
  進度: 70/195
  進度: 80/195
  進度: 90/195
  進度: 100/195
  進度: 110/195
  進度: 120/195
  進度: 130/195
  進度: 140/195
  進度: 150/195
  進度: 160/195
  進度: 170/195
  進度: 180/195
  進度: 190/195

已儲存: J:\git\city-urban-evolution-osm-sentinel2\output\osm_metrics_taichung.csv  (1560 筆)

各年份指標平均:
         lon    lat  building_count  building_area_mean  ...  shop_count  leisure_count  poi_diversity  road_length_total
year                                                     ...
2018  120.68  24.14           21.77             1327.23  ...        0.83           0.09           0.27               0.00
2019  120.68  24.14           18.76             1531.54  ...        0.97           0.09           0.28               0.00
2020  120.68  24.14           25.14             1394.51  ...        1.07           0.11           0.29               0.00
2021  120.68  24.14           30.37             1297.69  ...        1.18           0.14           0.31               0.00
2022  120.68  24.14           46.35             1100.24  ...        1.33           0.14           0.32               0.62
2023  120.68  24.14           86.57              980.48  ...        1.42           0.17           0.37               0.08
2024  120.68  24.14          106.08              873.73  ...        1.52           0.18           0.39               0.08
2025  120.68  24.14          113.71              778.93  ...        1.53           0.18           0.38               0.08

[8 rows x 9 columns]

完成！
```

---

**build metrics**

`python src/cv_metrics.py`

```
  warnings.warn(msg)
[2018->2019] 共同圖片 195 張
[2019->2020] 共同圖片 195 張
[2020->2021] 共同圖片 195 張
[2021->2022] 共同圖片 195 張
[2022->2023] 共同圖片 195 張
[2023->2024] 共同圖片 195 張
[2024->2025] 共同圖片 195 張

已儲存: J:\git\city-urban-evolution-osm-sentinel2\output\cv_change_metrics_taichung.csv  (1365 筆)
              ssim  resnet_cosine_dist
count  1365.000000         1365.000000
mean      0.385668            0.389719
std       0.343464            0.327077
min       0.000843            0.010009
25%       0.004686            0.064845
50%       0.505739            0.250665
75%       0.691621            0.757088
max       0.907491            0.823882

完成！
```

---

**bias correction**

`python src/bias_correction.py `

```
(py3.10) J:\git\city-urban-evolution-osm-sentinel2>python src/bias_correction.py 
OSM 數據: 1560 筆，年份: [np.int64(2018), np.int64(2019), np.int64(2020), np.int64(2021), np.int64(2022), np.int64(2023), np.int64(2024), np.int64(2025)]

年度平均總元素數（社群貢獻量代理）：
  2018: 24.8
  2019: 22.5
  2020: 29.6
  2021: 35.3
  2022: 51.5
  2023: 92.2
  2024: 111.9
  2025: 119.7

=================================================================
建築數量偏差修正（λ=0.5，基準年=2018）
=================================================================

      年份        原始均值        原始增長        修正因子        修正增長
  ───────────────────────────────────────────────────────
    2018       21.77       +0.0%      1.0000       +0.0%
    2019       18.76      -13.8%      1.0513       -9.4%
    2020       25.14      +15.5%      0.9156       +5.7%
    2021       30.37      +39.5%      0.8385      +16.9%
    2022       46.35     +112.9%      0.6940      +47.7%
    2023       86.57     +297.6%      0.5189     +106.3%
    2024      106.08     +387.2%      0.4709     +129.4%
    2025      113.71     +422.2%      0.4554     +137.8%

=======================================================
λ 靈敏度分析（建築數量，最終年相對基準年）
=======================================================

       λ         原始增長%        修正後增長%
  ───────────────────────────────────
    0.00        422.2%        422.2%
    0.25        422.2%        252.4%
    0.50        422.2%        137.8%
    0.75        422.2%         60.5%
    1.00        422.2%          8.3%

已儲存: J:\git\city-urban-evolution-osm-sentinel2\output\osm_bias_corrected_taichung.csv  (32 筆)
已儲存: J:\git\city-urban-evolution-osm-sentinel2\output\osm_bias_sensitivity_taichung.csv

OSM 偏差修正分析完成！
```

---

**build features**

`python src/build_features.py`

```
(py3.10) J:\git\city-urban-evolution-osm-sentinel2>python src/build_features.py
建構面板數據...
面板形狀: (1560, 15)
欄位: ['year', 'lon', 'lat', 'edge_density', 'building_coverage', 'texture_entropy', 'ssim', 'resnet_cosine_dist', 'building_count', 'building_area_mean', 'amenity_count', 'shop_count', 'leisure_count', 'poi_diversity', 'road_length_total']  
   year      lon     lat  edge_density  ...  shop_count  leisure_count  poi_diversity  road_length_total
0  2018  120.615  24.085      0.000000  ...           0              0            0.0                0.0
1  2019  120.615  24.085      0.067657  ...           0              0            0.0                0.0
2  2020  120.615  24.085      0.070672  ...           0              0            0.0                0.0
3  2021  120.615  24.085      0.095557  ...           0              0            0.0                0.0
4  2022  120.615  24.085      0.000000  ...           0              0            0.0                0.0

[5 rows x 15 columns]

計算變化量 (delta)...

已儲存: J:\git\city-urban-evolution-osm-sentinel2\output\panel_data_taichung.csv  (1560 筆)

各年份指標平均:
year                       2018       2019       2020       2021       2022      2023      2024      2025
edge_density             0.0000     0.0954     0.1157     0.1456     0.0000    0.1643    0.1567    0.1637
building_coverage        0.0000     0.2163     0.2276     0.2483     0.0000    0.2502    0.2438    0.2453
texture_entropy          0.0000     2.9814     3.0434     3.1864     0.0000    3.1907    3.1315    3.1537
ssim                        NaN     0.0049     0.6767     0.8001     0.0031    0.0079    0.5349    0.6720
resnet_cosine_dist          NaN     0.7478     0.1516     0.1345     0.7650    0.7612    0.0788    0.0891
building_count          21.7744    18.7641    25.1385    30.3692    46.3538   86.5744  106.0769  113.7077
building_area_mean    1327.2297  1531.5411  1394.5058  1297.6889  1100.2360  980.4794  873.7340  778.9269
amenity_count            2.1231     2.6256     3.2821     3.6051     3.7077    4.0051    4.1179    4.2256
shop_count               0.8256     0.9692     1.0718     1.1846     1.3282    1.4154    1.5231    1.5333
leisure_count            0.0923     0.0923     0.1077     0.1385     0.1385    0.1744    0.1846    0.1846
poi_diversity            0.2666     0.2845     0.2875     0.3111     0.3202    0.3680    0.3850    0.3825
road_length_total        0.0000     0.0000     0.0000     0.0000     0.6223    0.0819    0.0819    0.0819
d_edge_density              NaN     0.0954     0.0203     0.0299    -0.1456    0.1643   -0.0076    0.0070
d_building_coverage         NaN     0.2163     0.0114     0.0207    -0.2483    0.2502   -0.0064    0.0015
d_texture_entropy           NaN     2.9814     0.0620     0.1430    -3.1864    3.1907   -0.0592    0.0222
d_building_count            NaN    -3.0103     6.3744     5.2308    15.9846   40.2205   19.5026    7.6308
d_building_area_mean        NaN   204.3114  -137.0353   -96.8169  -197.4528 -119.7567 -106.7453  -94.8071
d_amenity_count             NaN     0.5026     0.6564     0.3231     0.1026    0.2974    0.1128    0.1077
d_shop_count                NaN     0.1436     0.1026     0.1128     0.1436    0.0872    0.1077    0.0103
d_leisure_count             NaN     0.0000     0.0154     0.0308     0.0000    0.0359    0.0103    0.0000
d_poi_diversity             NaN     0.0179     0.0030     0.0236     0.0092    0.0478    0.0170   -0.0026
d_road_length_total         NaN     0.0000     0.0000     0.0000     0.6223   -0.5404    0.0000    0.0000

Phase 3a (面板數據整合) 完成！
```

---

**spatial autocorrection**

`python src/spatial_autocorr.py`

```
(py3.10) J:\git\city-urban-evolution-osm-sentinel2>python src/spatial_autocorr.py
面板數據: 1560 筆

計算 Moran's I（逆距離空間權重，閾值 0.015°）...
已儲存: J:\git\city-urban-evolution-osm-sentinel2\output\spatial_autocorr_taichung.csv  (50 筆)

=======================================================
全局 Moran's I 結果摘要
=======================================================

───────────────────────────────────────────────────────
  edge_density
───────────────────────────────────────────────────────
    Year        I        Z        p  解釋
    2019   0.8956   23.357   0.0000**  clustered (high-high / low-low)
    2020   0.9052   23.588   0.0000**  clustered (high-high / low-low)
    2021   0.9054   23.617   0.0000**  clustered (high-high / low-low)
    2023   0.9056   23.620   0.0000**  clustered (high-high / low-low)
    2024   0.9075   23.668   0.0000**  clustered (high-high / low-low)
    2025   0.9084   23.704   0.0000**  clustered (high-high / low-low)

───────────────────────────────────────────────────────
  building_coverage
───────────────────────────────────────────────────────
    Year        I        Z        p  解釋
    2019   0.8686   22.666   0.0000**  clustered (high-high / low-low)
    2020   0.7301   19.073   0.0000**  clustered (high-high / low-low)
    2021   0.8562   22.366   0.0000**  clustered (high-high / low-low)
    2023   0.8648   22.585   0.0000**  clustered (high-high / low-low)
    2024   0.8664   22.635   0.0000**  clustered (high-high / low-low)
    2025   0.8667   22.644   0.0000**  clustered (high-high / low-low)

───────────────────────────────────────────────────────
  texture_entropy
───────────────────────────────────────────────────────
    Year        I        Z        p  解釋
    2019   0.8967   23.392   0.0000**  clustered (high-high / low-low)
    2020   0.9077   23.669   0.0000**  clustered (high-high / low-low)
    2021   0.8946   23.341   0.0000**  clustered (high-high / low-low)
    2023   0.9104   23.768   0.0000**  clustered (high-high / low-low)
    2024   0.9121   23.808   0.0000**  clustered (high-high / low-low)
    2025   0.9114   23.801   0.0000**  clustered (high-high / low-low)

───────────────────────────────────────────────────────
  building_count
───────────────────────────────────────────────────────
    Year        I        Z        p  解釋
    2018   0.2367    7.906   0.0000**  clustered (high-high / low-low)
    2019   0.3568    9.595   0.0000**  clustered (high-high / low-low)
    2020   0.3510    9.414   0.0000**  clustered (high-high / low-low)
    2021   0.3804   10.167   0.0000**  clustered (high-high / low-low)
    2022   0.3334    9.025   0.0000**  clustered (high-high / low-low)
    2023   0.3534   10.406   0.0000**  clustered (high-high / low-low)
    2024   0.4757   13.779   0.0000**  clustered (high-high / low-low)
    2025   0.4727   13.584   0.0000**  clustered (high-high / low-low)

───────────────────────────────────────────────────────
  amenity_count
───────────────────────────────────────────────────────
    Year        I        Z        p  解釋
    2018   0.3038    8.194   0.0000**  clustered (high-high / low-low)
    2019   0.3554    9.639   0.0000**  clustered (high-high / low-low)
    2020   0.3229    8.724   0.0000**  clustered (high-high / low-low)
    2021   0.3456    9.329   0.0000**  clustered (high-high / low-low)
    2022   0.3583    9.647   0.0000**  clustered (high-high / low-low)
    2023   0.3446    9.264   0.0000**  clustered (high-high / low-low)
    2024   0.3363    9.026   0.0000**  clustered (high-high / low-low)
    2025   0.3490    9.343   0.0000**  clustered (high-high / low-low)

───────────────────────────────────────────────────────
  shop_count
───────────────────────────────────────────────────────
    Year        I        Z        p  解釋
    2018   0.1976    5.589   0.0000**  clustered (high-high / low-low)
    2019   0.2260    6.354   0.0000**  clustered (high-high / low-low)
    2020   0.2205    6.104   0.0000**  clustered (high-high / low-low)
    2021   0.2199    6.079   0.0000**  clustered (high-high / low-low)
    2022   0.2462    6.716   0.0000**  clustered (high-high / low-low)
    2023   0.2326    6.345   0.0000**  clustered (high-high / low-low)
    2024   0.2563    6.942   0.0000**  clustered (high-high / low-low)
    2025   0.2577    6.978   0.0000**  clustered (high-high / low-low)

───────────────────────────────────────────────────────
  poi_diversity
───────────────────────────────────────────────────────
    Year        I        Z        p  解釋
    2018   0.3095    8.168   0.0000**  clustered (high-high / low-low)
    2019   0.3044    8.030   0.0000**  clustered (high-high / low-low)
    2020   0.3005    7.926   0.0000**  clustered (high-high / low-low)
    2021   0.3207    8.446   0.0000**  clustered (high-high / low-low)
    2022   0.3511    9.231   0.0000**  clustered (high-high / low-low)
    2023   0.3057    8.051   0.0000**  clustered (high-high / low-low)
    2024   0.2971    7.825   0.0000**  clustered (high-high / low-low)
    2025   0.2870    7.565   0.0000**  clustered (high-high / low-low)


年度平均 Moran's I（所有指標）
           I  z_score
year
2018  0.2619   7.4642
2019  0.5576  14.7190
2020  0.5340  14.0711
2021  0.5604  14.7636
2022  0.3223   8.6548
2023  0.5596  14.8627
2024  0.5788  15.3833
2025  0.5790  15.3741

空間自相關分析完成！
```

---

**trend test**

`python src/trend_test.py`

```
(py3.10) J:\git\city-urban-evolution-osm-sentinel2>python src/trend_test.py
面板數據: 1560 筆

============================================================
Mann–Kendall 趨勢顯著性檢定
============================================================

指標                                S    Z_MK      p值 趨勢                    Sen斜率
────────────────────────────────────────────────────────────────────────────────
  shop_count                     28   3.340  0.0008*** increasing         0.112179
  amenity_count                  28   3.340  0.0008*** increasing         0.282051
  building_count                 26   3.093  0.0020**  increasing        14.937179
  poi_diversity                  26   3.093  0.0020**  increasing         0.017884
  building_area_mean            -24  -2.846  0.0044**  decreasing      -118.341397
  edge_density                   17   1.995  0.0461*   increasing         0.014247
  texture_entropy                13   1.496  0.1346    no trend           0.039559
  building_coverage              13   1.496  0.1346    no trend           0.006509
  resnet_cosine_dist             -7  -0.901  0.3675    no trend          -0.017075
  ssim                            3   0.300  0.7639    no trend           0.004855

已儲存: J:\git\city-urban-evolution-osm-sentinel2\output\trend_test_taichung.csv

============================================================
Chow 結構突變點檢定
============================================================

指標                                最佳斷點年       F統計量       p值
────────────────────────────────────────────────────────────
  amenity_count                    2021     46.714   0.0017**
  building_area_mean               2020     31.602   0.0035**
  shop_count                       2024     19.209   0.0089**
  building_count                   2023     17.941   0.0101*
  poi_diversity                    2023     10.658   0.0250*
  ssim                             2022     10.429   0.0446*
  resnet_cosine_dist               2022      4.862   0.1145
  building_coverage                2022      2.519   0.1958
  edge_density                     2022      2.483   0.1990
  texture_entropy                  2022      2.408   0.2059

已儲存: J:\git\city-urban-evolution-osm-sentinel2\output\structural_break_taichung.csv

趨勢分析完成！
```

---

## Reproducing the Figures

Figures require the processed panel CSVs in `output/{city}/panel_data.csv`.
Complete the Data Preparation Pipeline above first, then run:

```bash
# Fig 1, 2, 4
python figures/paper_figures.py

# Fig 3
python figures/fig3_structural_break.py
```

Output is saved to `output/viz/paper/`.

---

## Sampling Design

- **Grid**: ~1 km spacing, 195 points per city
- **Buffer**: 500 m radius circular buffer per sampling point
- **Cities**: Taichung (120.61-120.76E, 24.08-24.22N) and Taipei (121.45-121.60E, 24.98-25.12N)
- **Period**: 2018-2025 (8 annual snapshots)
- **OSM download**: January 1st snapshot each year via Overpass time-travel filter

See `docs/overpass_queries.md` for exact query syntax.

---

## Data Availability

| Data | Availability |
|------|-------------|
| OSM GeoJSON (raw) + Sentinel-2 imagery | [Google Drive](https://drive.google.com/drive/folders/1Y_TWLWPU5XBfPrSbe8zrxPgSeEYOJITR?usp=sharing) (`paper/osm_data/`, `paper/raw_images/`) |
| OSM original source | Public via Overpass API (https://overpass-api.de/) |
| Satellite imagery original source | CDSE / Copernicus (https://dataspace.copernicus.eu/) (free registration) |
| Processed panel CSVs | Available from author upon request |
| Output figures | Generated by scripts in this repo |

---

## Citation

If you use this code, please cite the paper (citation to be updated upon acceptance).

---

## License

MIT License
