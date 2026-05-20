# Overpass API Queries

This document records the exact queries used to download OSM data for each city and year.

## Method

OSM data was downloaded via **osmnx** (`ox.features_from_bbox`), which internally constructs
Overpass QL queries. Downloads were performed on the dates listed in `data/image_dates.csv`.

All queries use the study area bounding boxes below with annual snapshots via the
`[date:"YYYY-MM-DDT00:00:00Z"]` Overpass time-travel filter.

---

## Bounding Boxes

| City | West | South | East | North |
|------|------|-------|------|-------|
| Taichung | 120.610° E | 24.080° N | 120.760° E | 24.220° N |
| Taipei | 121.445° E | 24.980° N | 121.595° E | 25.120° N |

---

## Tag Filters

| Layer | osmnx tags argument | Notes |
|-------|---------------------|-------|
| Buildings | `{"building": True}` | All polygon features with building=* |
| Amenities | `{"amenity": True}` | All point/polygon features with amenity=* |
| Shops | `{"shop": True}` | All features with shop=* |
| Leisure | `{"leisure": True}` | All features with leisure=* |
| Roads | `{"highway": True}` | All polyline features with highway=* |

---

## Equivalent Raw Overpass QL (example: Taichung buildings, 2022)

```overpassql
[out:json][timeout:180][date:"2022-01-01T00:00:00Z"];
(
  way["building"](24.080, 120.610, 24.220, 120.760);
  relation["building"](24.080, 120.610, 24.220, 120.760);
);
out body;
>;
out skel qt;
```

Replace `"building"` with `"amenity"`, `"shop"`, `"leisure"`, or `"highway"` for other layers.
Replace the date string for other years (always January 1st of the target year).

---

## Python Code (osmnx)

```python
import osmnx as ox

BBOX = {
    "taichung": dict(north=24.220, south=24.080, east=120.760, west=120.610),
    "taipei":   dict(north=25.120, south=24.980, east=121.595, west=121.445),
}
TAGS_LIST = [
    {"building": True},
    {"amenity": True},
    {"shop": True},
    {"leisure": True},
    {"highway": True},
]
YEARS = range(2018, 2026)

for city, bbox in BBOX.items():
    for year in YEARS:
        ox.settings.overpass_settings = f'[out:json][timeout:180][date:"{year}-01-01T00:00:00Z"]'
        for tags in TAGS_LIST:
            gdf = ox.features_from_bbox(
                north=bbox["north"], south=bbox["south"],
                east=bbox["east"],  west=bbox["west"],
                tags=tags
            )
            # save to osm_data/{city}/osm_{year}.geojson
```

---

## Notes

- Download date: always `{year}-01-01T00:00:00Z` (start of year snapshot)
- OSM data is subject to volunteer editing activity; counts reflect mapping completeness
  at the time of download, not solely physical urban change.
- Road data (`highway=*`) was downloaded separately due to geometry type differences
  (polylines vs polygons).
- Cloud masking and satellite image acquisition dates are listed in `data/image_dates.csv`.
