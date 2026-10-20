# Do mobile phone data improve on the clinic address?

Code for *Do mobile phone data improve on the clinic address? Visitor-origin social vulnerability at Manhattan outpatient clinics checked against Medicare patients*.

| File | What it is |
|---|---|
| `reproduce_paper.ipynb` | The analysis notebook. Run the cells in order. |
| `nas_core.py` | Shared machinery: coordinates, neighbour graphs, spatially blocked folds, ridge out-of-fold prediction, bootstrap helpers. |
| `nas_rank.py` | Ranking comparisons (displacement, top-K overlap, decision impact). Imports `nas_core`. |
| `build_facilities.py` | Reads the weekly foot-traffic files and keeps Manhattan healthcare practices. |

## Setup

Python 3.10 or later.

```
pip install numpy pandas scipy scikit-learn statsmodels matplotlib geopandas esda libpysal pyarrow requests
```

Keep the three `.py` files in the same folder as the notebook.

## How the notebook is organised

1. **Setup.** The first cell holds the one setting you need to change, `DATA_DIR` (see below).
2. **Stage 0.** Builds the working files from the licensed weekly patterns: the practitioner table, the parent-venue map, the weekly panel and annual catchment, and the building-level facility table. If these files already exist in `DATA_DIR`, skip this section.
3. **Analysis.** Builds the site crosswalk, the visitor-origin scores under both reporting schemes, the reporting audit, the benchmark and the checks against HRSA health centres.
4. **Expected visitors and the Medicare patient check.** Predicts each site's visitor-origin score from public information, then compares the scores with the Medicare patients of the practitioners at each site.
5. **Tables and figures.** Table 7 intervals and the print-size figures.

## Data

All input files go in one folder. Point the notebook at it by editing the single line at the top of the first cell:

```python
DATA_DIR = 'data'
```

If you run on Google Colab with the files in Google Drive, set it to the Drive folder instead, for example `DATA_DIR = '/content/drive/MyDrive/Ranking_Health_Facility'`. The notebook mounts Drive only when `DATA_DIR` points there. Results and figures are written to `DATA_DIR/R_series_results/`.

### Public files: download and place in `DATA_DIR`

| File | Source |
|---|---|
| `SVI_US_2022.csv` | CDC/ATSDR Social Vulnerability Index 2022, United States, census tracts (CSV) |
| `NEWYORK.csv`, `NEWJERSEY.csv`, `CONNECTICUT.csv` | CDC/ATSDR SVI 2022, one census-tract file per state (used in the earlier New York cells and the tri-state check) |
| `2022_Gaz_tracts_national.txt` | US Census Bureau 2022 Gazetteer, census tracts |
| `tab20_tract20_tract10_st36.txt` | US Census Bureau 2020-to-2010 tract relationship file, New York |

The notebook downloads the remaining public data itself:

- census tract boundaries (Census TIGER/Line 2022);
- community-district boundaries (NYC Open Data, dataset 5crt-au7u);
- the HRSA health-centre site list;
- the MTA subway station list (data.ny.gov, dataset 39hk-dx4f);
- Medicare Physician & Other Practitioners by Provider, data year 2023 (data.cms.gov API, dataset 8ba584c6-a43a-4b0b-a35a-eb9a59e3a571).

The Medicare download is cached as `R_series_results/cms_providers_manhattan_2023.parquet`, and the cell reuses that file if it exists.

### Licensed files: not included

The weekly foot-traffic patterns for New York City, 2024, are licensed from the data vendor and cannot be redistributed. Place the 48 weekly files, unchanged, in `DATA_DIR/NYC-2024-Weekly-Patterns/`. Every file derived from them is also left out, because it carries visitor counts or origins from the licensed data.

### Files the notebook creates

Every other file the notebook reads is written by an earlier cell. Examples are `site_crosswalk.parquet`, `site_table_E1_with_null.parquet`, `flows_by_week.parquet` and `R_series_results/expected_visitors_E1.parquet`. Run the cells in order the first time.

## Licence

MIT. See `LICENSE`.
