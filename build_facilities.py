"""
build_facilities.py -- S0: weekly patterns -> annual Manhattan facility table.

Streams the weekly foot-traffic files, keeps New York County healthcare POIs,
and aggregates them to one row per facility for the year.

Three decisions are made here deliberately, because each was a defect in the
earlier pipeline:

  1. GEOGRAPHY comes from the county FIPS prefix of poi_cbg, never from the
     `city` field. In this vendor's data `city` is the USPS mailing city;
     Queens alone is split across ~50 neighbourhood names, so any city-based
     filter silently destroys boroughs.

  2. CLASSIFICATION is NAICS-primary. Name matching is used ONLY to exclude,
     never to include. The previous inclusive matching on tokens like "care"
     and "CVS" is what pulled petrol stations and pizzerias into a healthcare
     facility set.

  3. THRESHOLDS are annual, not weekly. A facility that fails a 100-visit
     weekly minimum may comfortably clear an annual one. Applying the
     threshold after aggregation recovers those facilities and is the direct
     fix for the small-N instability in the 2024 window.

Memory: reads only the needed columns, filters each file immediately, and
never holds more than one weekly file at a time.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import re
import warnings

import numpy as np
import pandas as pd

MANHATTAN_FIPS = "36061"

# The paper's stated NAICS whitelist. 6211 offices of physicians,
# 6213 offices of other health practitioners, 6214 outpatient care centres.
NAICS_DEFAULT = ("6211", "6213", "6214")

# Optional wider set, for a sensitivity check. 6212 dentists, 6215 medical
# labs, 6216 home health, 6219 other ambulatory, 622 hospitals, 623 nursing.
NAICS_WIDE = ("6211", "6212", "6213", "6214", "6215", "6216", "6219",
              "6221", "6222", "6223", "6231")

# Exclusion patterns only. If a NAICS-qualifying POI matches one of these,
# it is flagged for inspection -- not silently dropped.
NAME_SUSPECT = re.compile(
    r"(?i)\b(?:citgo|shell|exxon|mobil|bp|sunoco|gulf|speedway|pizza|deli|"
    r"bagel|restaurant|cafe|coffee|starbucks|dunkin|mcdonald|burger|taco|"
    r"diner|grill|tesla|auto|tire|car\s*wash|garage|grocery|liquor|laundr|"
    r"salon|barber|bank|atm|hotel|motel|gym|fitness|school)\b")

WANT = ["placekey", "location_name", "naics_code", "latitude", "longitude",
        "poi_cbg", "raw_visit_counts", "raw_visitor_counts", "median_dwell",
        "distance_from_home", "visits_by_day", "visitor_home_cbgs",
        "date_range_start", "top_category", "sub_category"]


# --------------------------------------------------------------------------
def find_weekly_files(week_dir: str) -> list[tuple[str, str]]:
    """(date, path) for each weekly file. followlinks: Drive shortcuts."""
    files = []
    for dp, _, fn in os.walk(week_dir, followlinks=True):
        for f in fn:
            if f.endswith((".csv.gz", ".csv", ".parquet")):
                files.append(os.path.join(dp, f))
    pairs = sorted({(m.group(1), p) for p in files
                    if (m := re.search(r"(\d{4}-\d{2}-\d{2})", p))})
    if not pairs:
        pairs = sorted((str(i), p) for i, p in enumerate(sorted(files)))
    return pairs


def header_of(path: str) -> list[str]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema.names)
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="ignore") as fh:
        return pd.read_csv(fh, nrows=0).columns.tolist()


def county_of(s: pd.Series) -> pd.Series:
    """Numeric-first, so a float-typed poi_cbg can't render as 3.6e+11."""
    v = pd.to_numeric(s, errors="coerce")
    return (v.astype("Int64").astype(str)
            .str.replace("<NA>", "", regex=False).str.zfill(12).str[:5])


def _read_filtered(path: str, cols: list[str], naics: tuple) -> pd.DataFrame:
    """Read one weekly file, keep Manhattan healthcare rows only."""
    keep = []
    if path.endswith(".parquet"):
        chunks = [pd.read_parquet(path, columns=cols)]
    else:
        chunks = pd.read_csv(
            path, usecols=cols, chunksize=200_000, low_memory=False,
            compression="gzip" if path.endswith(".gz") else None)
    for ch in chunks:
        if "poi_cbg" not in ch.columns:
            continue
        ch = ch[county_of(ch["poi_cbg"]) == MANHATTAN_FIPS]
        if "naics_code" in ch.columns and len(ch):
            n = (ch["naics_code"].astype(str)
                 .str.replace(r"\.0$", "", regex=True))
            ch = ch[n.str[:4].isin(naics) | n.str[:3].isin(
                [c for c in naics if len(c) == 3])]
        if len(ch):
            keep.append(ch)
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------
def _catchment(js) -> tuple[float, float]:
    """(n distinct home CBGs, Shannon entropy of the home distribution)."""
    if not isinstance(js, str) or not js.strip().startswith("{"):
        return (np.nan, np.nan)
    try:
        d = json.loads(js)
    except Exception:
        return (np.nan, np.nan)
    v = np.array([x for x in d.values() if isinstance(x, (int, float)) and x > 0],
                 dtype=float)
    if v.size == 0:
        return (0.0, 0.0)
    p = v / v.sum()
    return (float(v.size), float(-(p * np.log(p)).sum()))


def _weekend_share(js) -> float:
    """visits_by_day is a 7-element list starting Monday."""
    if not isinstance(js, str) or not js.strip().startswith("["):
        return np.nan
    try:
        a = np.asarray(json.loads(js), dtype=float)
    except Exception:
        return np.nan
    return float(a[5:].sum() / a.sum()) if a.size == 7 and a.sum() > 0 else np.nan


# --------------------------------------------------------------------------
def build(week_dir: str, naics: tuple = NAICS_DEFAULT,
          annual_min_visits: int = 100, min_weeks: int = 26,
          max_files: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Stream every weekly file and aggregate to one row per facility."""
    pairs = find_weekly_files(week_dir)
    assert pairs, f"no weekly files under {week_dir}"
    if max_files:
        pairs = pairs[:max_files]

    have = set(header_of(pairs[0][1]))
    cols = [c for c in WANT if c in have]
    missing = [c for c in WANT if c not in have]
    if verbose:
        print(f"{len(pairs)} weekly files")
        print(f"reading columns: {cols}")
        if missing:
            print(f"absent (features using them will be skipped): {missing}")
        dates = [d for d, _ in pairs]
        dt = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
        if len(dt) > 1:
            gaps = dt.diff().dt.days.dropna()
            nmiss = int((gaps > 7).sum())
            print(f"weeks: {dates[0]} to {dates[-1]}   "
                  f"{'contiguous' if nmiss == 0 else f'{nmiss} GAP(S) > 7 days'}")
            if nmiss:
                print("  >> do not describe this as a full year; state coverage")

    frames = []
    for i, (d, p) in enumerate(pairs, 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = _read_filtered(p, cols, naics)
        if len(f):
            f["_week"] = d
            frames.append(f)
        if verbose:
            print(f"  [{i:>2}/{len(pairs)}] {d}  kept {len(f):>6,}", end="\r")
    if verbose:
        print()
    panel = pd.concat(frames, ignore_index=True)
    assert len(panel), "no Manhattan healthcare rows found -- check NAICS filter"
    if verbose:
        print(f"panel: {len(panel):,} facility-weeks, "
              f"{panel['placekey'].nunique():,} distinct facilities")

    # ---- per-week derived quantities, computed on the small subset only ---
    if "visitor_home_cbgs" in panel.columns:
        cat = panel["visitor_home_cbgs"].map(_catchment)
        panel["_n_home_cbgs"] = [c[0] for c in cat]
        panel["_catch_entropy"] = [c[1] for c in cat]
    if "visits_by_day" in panel.columns:
        panel["_weekend_share"] = panel["visits_by_day"].map(_weekend_share)

    # ---- aggregate to one row per facility --------------------------------
    g = panel.groupby("placekey", sort=False)
    agg = pd.DataFrame({
        "location_name": g["location_name"].first() if "location_name" in panel else np.nan,
        "naics_code": g["naics_code"].first(),
        "latitude": g["latitude"].median(),
        "longitude": g["longitude"].median(),
        "poi_cbg": g["poi_cbg"].first(),
        "weeks_present": g.size(),
        "annual_visits": g["raw_visit_counts"].sum(),
        "mean_weekly_visits": g["raw_visit_counts"].mean(),
        "sd_weekly_visits": g["raw_visit_counts"].std(),
    })
    if "raw_visitor_counts" in panel.columns:
        agg["annual_visitors"] = g["raw_visitor_counts"].sum()
        agg["visits_per_visitor"] = agg["annual_visits"] / agg["annual_visitors"].replace(0, np.nan)
    if "median_dwell" in panel.columns:
        agg["mean_dwell"] = g["median_dwell"].mean()
    if "distance_from_home" in panel.columns:
        agg["mean_distance_from_home"] = g["distance_from_home"].mean()
    if "_n_home_cbgs" in panel.columns:
        agg["catchment_breadth"] = g["_n_home_cbgs"].mean()
        agg["catchment_entropy"] = g["_catch_entropy"].mean()
    if "_weekend_share" in panel.columns:
        agg["weekend_share"] = g["_weekend_share"].mean()

    agg["visit_volatility"] = agg["sd_weekly_visits"] / agg["mean_weekly_visits"].replace(0, np.nan)
    agg = agg.reset_index()

    # ---- thresholds applied ANNUALLY --------------------------------------
    before = len(agg)
    agg = agg[(agg["annual_visits"] >= annual_min_visits) &
              (agg["weeks_present"] >= min_weeks)].reset_index(drop=True)
    if verbose:
        print(f"annual thresholds (>= {annual_min_visits} visits, "
              f">= {min_weeks} weeks): {before:,} -> {len(agg):,} facilities")

    # ---- classification audit: flag, do not silently drop -----------------
    if "location_name" in agg.columns:
        susp = agg["location_name"].astype(str).str.contains(NAME_SUSPECT, na=False)
        agg["name_suspect"] = susp
        if verbose:
            print(f"\nNAICS-qualifying but non-healthcare-looking names: "
                  f"{int(susp.sum())} ({susp.mean():.1%})")
            if susp.any():
                for s in agg.loc[susp, "location_name"].head(15):
                    print("   ?", s)
                print("   >> inspect these before they enter a figure")
            n4 = agg["naics_code"].astype(str).str.replace(r"\.0$", "", regex=True).str[:4]
            print(f"\nNAICS composition: {n4.value_counts().head(8).to_dict()}")

    return agg


# --------------------------------------------------------------------------
def add_spatial_context(fac: pd.DataFrame, radii=(500, 1000, 2000)) -> pd.DataFrame:
    """Facility-density and isolation covariates, for R3's attribute set.

    These describe the PLACE, not the facility's own traffic, which is what
    makes them usable as explanatory variables for rank displacement.
    """
    import nas_core as nas
    xy = nas.local_xy(fac["latitude"].to_numpy(), fac["longitude"].to_numpy())
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    out = fac.copy()
    for r in radii:
        out[f"n_facilities_{r}m"] = (D < r).sum(1)
    out["dist_nearest_facility_m"] = D.min(1)
    out["dist_5th_nearest_m"] = np.sort(D, axis=1)[:, min(4, D.shape[1] - 1)]
    return out
