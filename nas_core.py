"""
nas_core.py -- Neighborhood-Aware Staircase, Manhattan study.

Implements the M-series experiments for the revised (de-GNN'd) paper:

    M1  heterogeneity-stratified staircase   <- headline
    M2  weekly staircase over the full year  <- second headline
    M3  dual target (constructed index + observed visits)
    M4  anisotropic-metric K sweep
    M5  temporal neighborhood channels

Plus the shared machinery every one of them needs: spatially blocked folds,
the [x, xbar, c] representation, ridge OOF prediction, the F statistic with a
paired bootstrap CI, and a TOST equivalence test.

Design commitments, stated so they can be checked:

  * Every reported number uses spatially blocked folds. Random folds inflate
    rho substantially here and are not comparable. There is no switch for it.
  * Alpha is selected inside the training folds only. Scalers likewise.
  * Neighborhood channels use neighbors' FEATURES, never their targets, so
    neighbors crossing a fold boundary are not leakage -- in deployment you
    observe every facility's features. This is deliberate; see build_repr().
  * F is bootstrapped, not computed as a ratio of three point estimates.

Author: (you). CPU only; no GPU, no torch.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

__all__ = [
    "detect_columns", "local_xy", "grid_metric", "knn_graph",
    "build_repr", "spatial_folds", "ridge_oof", "spearman",
    "staircase", "bootstrap_F", "tost_equivalence",
    "heterogeneity", "M1_stratified", "M2_weekly", "M3_dual_target",
    "M4_k_sweep", "M5_temporal_channels", "selftest",
]

ALPHAS = np.logspace(-3, 4, 25)

# Manhattan's street grid is rotated ~29 deg east of true north. Avenues run
# NNE-SSW (the fast axis); streets run WNW-ESE (crosstown, the slow axis).
MANHATTAN_GRID_DEG = 29.0


# --------------------------------------------------------------------------
# 0. column detection -- so this runs without you renaming anything
# --------------------------------------------------------------------------

_CANDIDATES = {
    "lat":      ["latitude", "lat", "y", "poi_lat"],
    "lon":      ["longitude", "lon", "lng", "long", "x", "poi_lon"],
    "id":       ["placekey", "facility_id", "poi_id", "id", "safegraph_place_id"],
    "week":     ["week", "date_range_start", "date", "week_start", "period"],
    "target":   ["need_index", "target", "y", "planning_index", "access_need",
                 "need_score", "index"],
    "visits":   ["raw_visit_counts", "visits", "visit_count", "raw_visits",
                 "annual_visits"],
}


def detect_columns(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Best-effort column mapping. Prints what it found; override manually."""
    low = {c.lower(): c for c in df.columns}
    found = {}
    for role, names in _CANDIDATES.items():
        hit = None
        for n in names:
            if n in df.columns:
                hit = n
                break
            if n.lower() in low:
                hit = low[n.lower()]
                break
        found[role] = hit
    if verbose:
        print("column detection:")
        for k, v in found.items():
            mark = "ok " if v else "-- "
            print(f"  {mark}{k:<8} -> {v}")
        if not found["lat"] or not found["lon"]:
            print("  !! no coordinates found. Spatial folds and the graph both "
                  "need them. Pass lat/lon column names explicitly.")
    return found


def feature_matrix(df: pd.DataFrame, exclude: Sequence[str] = (),
                   verbose: bool = True) -> tuple[np.ndarray, list[str]]:
    """Numeric columns minus coordinates, ids, targets and anything excluded.

    Coordinates are deliberately excluded from x. If raw lat/lon are features,
    the model can memorize position and the neighborhood channels stop being
    the thing under test.
    """
    ex = {c.lower() for c in exclude}
    ex |= {"latitude", "lat", "longitude", "lon", "lng", "long"}
    cols = [c for c in df.columns
            if c.lower() not in ex
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().mean() > 0.5
            and df[c].nunique() > 2]
    X = df[cols].to_numpy(dtype=float)
    # median-impute; report it rather than doing it silently
    nan_frac = np.isnan(X).mean()
    if nan_frac > 0:
        med = np.nanmedian(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(med, idx[1])
        if verbose:
            print(f"  imputed {nan_frac:.2%} of feature cells with column medians")
    if verbose:
        print(f"  feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
        print(f"  features: {cols}")
    return X, cols


# --------------------------------------------------------------------------
# 1. geometry
# --------------------------------------------------------------------------

def local_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Equirectangular projection to metres about the sample centroid.

    Accurate to well under a metre over Manhattan's ~20 km extent, and it
    keeps everything in a plane so rotation is meaningful.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lat0 = np.nanmean(lat)
    R = 6_371_000.0
    x = np.radians(lon - np.nanmean(lon)) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R
    return np.column_stack([x, y])


def grid_metric(xy: np.ndarray, theta_deg: float = MANHATTAN_GRID_DEG,
                w_cross: float = 1.0) -> np.ndarray:
    """Rotate into grid-aligned axes and inflate the crosstown axis.

    w_cross = 1.0 recovers plain Euclidean (rotation alone is isometric).
    w_cross > 1 makes crosstown separation count for more, which is the
    point: in Manhattan, ten blocks uptown is a far shorter trip than two
    avenues across. This is the anisotropy the paper argues for in M4.

    After rotation, axis 0 is crosstown (street direction) and axis 1 is
    uptown-downtown (avenue direction).
    """
    t = np.radians(theta_deg)
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    rot = xy @ R.T
    return rot * np.array([float(w_cross), 1.0])


def knn_graph(xy: np.ndarray, k: int = 10,
              decay: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """K nearest neighbours, self excluded.

    Returns (idx, w) with shapes (n, k). If decay is None the weights are
    uniform; otherwise w = exp(-d / decay) row-normalised, which is the
    distance-decayed variant.
    """
    n = xy.shape[0]
    k = int(min(k, n - 1))
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    idx = np.argsort(d2, axis=1)[:, :k]
    d = np.sqrt(np.take_along_axis(d2, idx, axis=1))
    if decay is None:
        w = np.full_like(d, 1.0 / k)
    else:
        w = np.exp(-d / float(decay))
        w = w / np.clip(w.sum(1, keepdims=True), 1e-12, None)
    return idx, w


# --------------------------------------------------------------------------
# 2. the representation under test
# --------------------------------------------------------------------------

def build_repr(X: np.ndarray, idx: np.ndarray, w: np.ndarray,
               channels: str = "x_xbar_c") -> np.ndarray:
    """Assemble the neighborhood-aware input.

        x     the facility's own features
        xbar  the (optionally weighted) mean of its neighbours' features
        c     x - xbar, the contrast

    Note span([x, xbar, c]) == span([x, xbar]); the contrast channel is a
    reparameterisation, not new information. Any effect it has is a
    regularisation effect and must be reported as such. That is exactly what
    the channel ablation (variants below) is for.

    Uses neighbours' features only -- never their targets -- so neighbours
    that fall in another CV fold are not leakage.
    """
    xbar = np.einsum("nk,nkf->nf", w, X[idx])
    c = X - xbar
    parts = {"x": X, "xbar": xbar, "c": c}
    keys = channels.split("_")
    unknown = [k for k in keys if k not in parts]
    if unknown:
        raise ValueError(f"unknown channel(s) {unknown}; use x, xbar, c")
    return np.hstack([parts[k] for k in keys])


CHANNEL_VARIANTS = ["x", "xbar", "x_xbar", "x_c", "xbar_c", "x_xbar_c"]


# --------------------------------------------------------------------------
# 3. evaluation machinery
# --------------------------------------------------------------------------

def spatial_folds(xy: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Contiguous spatial blocks via KMeans on coordinates.

    Blocked, not random. Random folds put a facility's own neighbours in the
    training set and inflate every number in this file.
    """
    km = KMeans(n_clusters=n_folds, n_init=10, random_state=seed)
    return km.fit_predict(xy)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def ridge_oof(Z: np.ndarray, y: np.ndarray, folds: np.ndarray,
              rank_target: bool = True, alphas=ALPHAS) -> np.ndarray:
    """Out-of-fold ridge predictions.

    rank_target=True fits on the normal-scores transform of y. The metric is
    Spearman, so fitting squared error on a skewed index is a mismatch; the
    transform costs nothing and keeps the model linear and auditable.
    Scaler and alpha are fitted inside the training folds only.
    """
    y = np.asarray(y, float).ravel()
    oof = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if tr.sum() < 10 or te.sum() < 2:
            continue
        sc = StandardScaler().fit(Z[tr])
        ytr = y[tr]
        if rank_target:
            r = rankdata(ytr) / (len(ytr) + 1.0)
            from scipy.stats import norm
            ytr = norm.ppf(r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = RidgeCV(alphas=alphas).fit(sc.transform(Z[tr]), ytr)
        oof[te] = m.predict(sc.transform(Z[te]))
    return oof


@dataclass
class Staircase:
    rho: dict = field(default_factory=dict)
    pred: dict = field(default_factory=dict)
    F: float = np.nan
    n: int = 0

    def __repr__(self):
        s = "  ".join(f"{k}={v:.4f}" for k, v in self.rho.items())
        return f"<Staircase n={self.n} {s} F={self.F:.3f}>"


def staircase(X: np.ndarray, y: np.ndarray, xy: np.ndarray, k: int = 10,
              n_folds: int = 5, seed: int = 0, decay: float | None = None,
              model_pred: np.ndarray | None = None,
              rank_target: bool = True) -> Staircase:
    """The three rungs: own features, + neighborhood, + graph training.

    model_pred supplies OOF predictions from the trainable comparator if you
    have them. Without it F is undefined and only the first two rungs report.
    """
    folds = spatial_folds(xy, n_folds, seed)
    idx, w = knn_graph(xy, k, decay)
    p_ridge = ridge_oof(build_repr(X, idx, w, "x"), y, folds, rank_target)
    p_rs = ridge_oof(build_repr(X, idx, w, "x_xbar_c"), y, folds, rank_target)
    out = Staircase(n=len(y))
    out.pred = {"Ridge": p_ridge, "Ridge-Spatial": p_rs}
    out.rho = {"Ridge": spearman(y, p_ridge), "Ridge-Spatial": spearman(y, p_rs)}
    if model_pred is not None:
        out.pred["Trainable"] = np.asarray(model_pred, float)
        out.rho["Trainable"] = spearman(y, model_pred)
        den = out.rho["Trainable"] - out.rho["Ridge"]
        out.F = ((out.rho["Ridge-Spatial"] - out.rho["Ridge"]) / den
                 if den > 1e-3 else np.nan)
    return out


def bootstrap_F(y, pred_ridge, pred_rs, pred_model=None, n_boot: int = 5000,
                seed: int = 0, subset: np.ndarray | None = None) -> dict:
    """Paired facility-level bootstrap on the staircase differences and F.

    One resample of facility indices scores every model, so the models'
    dependence is preserved. F is a ratio and blows up when its denominator
    approaches zero; the fraction of resamples where that happens is
    reported, and any F with invalid_frac above ~0.10 should not be quoted
    as a point estimate.
    """
    y = np.asarray(y, float).ravel()
    if subset is not None:
        sel = np.asarray(subset)
        y = y[sel]
        pred_ridge, pred_rs = pred_ridge[sel], pred_rs[sel]
        pred_model = None if pred_model is None else pred_model[sel]
    n = len(y)
    rng = np.random.default_rng(seed)
    pt = {"Ridge": spearman(y, pred_ridge), "Ridge-Spatial": spearman(y, pred_rs)}
    if pred_model is not None:
        pt["Trainable"] = spearman(y, pred_model)

    d_rs, d_mod, d_gap, Fv = [], [], [], []
    invalid = 0
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        if len(np.unique(y[b])) < 3:
            invalid += 1
            continue
        r_r = spearman(y[b], pred_ridge[b])
        r_s = spearman(y[b], pred_rs[b])
        d_rs.append(r_s - r_r)
        if pred_model is not None:
            r_m = spearman(y[b], pred_model[b])
            d_mod.append(r_m - r_r)
            d_gap.append(r_m - r_s)
            if (r_m - r_r) > 1e-3:
                Fv.append((r_s - r_r) / (r_m - r_r))
            else:
                invalid += 1

    def ci(v):
        v = np.asarray(v)
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    res = {"n": n, "rho": pt,
           "gain_rs": float(np.mean(d_rs)), "gain_rs_ci": ci(d_rs)}
    if pred_model is not None and d_mod:
        res.update({
            "gain_model": float(np.mean(d_mod)), "gain_model_ci": ci(d_mod),
            "model_minus_rs": float(np.mean(d_gap)),
            "model_minus_rs_ci": ci(d_gap),
        })
        den = pt["Trainable"] - pt["Ridge"]
        res["F"] = (pt["Ridge-Spatial"] - pt["Ridge"]) / den if den > 1e-3 else np.nan
        if Fv:
            res["F_median"] = float(np.median(Fv))
            res["F_ci"] = ci(Fv)
        res["F_invalid_frac"] = invalid / float(n_boot)
    return res


def tost_equivalence(boot: dict, delta: float = 0.05) -> str:
    """Equivalence verdict on trainable-minus-representation.

    'We could not detect a difference' is not 'the difference is negligible'.
    This bounds it against a margin you declare BEFORE running.
    """
    if "model_minus_rs_ci" not in boot:
        return "no trainable-model predictions supplied"
    lo, hi = boot["model_minus_rs_ci"]
    if -delta < lo and hi < delta:
        return f"EQUIVALENT at delta={delta:.3f} (95% CI [{lo:.4f}, {hi:.4f}] inside margin)"
    if lo > delta or hi < -delta:
        return f"NOT equivalent: CI [{lo:.4f}, {hi:.4f}] excludes the margin"
    return (f"INDETERMINATE: difference bounded by {max(abs(lo), abs(hi)):.4f}, "
            f"equivalence at delta={delta:.3f} not established")


# --------------------------------------------------------------------------
# 4. M1 -- heterogeneity-stratified staircase  (headline)
# --------------------------------------------------------------------------

def heterogeneity(X: np.ndarray, idx: np.ndarray,
                  method: str = "dispersion") -> np.ndarray:
    """How internally varied is each facility's neighbourhood?

    method='dispersion' (default, and the one to report): mean over features
    of the standard deviation among the K neighbours, on standardised
    features. It describes the neighbourhood alone.

    method='contrast': ||x_i - xbar_i||. Do NOT stratify on this for the
    headline result -- the contrast channel is a model input, so stratifying
    on it and then showing the contrast channel helps is close to circular.
    Provided only as a sensitivity check.
    """
    Xs = StandardScaler().fit_transform(X)
    if method == "dispersion":
        return Xs[idx].std(axis=1).mean(axis=1)
    if method == "contrast":
        return np.linalg.norm(Xs - Xs[idx].mean(1), axis=1)
    raise ValueError("method must be 'dispersion' or 'contrast'")


def M1_stratified(X, y, xy, k: int = 10, n_strata: int = 5, n_folds: int = 5,
                  seed: int = 0, model_pred=None, n_boot: int = 3000,
                  het_method: str = "dispersion") -> pd.DataFrame:
    """Fit once on everything, then evaluate the staircase within strata.

    Pooled fit is deliberate: it isolates WHERE the representation pays while
    holding the model fixed. Fitting separately per stratum would confound
    the effect with changes in n and in training distribution.

    Read the output with range restriction in mind -- rho within a stratum is
    computed on a narrower slice of the target, so all strata sit lower than
    the pooled value. F is a ratio of differences and is the robust column.
    """
    idx, w = knn_graph(xy, k)
    het = heterogeneity(X, idx, het_method)
    st = staircase(X, y, xy, k=k, n_folds=n_folds, seed=seed,
                   model_pred=model_pred)
    q = pd.qcut(het, n_strata, labels=False, duplicates="drop")

    rows = []
    for s in sorted(pd.unique(q[~pd.isna(q)])):
        m = (q == s)
        if m.sum() < 30:
            print(f"  !! stratum {s}: n={m.sum()} is too small to interpret")
        b = bootstrap_F(y, st.pred["Ridge"], st.pred["Ridge-Spatial"],
                        st.pred.get("Trainable"), n_boot=n_boot, seed=seed,
                        subset=m)
        rows.append({
            "stratum": int(s) + 1, "n": int(m.sum()),
            "het_median": float(np.median(het[m])),
            "target_std": float(np.std(np.asarray(y)[m])),
            "rho_Ridge": b["rho"]["Ridge"],
            "rho_RidgeSpatial": b["rho"]["Ridge-Spatial"],
            "gain": b["gain_rs"],
            "gain_lo": b["gain_rs_ci"][0], "gain_hi": b["gain_rs_ci"][1],
            "F": b.get("F", np.nan),
            "F_lo": b.get("F_ci", (np.nan, np.nan))[0],
            "F_hi": b.get("F_ci", (np.nan, np.nan))[1],
        })
    df = pd.DataFrame(rows)
    if len(df) > 2:
        rho_trend = spearman(df["het_median"].to_numpy(), df["gain"].to_numpy())
        print(f"\n  monotonicity of gain in heterogeneity: rho = {rho_trend:.3f}")
        print("  (the M1 hypothesis predicts this is strongly positive)")
    return df


# --------------------------------------------------------------------------
# 5. M2 -- weekly staircase over the full year
# --------------------------------------------------------------------------

def M2_weekly(week_tables: dict, k: int = 10, n_folds: int = 5, seed: int = 0,
              n_boot: int = 1000, model_pred_by_week: dict | None = None
              ) -> pd.DataFrame:
    """Run the staircase independently in each week.

    week_tables: {week_label: (X, y, xy)}. Build these with your own feature
    pipeline; this function only evaluates them.

    The point is not that any single week is interesting. It is whether the
    gain and F are stable across ~48 weeks. Stability is the replication axis
    that partly substitutes for the spatial breadth Manhattan-only gives up.
    """
    rows = []
    for wk in sorted(week_tables):
        X, y, xy = week_tables[wk]
        mp = None if model_pred_by_week is None else model_pred_by_week.get(wk)
        st = staircase(X, y, xy, k=k, n_folds=n_folds, seed=seed, model_pred=mp)
        b = bootstrap_F(y, st.pred["Ridge"], st.pred["Ridge-Spatial"],
                        st.pred.get("Trainable"), n_boot=n_boot, seed=seed)
        rows.append({
            "week": wk, "n": len(y),
            "rho_Ridge": st.rho["Ridge"],
            "rho_RidgeSpatial": st.rho["Ridge-Spatial"],
            "gain": b["gain_rs"],
            "gain_lo": b["gain_rs_ci"][0], "gain_hi": b["gain_rs_ci"][1],
            "F": b.get("F", np.nan),
        })
    df = pd.DataFrame(rows)
    if len(df) > 1:
        print(f"\n  gain across weeks: mean={df['gain'].mean():.4f}  "
              f"sd={df['gain'].std():.4f}  "
              f"min={df['gain'].min():.4f}  max={df['gain'].max():.4f}")
        print("  Report the SD next to the gain. If the between-week SD is of")
        print("  the same order as the gain itself, the effect is not stable")
        print("  and the paper must say so.")
    return df


def rank_stability(week_tables: dict, pred_by_week: dict,
                   top_frac: float = 0.10) -> pd.DataFrame:
    """Week-to-week churn in the top decile of the ranking.

    Planners act on the head of the list. If it reshuffles every week, that
    is a finding about the method's deployability, not a nuisance.
    """
    weeks = sorted(pred_by_week)
    rows = []
    for a, b in zip(weeks[:-1], weeks[1:]):
        pa, pb = pred_by_week[a], pred_by_week[b]
        n = min(len(pa), len(pb))
        m = max(1, int(round(top_frac * n)))
        ta = set(np.argsort(-pa[:n])[:m])
        tb = set(np.argsort(-pb[:n])[:m])
        rows.append({"from": a, "to": b, "top_n": m,
                     "overlap": len(ta & tb) / m,
                     "spearman_full": spearman(pa[:n], pb[:n])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. M3 -- dual target
# --------------------------------------------------------------------------

def M3_dual_target(X, xy, y_index, y_visits, k: int = 10, n_folds: int = 5,
                   seed: int = 0, n_boot: int = 3000) -> pd.DataFrame:
    """Same staircase, two targets: the constructed index and observed visits.

    This is the direct answer to the circularity objection. If the staircase
    holds on a MEASURED quantity as well as on the index you built, the
    finding is not an artifact of the index's construction. If it holds only
    on the index, you have found circularity in your own work -- which is
    worth knowing, and worth reporting.

    Keep the index primary. Switching wholesale to visits turns this into a
    demand-prediction paper and the planning framing disappears.
    """
    rows = []
    for name, y in [("constructed index", y_index), ("observed visits", y_visits)]:
        if y is None:
            continue
        st = staircase(X, y, xy, k=k, n_folds=n_folds, seed=seed)
        b = bootstrap_F(y, st.pred["Ridge"], st.pred["Ridge-Spatial"],
                        n_boot=n_boot, seed=seed)
        rows.append({"target": name,
                     "rho_Ridge": st.rho["Ridge"],
                     "rho_RidgeSpatial": st.rho["Ridge-Spatial"],
                     "gain": b["gain_rs"],
                     "gain_lo": b["gain_rs_ci"][0],
                     "gain_hi": b["gain_rs_ci"][1]})
    df = pd.DataFrame(rows)
    if len(df) == 2:
        print(f"\n  correlation between the two targets: "
              f"{spearman(np.asarray(y_index), np.asarray(y_visits)):.3f}")
        print("  If this is very high, the 'independent' target is not "
              "independent and the check is weaker than it looks.")
    return df


# --------------------------------------------------------------------------
# 7. M4 -- anisotropic K sweep
# --------------------------------------------------------------------------

def M4_k_sweep(X, y, xy_raw, ks: Iterable[int] = (5, 10, 25, 50, 100),
               w_cross: Iterable[float] = (1.0, 1.5, 2.5),
               theta_deg: float = MANHATTAN_GRID_DEG, decay=None,
               n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Sweep neighbourhood size against grid anisotropy.

    w_cross=1.0 is plain Euclidean and is the control row. K=10 was tuned on
    a five-borough sample; in Manhattan the informative scale is almost
    certainly larger, so the sweep runs well past 50.
    """
    rows = []
    for wc in w_cross:
        xy = grid_metric(xy_raw, theta_deg, wc)
        folds = spatial_folds(xy_raw, n_folds, seed)   # folds from true geography
        for k in ks:
            if k >= len(y):
                continue
            idx, w = knn_graph(xy, k, decay)
            p_r = ridge_oof(build_repr(X, idx, w, "x"), y, folds)
            p_s = ridge_oof(build_repr(X, idx, w, "x_xbar_c"), y, folds)
            rows.append({"w_cross": wc, "K": k,
                         "rho_Ridge": spearman(y, p_r),
                         "rho_RidgeSpatial": spearman(y, p_s),
                         "gain": spearman(y, p_s) - spearman(y, p_r)})
    df = pd.DataFrame(rows)
    print("\n  Pick K by the sweep, then report the sweep. Selecting K on the")
    print("  same folds you report is mild optimism -- state it plainly, or")
    print("  nest the choice inside an outer CV loop if a reviewer presses.")
    return df


def M4_channel_ablation(X, y, xy, k: int = 10, n_folds: int = 5,
                        seed: int = 0) -> pd.DataFrame:
    """Which channel carries the signal? (the S2 experiment, folded in here)

    Remember span([x,xbar,c]) == span([x,xbar]). If x_xbar_c beats x_xbar,
    that is regularisation, not information, and must be described that way.
    """
    folds = spatial_folds(xy, n_folds, seed)
    idx, w = knn_graph(xy, k)
    rows = []
    for ch in CHANNEL_VARIANTS:
        p = ridge_oof(build_repr(X, idx, w, ch), y, folds)
        rows.append({"channels": ch, "d": build_repr(X, idx, w, ch).shape[1],
                     "rho": spearman(y, p)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 8. M5 -- temporal neighborhood channels
# --------------------------------------------------------------------------

def M5_temporal_channels(panel: pd.DataFrame, id_col: str, week_col: str,
                         feat_cols: Sequence[str], target_col: str,
                         lat_col: str, lon_col: str, k: int = 10,
                         lags: Sequence[int] = (1,), trail: int = 4,
                         n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Add lagged neighbourhood context and own-trend deviation.

    Two new channel types the original design could not have:
      xbar_lag  the neighbourhood mean L weeks ago
      own_dev   x_t minus the facility's own trailing mean over `trail` weeks

    Evaluated on the final week, so the lags are strictly in the past. This
    is the one place where fold leakage could sneak in through time rather
    than space, so the target week is held out as a whole.
    """
    panel = panel.sort_values([id_col, week_col])
    weeks = sorted(panel[week_col].unique())
    if len(weeks) < max(lags) + trail + 1:
        raise ValueError(f"need at least {max(lags) + trail + 1} weeks, got {len(weeks)}")
    tgt_week = weeks[-1]

    cur = panel[panel[week_col] == tgt_week].set_index(id_col)
    ids = cur.index.to_numpy()
    X = cur[list(feat_cols)].to_numpy(float)
    y = cur[target_col].to_numpy(float)
    xy = local_xy(cur[lat_col].to_numpy(), cur[lon_col].to_numpy())
    idx, w = knn_graph(xy, k)
    folds = spatial_folds(xy, n_folds, seed)

    blocks = {"base": build_repr(X, idx, w, "x_xbar_c")}

    for L in lags:
        wk = weeks[-1 - L]
        past = (panel[panel[week_col] == wk].set_index(id_col)
                .reindex(ids)[list(feat_cols)].to_numpy(float))
        past = np.where(np.isnan(past), np.nanmedian(past, axis=0), past)
        blocks[f"xbar_lag{L}"] = np.einsum("nk,nkf->nf", w, past[idx])

    hist = panel[panel[week_col].isin(weeks[-1 - trail:-1])]
    trail_mean = (hist.groupby(id_col)[list(feat_cols)].mean()
                  .reindex(ids).to_numpy(float))
    trail_mean = np.where(np.isnan(trail_mean),
                          np.nanmedian(trail_mean, axis=0), trail_mean)
    blocks["own_dev"] = X - trail_mean

    rows = []
    cumulative, names = [], []
    for name, B in blocks.items():
        cumulative.append(B)
        names.append(name)
        Z = np.hstack(cumulative)
        rows.append({"added": name, "channels": "+".join(names),
                     "d": Z.shape[1], "rho": spearman(y, ridge_oof(Z, y, folds))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 9. self-test on synthetic data with a known answer
# --------------------------------------------------------------------------

def selftest(n: int = 600, d: int = 8, seed: int = 0, verbose: bool = True):
    """Synthetic Manhattan-shaped data where the M1 hypothesis is TRUE.

    Facilities sit in an elongated north-south strip. The target depends on
    the neighbourhood contrast, and it does so MORE strongly where the local
    neighbourhood is heterogeneous. If M1 does not recover a rising gain
    here, the code is wrong -- not the data.
    """
    rng = np.random.default_rng(seed)
    # elongated north-south strip, Manhattan's aspect ratio
    lat = 40.70 + 0.16 * rng.random(n)
    lon = -74.01 + 0.025 * rng.random(n)
    xy = local_xy(lat, lon)

    # The mechanism, made explicit: a smooth spatial field plus LOCAL
    # idiosyncrasy whose size varies north-to-south. Where idiosyncrasy is
    # small the neighbours look like the facility, so c -> 0 and the contrast
    # channel is a constant carrying nothing. Where it is large, c varies and
    # carries signal. This is precisely the homogeneity problem Manhattan is
    # expected to have.
    het = (lat - lat.min()) / (lat.max() - lat.min())        # 0 = homogeneous
    field = np.column_stack([np.sin(3 * het * np.pi + j) for j in range(d)]) * 2.0
    X = field + rng.normal(0, 1, (n, d)) * (0.05 + 1.6 * het)[:, None]

    idx, w = knn_graph(xy, 10)
    c = X - np.einsum("nk,nkf->nf", w, X[idx])
    beta_c, beta_x = rng.normal(0, 1, d), rng.normal(0, 1, d)
    # noise chosen so rho lands near the 0.4-0.6 range the real study reports,
    # rather than at a ceiling where no experiment can separate anything
    y = 2.0 * (c @ beta_c) + 0.8 * (X @ beta_x) + rng.normal(0, 6.0, n)

    if verbose:
        print("=" * 70)
        print("SELF-TEST -- synthetic data, M1 hypothesis true by construction")
        print("=" * 70)

    st = staircase(X, y, xy, k=10)
    # stand-in for the trainable comparator: RS plus a small real improvement
    model_pred = st.pred["Ridge-Spatial"] + 0.25 * rng.normal(0, 1, n)
    st = staircase(X, y, xy, k=10, model_pred=model_pred)

    if verbose:
        print(f"\nstaircase: {st}")
    b = bootstrap_F(y, st.pred["Ridge"], st.pred["Ridge-Spatial"],
                    st.pred["Trainable"], n_boot=2000)
    if verbose:
        print(f"  gain      = {b['gain_rs']:.4f}  95% CI {b['gain_rs_ci']}")
        print(f"  F         = {b.get('F', float('nan')):.3f}  "
              f"95% CI {b.get('F_ci')}  invalid {b.get('F_invalid_frac', 0):.1%}")
        print(f"  TOST      : {tost_equivalence(b, 0.05)}")
        print("\nM1 stratified:")
    m1 = M1_stratified(X, y, xy, k=10, n_boot=800, model_pred=model_pred)
    if verbose:
        print(m1.round(4).to_string(index=False))
        print("\nchannel ablation:")
        print(M4_channel_ablation(X, y, xy).round(4).to_string(index=False))
        print("\nK sweep x anisotropy:")
        print(M4_k_sweep(X, y, xy, ks=(5, 10, 25, 50)).round(4).to_string(index=False))

    trend = spearman(m1["het_median"].to_numpy(), m1["gain"].to_numpy())
    ok = (st.rho["Ridge-Spatial"] > st.rho["Ridge"] and trend > 0.5
          and 0.2 < st.rho["Ridge"] < 0.8)
    print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'}  "
          f"(gain>0, M1 trend={trend:.2f}>0.5, rho in a realistic range)")
    return ok


if __name__ == "__main__":
    selftest()
