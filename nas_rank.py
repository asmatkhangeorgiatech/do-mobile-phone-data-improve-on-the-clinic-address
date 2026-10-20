"""
nas_rank.py -- Ranking-impact experiments (R-series).

The question this module answers is NOT "does the model fit better." It is:
when neighbourhood context is made explicit, WHICH healthcare facilities move
in the need ranking, HOW FAR, and IS THE MOVEMENT SYSTEMATIC?

That distinction drives everything here. Spearman rho compresses a whole
reordering into one number and hides the thing a planner cares about: a gain
of 0.05 in rho is compatible with every facility shifting slightly OR with a
dozen facilities moving a hundred places. Only the second is a finding about
facility ranking.

Claim strength, stated up front so the write-up respects it:

  STRONG  (descriptive, target-independent)
      The ranking changes by this much, these facilities move, and the
      movement correlates with these place characteristics.

  WEAKER  (evaluative, target-dependent)
      The changed ranking agrees better/worse with the planning index.
      This is only as trustworthy as the index.

  STRONG  (decision-relevant)
      Selecting the top N for intervention yields a different set.

Lead with the first and third. The second is secondary.

Requires nas_core in the same directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import nas_core as nas

__all__ = [
    "R1_displacement", "R2_topk_churn", "R3_who_moves",
    "R4_head_vs_body", "R5_decision_impact", "R6_displacement_stability",
    "ndcg_at_k", "precision_at_k", "ranking_report", "selftest_rank",
]


# --------------------------------------------------------------------------
# ranking metrics
# --------------------------------------------------------------------------

def _to_rank(pred: np.ndarray) -> np.ndarray:
    """Rank 1 = highest predicted need. Ties averaged."""
    return rankdata(-np.asarray(pred, float))


def ndcg_at_k(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    """NDCG@k with gain proportional to the need index.

    y is min-max scaled to [0, 1] so that gains are non-negative and
    comparable across targets. The ideal ranking sorts by y itself.
    """
    y = np.asarray(y, float)
    rng = y.max() - y.min()
    g = (y - y.min()) / rng if rng > 0 else np.zeros_like(y)
    k = int(min(k, len(y)))
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    order = np.argsort(-np.asarray(pred, float))[:k]
    ideal = np.argsort(-g)[:k]
    dcg = float((g[order] * disc).sum())
    idcg = float((g[ideal] * disc).sum())
    return dcg / idcg if idcg > 0 else np.nan


def precision_at_k(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Fraction of the predicted top-k that belong to the true top-k."""
    k = int(min(k, len(y)))
    true_top = set(np.argsort(-np.asarray(y, float))[:k])
    pred_top = set(np.argsort(-np.asarray(pred, float))[:k])
    return len(true_top & pred_top) / k


# --------------------------------------------------------------------------
# R1 -- how far does the ranking move?
# --------------------------------------------------------------------------

def R1_displacement(pred_base: np.ndarray, pred_nbr: np.ndarray,
                    labels: pd.Series | None = None,
                    n_boot: int = 2000, seed: int = 0) -> dict:
    """Per-facility rank displacement when neighbourhood context is added.

    This is the paper's headline quantity. It is computed entirely from the
    two rankings and does NOT involve the target, so it is immune to the
    "your index is constructed" objection.

    Returns the per-facility table plus summary statistics with bootstrap
    intervals on the summaries.
    """
    r0, r1 = _to_rank(pred_base), _to_rank(pred_nbr)
    n = len(r0)
    delta = r1 - r0                      # negative = moved UP in need ranking
    absd = np.abs(delta)

    tab = pd.DataFrame({
        "rank_own_features": r0.astype(int),
        "rank_neighbour_aware": r1.astype(int),
        "delta": delta.astype(int),
        "abs_delta": absd.astype(int),
        "pct_of_list": 100 * absd / n,
    })
    if labels is not None:
        tab.insert(0, "facility", np.asarray(labels))

    rng = np.random.default_rng(seed)
    boots = {"mean_abs": [], "median_abs": [], "p90_abs": [], "frac_moved_10pct": []}
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        a = absd[b]
        boots["mean_abs"].append(a.mean())
        boots["median_abs"].append(np.median(a))
        boots["p90_abs"].append(np.percentile(a, 90))
        boots["frac_moved_10pct"].append((a > 0.10 * n).mean())

    def ci(v):
        v = np.asarray(v)
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    summary = {
        "n": n,
        "delta_array": delta,
        "abs_delta_array": absd,
        "mean_abs_displacement": float(absd.mean()),
        "mean_abs_ci": ci(boots["mean_abs"]),
        "median_abs_displacement": float(np.median(absd)),
        "median_abs_ci": ci(boots["median_abs"]),
        "p90_abs_displacement": float(np.percentile(absd, 90)),
        "p90_abs_ci": ci(boots["p90_abs"]),
        "max_abs_displacement": int(absd.max()),
        "frac_moved_over_10pct_of_list": float((absd > 0.10 * n).mean()),
        "frac_moved_ci": ci(boots["frac_moved_10pct"]),
        "spearman_between_rankings": nas.spearman(pred_base, pred_nbr),
    }

    print(f"R1  n={n}")
    print(f"    mean |displacement|   = {summary['mean_abs_displacement']:.1f} "
          f"positions  95% CI {np.round(summary['mean_abs_ci'], 1)}")
    print(f"    median                = {summary['median_abs_displacement']:.1f}")
    print(f"    90th percentile       = {summary['p90_abs_displacement']:.1f}")
    print(f"    max                   = {summary['max_abs_displacement']}")
    print(f"    moved >10% of list    = "
          f"{summary['frac_moved_over_10pct_of_list']:.1%}")
    print(f"    rank correlation between the two rankings = "
          f"{summary['spearman_between_rankings']:.3f}")
    print("    (a high correlation with a large tail means the reordering is")
    print("     concentrated in a minority of facilities -- report both)")
    return {"table": tab.sort_values("abs_delta", ascending=False), **summary}


# --------------------------------------------------------------------------
# R2 -- does the head of the list change?
# --------------------------------------------------------------------------

def R2_topk_churn(pred_base, pred_nbr, ks=(10, 25, 50, 100),
                  labels: pd.Series | None = None) -> tuple:
    """Composition change in the top-K, which is what planners actually use.

    Returns (summary_frame, entered_left_detail). Overlap is the fraction of
    the neighbour-aware top-K also present in the own-features top-K.
    """
    rows, detail = [], {}
    n = len(pred_base)
    for k in ks:
        if k > n:
            continue
        a = set(np.argsort(-np.asarray(pred_base, float))[:k])
        b = set(np.argsort(-np.asarray(pred_nbr, float))[:k])
        rows.append({"K": k, "overlap": len(a & b) / k,
                     "entered": k - len(a & b), "left": k - len(a & b),
                     "jaccard": len(a & b) / len(a | b)})
        ent, lft = sorted(b - a), sorted(a - b)
        detail[k] = {
            "entered": (labels.iloc[ent].tolist() if labels is not None else ent),
            "left": (labels.iloc[lft].tolist() if labels is not None else lft),
        }
    df = pd.DataFrame(rows)
    print("R2  top-K composition change")
    print(df.round(3).to_string(index=False))
    print("    Low overlap at small K is the decision-relevant number: it says")
    print("    the two methods would send planners to different facilities.")
    return df, detail


# --------------------------------------------------------------------------
# R3 -- is the movement systematic?
# --------------------------------------------------------------------------

def R3_who_moves(delta_rank: np.ndarray, attributes: pd.DataFrame,
                 n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Which place characteristics predict promotion in the ranking?

    Regresses signed displacement (negative = promoted toward higher need) on
    standardised place attributes. This is the substantive health-geography
    result: neighbourhood-awareness does not shuffle facilities at random, it
    systematically promotes facilities of a particular kind.

    Supply attributes a planner would recognise -- local facility density,
    neighbourhood heterogeneity, deprivation proxies, distance to the nearest
    alternative facility. Attributes that are themselves model inputs will
    load trivially; note which are which when reporting.
    """
    A = attributes.select_dtypes(include=[np.number]).copy()
    A = A.loc[:, A.notna().mean() > 0.5]
    A = A.fillna(A.median())
    Z = StandardScaler().fit_transform(A.to_numpy(float))
    # promotion is positive: flip the sign of delta (negative delta = moved up)
    t = -np.asarray(delta_rank, float)
    t = (t - t.mean()) / (t.std() + 1e-12)

    m = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(Z, t)
    rng = np.random.default_rng(seed)
    B = np.empty((n_boot, Z.shape[1]))
    for i in range(n_boot):
        b = rng.integers(0, len(t), len(t))
        B[i] = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(Z[b], t[b]).coef_

    out = pd.DataFrame({
        "attribute": A.columns,
        "coef_promotion": m.coef_,
        "lo": np.percentile(B, 2.5, axis=0),
        "hi": np.percentile(B, 97.5, axis=0),
    })
    out["excludes_zero"] = (out["lo"] > 0) | (out["hi"] < 0)
    out = out.reindex(out["coef_promotion"].abs().sort_values(ascending=False).index)
    print("R3  who gets promoted by neighbourhood-awareness")
    print(out.round(3).to_string(index=False))
    print("    Positive coefficient = facilities high on this attribute move UP")
    print("    the need ranking when neighbourhood context is added.")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# R4 -- head versus body
# --------------------------------------------------------------------------

def R4_head_vs_body(y, pred_base, pred_nbr, ks=(5, 10, 25, 50, 100, 250)
                    ) -> pd.DataFrame:
    """Does neighbourhood-awareness help everywhere, or only away from the top?

    The five-borough analysis found NDCG@10 markedly WORSE for the
    neighbour-aware ranking despite a better global rho. If that reappears,
    it is a finding, not an anomaly: neighbourhood averaging pulls extreme
    facilities toward their local means, which is exactly what you do not
    want at the head of a priority list.

    Report this as a scope condition on the method, and as the basis for a
    two-stage recommendation (own-features for triage of the top, neighbour-
    aware for the body).
    """
    rows = []
    for k in ks:
        if k > len(y):
            continue
        rows.append({
            "K": k,
            "ndcg_own": ndcg_at_k(y, pred_base, k),
            "ndcg_nbr": ndcg_at_k(y, pred_nbr, k),
            "prec_own": precision_at_k(y, pred_base, k),
            "prec_nbr": precision_at_k(y, pred_nbr, k),
        })
    df = pd.DataFrame(rows)
    df["ndcg_delta"] = df["ndcg_nbr"] - df["ndcg_own"]
    df["prec_delta"] = df["prec_nbr"] - df["prec_own"]
    print("R4  head vs body")
    print(df.round(4).to_string(index=False))
    flip = df[(df["ndcg_delta"] < 0)]
    if len(flip):
        print(f"    NDCG is WORSE with neighbourhood context at K = "
              f"{flip['K'].tolist()}")
        print("    Report this in Results as a scope condition. Do not bury it.")
    return df


# --------------------------------------------------------------------------
# R5 -- would decisions change?
# --------------------------------------------------------------------------

def R5_decision_impact(pred_base, pred_nbr, budgets=(5, 10, 20, 50),
                       labels: pd.Series | None = None) -> pd.DataFrame:
    """If a programme could reach N facilities, would it reach the same N?

    This converts the ranking difference into the unit a planner thinks in.
    It is target-independent: it compares two rankings, not either ranking
    against a truth.
    """
    rows = []
    for N in budgets:
        if N > len(pred_base):
            continue
        a = set(np.argsort(-np.asarray(pred_base, float))[:N])
        b = set(np.argsort(-np.asarray(pred_nbr, float))[:N])
        rows.append({"budget_N": N, "same": len(a & b),
                     "different": N - len(a & b),
                     "pct_different": 100 * (N - len(a & b)) / N})
    df = pd.DataFrame(rows)
    print("R5  decision impact")
    print(df.round(1).to_string(index=False))
    print("    'pct_different' is the sentence for the abstract: at a budget of")
    print("    N facilities, the two approaches disagree on this share of them.")
    return df


# --------------------------------------------------------------------------
# R6 -- is the reordering stable, or noise?
# --------------------------------------------------------------------------

def R6_displacement_stability(X, y, xy, k=10, seeds=range(10),
                              n_folds: int = 5) -> pd.DataFrame:
    """Is the same facility displaced the same way under different fold seeds?

    A reordering that changes with the fold seed is not a finding about
    facilities; it is fold noise wearing a map. This is the check that
    licenses naming individual facilities in the paper.
    """
    deltas = []
    for s in seeds:
        st = nas.staircase(X, y, xy, k=k, n_folds=n_folds, seed=s)
        deltas.append(_to_rank(st.pred["Ridge-Spatial"]) -
                      _to_rank(st.pred["Ridge"]))
    D = np.vstack(deltas)
    mean_d, sd_d = D.mean(0), D.std(0)
    consistent = np.abs(mean_d) > 2 * np.maximum(sd_d, 1e-9)
    out = pd.DataFrame({"mean_delta": mean_d, "sd_delta": sd_d,
                        "consistent": consistent})
    print("R6  displacement stability across fold seeds")
    print(f"    facilities with a seed-consistent displacement: "
          f"{consistent.mean():.1%}")
    print(f"    median across-seed SD of displacement: {np.median(sd_d):.1f} positions")
    print("    Only name individual facilities in the paper if they are")
    print("    seed-consistent. Otherwise the case study is fold noise.")
    return out


# --------------------------------------------------------------------------
# one-call report
# --------------------------------------------------------------------------

def ranking_report(X, y, xy, attributes: pd.DataFrame | None = None,
                   labels: pd.Series | None = None, k: int = 10,
                   n_folds: int = 5, seed: int = 0, out_dir: str | None = None
                   ) -> dict:
    """Run R1-R5 in order and return everything. R6 is separate (slower)."""
    st = nas.staircase(X, y, xy, k=k, n_folds=n_folds, seed=seed)
    pb, pn = st.pred["Ridge"], st.pred["Ridge-Spatial"]

    print("=" * 70)
    print("RANKING IMPACT OF NEIGHBOURHOOD-AWARENESS")
    print(f"  rho own-features   = {st.rho['Ridge']:.4f}")
    print(f"  rho neighbour-aware= {st.rho['Ridge-Spatial']:.4f}")
    print("  (agreement with the index is SECONDARY -- it is only as good as")
    print("   the index. The displacement results below are target-independent.)")
    print("=" * 70)

    res = {"staircase": st}
    res["R1"] = R1_displacement(pb, pn, labels=labels); print()
    res["R2"], res["R2_detail"] = R2_topk_churn(pb, pn, labels=labels); print()
    if attributes is not None:
        res["R3"] = R3_who_moves(res["R1"]["delta_array"], attributes)
        print()
    res["R4"] = R4_head_vs_body(y, pb, pn); print()
    res["R5"] = R5_decision_impact(pb, pn, labels=labels)

    if out_dir:
        res["R1"]["table"].to_csv(f"{out_dir}/R1_displacement.csv", index=False)
        res["R2"].to_csv(f"{out_dir}/R2_topk_churn.csv", index=False)
        res["R4"].to_csv(f"{out_dir}/R4_head_vs_body.csv", index=False)
        res["R5"].to_csv(f"{out_dir}/R5_decision_impact.csv", index=False)
        if "R3" in res:
            res["R3"].to_csv(f"{out_dir}/R3_who_moves.csv", index=False)
    return res


# --------------------------------------------------------------------------
def selftest_rank(n: int = 500, d: int = 8, seed: int = 0) -> bool:
    """Synthetic data where neighbourhood context genuinely reorders the list."""
    rng = np.random.default_rng(seed)
    lat = 40.70 + 0.16 * rng.random(n)
    lon = -74.01 + 0.025 * rng.random(n)
    xy = nas.local_xy(lat, lon)
    het = (lat - lat.min()) / (lat.max() - lat.min())
    field = np.column_stack([np.sin(3 * het * np.pi + j) for j in range(d)]) * 2.0
    X = field + rng.normal(0, 1, (n, d)) * (0.05 + 1.6 * het)[:, None]
    idx, w = nas.knn_graph(xy, 10)
    c = X - np.einsum("nk,nkf->nf", w, X[idx])
    y = 2.0 * (c @ rng.normal(0, 1, d)) + 0.8 * (X @ rng.normal(0, 1, d)) \
        + rng.normal(0, 6.0, n)

    attrs = pd.DataFrame({
        "local_heterogeneity": nas.heterogeneity(X, idx),
        "latitude_band": het,
        "feature_0": X[:, 0],
    })
    labels = pd.Series([f"facility_{i:03d}" for i in range(n)])

    print("=" * 70)
    print("SELF-TEST -- ranking impact module")
    print("=" * 70)
    r = ranking_report(X, y, xy, attributes=attrs, labels=labels)
    print()
    R6_displacement_stability(X, y, xy, seeds=range(4))

    ok = (r["R1"]["mean_abs_displacement"] > 1
          and r["R2"]["overlap"].iloc[0] < 1.0)
    print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    selftest_rank()
