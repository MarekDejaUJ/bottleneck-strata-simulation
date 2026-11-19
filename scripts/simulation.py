"""Simulation study of bottleneck strata for bounds on probabilities of causation.

Implements PROTOCOL.md (Experiments 1, 2 and 3). Usage:

    python simulation.py --experiment all --workers 8            # the study
    python simulation.py --experiment all --limit 2 --out tmp    # harness test only

Every cell is written to ``<out>/parts`` when its replicates are complete, and a rerun
with the same arguments continues with the cells that are missing.
"""

from __future__ import annotations

import argparse
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

from ibstrata.bounds.arms import width_loss_certificate
from ibstrata.simulate import make_regime_dgp, make_two_regime_dgp, purity_compositions
from ibstrata.stratify import (
    BottleneckEncoder,
    Design,
    EffectQuantileEncoder,
    FittedKMeansEncoder,
    KMeansEncoder,
    ReachPracticeEncoder,
    TermPairEncoder,
    _fold_labels,
    best_term_pairs,
    crossfit,
    imbens_manski,
    pair_widths,
    selected_term_bounds,
    stratified_from_memberships,
)

MASTER_SEED = 20261117
N_EVAL = 20_000
BOOT = 200
PI = np.full(3, 1 / 3)

# Experiment 3: response-type compositions (never, helped, hurt, always) of four latent
# profiles with effects of both signs, and the logits of choosing x by profile.
EXP3_TYPES = ((0.08, 0.66, 0.05, 0.21),
              (0.25, 0.05, 0.52, 0.18),
              (0.58, 0.10, 0.06, 0.26),
              (0.12, 0.20, 0.10, 0.58))
EXP3_CHOICE = (0.7, -0.6, -0.1, 0.3)

# Fixed positions of the estimators in the seed stream of the added computations.
SLOTS = ("plain", "oracle", "bottleneck", "kmeans", "random", "reach-practice",
         "kmeans-weighted", "bottleneck-logistic", "kmeans-fitted", "effect-quantiles",
         "bottleneck-unweighted", "term-pairs")
LEARNED = ("bottleneck", "kmeans", "reach-practice", "kmeans-weighted", "bottleneck-logistic",
           "kmeans-fitted", "effect-quantiles", "bottleneck-unweighted", "term-pairs")


# ---------------------------------------------------------------------------
# Cells


@dataclass(frozen=True)
class Cell:
    experiment: int
    purity: float
    decoy: bool
    n: int
    k: int


def cells_exp1() -> list[Cell]:
    return [Cell(1, p, d, n, k) for p in (0.0, 0.6, 0.85) for d in (False, True)
            for n in (300, 600, 900, 1500) for k in (2, 3)]


def cells_exp2() -> list[Cell]:
    return [Cell(2, 0.85, d, n, 4) for d in (False, True) for n in (600, 900, 1500)]


def cells_exp3() -> list[Cell]:
    return [Cell(3, 0.0, d, n, 4) for d in (False, True) for n in (600, 1500)]


def ablation_cell(cell: Cell) -> bool:
    """Cells that run the bottleneck without relevance weights."""
    return cell.experiment == 3 or (cell.experiment == 1 and cell.k == 3 and cell.purity > 0)


def build_dgp(cell: Cell, master: int = MASTER_SEED, rep: int = 0):
    if cell.experiment == 1:
        return make_regime_dgp(k=cell.k, d_signal=12, d_noise=4, separation=0.5,
                               confounding=1.0, type_probs=purity_compositions(cell.k, cell.purity),
                               decoy_clusters=3 if cell.decoy else 0,
                               d_decoy=12 if cell.decoy else 0, seed=master % 1000)
    if cell.experiment == 2:
        return make_two_regime_dgp(d_signal=8, d_noise=2, separation=0.6, purity=cell.purity,
                                   decoy_clusters=3 if cell.decoy else 0,
                                   d_decoy=12 if cell.decoy else 0, seed=master % 1000)
    proto_seed = int(np.random.SeedSequence([master, 3, rep]).generate_state(1)[0])
    return make_regime_dgp(k=4, d_signal=12, d_noise=4, separation=0.5, confounding=1.0,
                           type_probs=EXP3_TYPES, choice_base=EXP3_CHOICE,
                           decoy_clusters=3 if cell.decoy else 0,
                           d_decoy=12 if cell.decoy else 0, seed=proto_seed)


def feature_groups(cell: Cell, d_total: int):
    """Access and practice columns for Experiment 2 (decoy columns join the access group)."""
    access = list(range(0, 10))
    practice = list(range(10, 20))
    access += list(range(20, d_total))
    return np.array(access), np.array(practice)


# ---------------------------------------------------------------------------
# Population quantities


def tp_bounds_from_q(q: np.ndarray, pi: np.ndarray = PI) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Tian-Pearl PNS bounds for rows of arm-augmented distributions
    (x = 1, y = 1; layout of ibstrata.bounds.arms)."""
    e0 = q[:, 0:2] / pi[0]
    e1 = q[:, 2:4] / pi[1]
    ob = q[:, 4:8] / pi[2]
    p_yx, p_yxp, p_ypxp = e1[:, 1], e0[:, 1], e0[:, 0]
    pxy, pxyp, pxpy, pxpyp = ob[:, 3], ob[:, 2], ob[:, 1], ob[:, 0]
    py = pxy + pxpy
    lo = np.maximum.reduce([np.zeros_like(py), p_yx - p_yxp, py - p_yxp, p_yx - py])
    hi = np.minimum.reduce([p_yx, p_ypxp, pxy + pxpyp, p_yx - p_yxp + pxyp + pxpy])
    return lo, hi


def binding_patterns(q: np.ndarray, pi: np.ndarray = PI, share: float = 0.01) -> int:
    """Number of pairs (binding lower term, binding upper term) of the Tian-Pearl PNS
    bounds that hold at least ``share`` of the rows of ``q``."""
    e0 = q[:, 0:2] / pi[0]
    e1 = q[:, 2:4] / pi[1]
    ob = q[:, 4:8] / pi[2]
    p_yx, p_yxp, p_ypxp = e1[:, 1], e0[:, 1], e0[:, 0]
    pxy, pxyp, pxpy, pxpyp = ob[:, 3], ob[:, 2], ob[:, 1], ob[:, 0]
    py = pxy + pxpy
    lo = np.argmax(np.stack([np.zeros_like(py), p_yx - p_yxp, py - p_yxp, p_yx - py]), axis=0)
    hi = np.argmin(np.stack([p_yx, p_ypxp, pxy + pxpyp, p_yx - p_yxp + pxyp + pxpy]), axis=0)
    counts = np.bincount(lo * 4 + hi, minlength=16) / q.shape[0]
    return int((counts >= share).sum())


def numerators_from_q(q: np.ndarray, pi: np.ndarray = PI):
    """Tian-Pearl bounds on the numerators of PN and PS, P(y'_{x'}, x, y) and
    P(y_x, x', y'), and the denominators P(x, y) and P(x', y'), for rows of ``q``."""
    e0 = q[:, 0:2] / pi[0]
    e1 = q[:, 2:4] / pi[1]
    ob = q[:, 4:8] / pi[2]
    p_yx, p_yxp, p_ypxp = e1[:, 1], e0[:, 1], e0[:, 0]
    pxy, pxpy, pxpyp = ob[:, 3], ob[:, 1], ob[:, 0]
    py = pxy + pxpy
    pn_lo = np.maximum(0.0, py - p_yxp)
    pn_hi = np.minimum(pxy, p_ypxp - pxpyp)
    ps_lo = np.maximum(0.0, p_yx - py)
    ps_hi = np.minimum(pxpyp, p_yx - pxy)
    return pn_lo, pn_hi, ps_lo, ps_hi, pxy, pxpyp


def population_of_stratifier(q_eval: np.ndarray, g_eval: np.ndarray):
    """Population PNS bounds of a stratifier with memberships ``g_eval`` on the evaluation
    sample, and I(C; V | T) on that sample."""
    p_t = g_eval.mean(axis=0)
    keep = p_t > 0
    g = g_eval[:, keep]
    p_t = p_t[keep]
    q_t = (g.T @ q_eval) / (g.sum(axis=0)[:, None])
    lo_t, hi_t = tp_bounds_from_q(q_t)
    lower = float(p_t @ lo_t)
    upper = float(p_t @ hi_t)
    # I(C; V | T) with P(c) uniform over the evaluation sample
    logq = np.log(np.clip(q_eval, 1e-300, None))
    logqt = np.log(np.clip(q_t, 1e-300, None))
    h_c = (q_eval * logq).sum(axis=1)  # sum_v q_c log q_c
    cross = q_eval @ logqt.T  # sum_v q_c log q_t
    kl = h_c[:, None] - cross
    info = float((g * kl).sum() / q_eval.shape[0])
    return lower, upper, max(info, 0.0)


def population_pn_ps(q_eval: np.ndarray, g_eval: np.ndarray | None) -> dict[str, float]:
    """Population bounds on PN and PS of a stratifier (``None``: the full covariates):
    the bounds on the numerators averaged over the strata, divided by the population
    probabilities of the conditioning cells."""
    if g_eval is None:
        rows, p_t = q_eval, np.full(q_eval.shape[0], 1.0 / q_eval.shape[0])
    else:
        p_t = g_eval.mean(axis=0)
        keep = p_t > 0
        g = g_eval[:, keep]
        p_t = p_t[keep]
        rows = (g.T @ q_eval) / (g.sum(axis=0)[:, None])
    pn_lo, pn_hi, ps_lo, ps_hi, pxy, pxpyp = numerators_from_q(rows)
    d_pn, d_ps = float(p_t @ pxy), float(p_t @ pxpyp)
    return dict(pn_lower=max(0.0, float(p_t @ pn_lo) / d_pn),
                pn_upper=min(1.0, float(p_t @ pn_hi) / d_pn),
                ps_lower=max(0.0, float(p_t @ ps_lo) / d_ps),
                ps_upper=min(1.0, float(p_t @ ps_hi) / d_ps))


# ---------------------------------------------------------------------------
# Adjusted mutual information (Vinh, Epps and Bailey, 2010)


def adjusted_mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    n = a.size
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    r, c = table.sum(1), table.sum(0)

    def h(x):
        p = x[x > 0] / n
        return float(-(p * np.log(p)).sum())

    nz = table > 0
    mi = float((table[nz] / n * np.log(table[nz] * n / np.outer(r, c)[nz])).sum())
    emi = 0.0
    for ai in r:
        for bj in c:
            lo, hi = int(max(1, ai + bj - n)), int(min(ai, bj))
            for nij in range(lo, hi + 1):
                term = nij / n * np.log(nij * n / (ai * bj))
                logp = (gammaln(ai + 1) + gammaln(bj + 1) + gammaln(n - ai + 1)
                        + gammaln(n - bj + 1) - gammaln(n + 1) - gammaln(nij + 1)
                        - gammaln(ai - nij + 1) - gammaln(bj - nij + 1)
                        - gammaln(n - ai - bj + nij + 1))
                emi += term * np.exp(logp)
    denom = 0.5 * (h(r) + h(c)) - emi
    return float((mi - emi) / denom) if abs(denom) > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# One replicate


def first_encoders(cell: Cell, d_total: int, seed: int):
    """Stratifiers of the first block, in the order of their seed stream."""
    encs = {
        "oracle": None,
        "bottleneck": BottleneckEncoder(k=cell.k, seed=seed),
        "kmeans": KMeansEncoder(k=cell.k, seed=seed),
    }
    if cell.experiment == 2:
        access, practice = feature_groups(cell, d_total)
        encs["reach-practice"] = ReachPracticeEncoder(access, practice, k_a=2, k_m=2, seed=seed)
    else:
        encs["random"] = None
    return encs


def second_encoders(cell: Cell, seed: int):
    """Stratifiers of the second block."""
    encs = {
        "kmeans-weighted": KMeansEncoder(k=cell.k, seed=seed, weighted=True),
        "bottleneck-logistic": BottleneckEncoder(k=cell.k, seed=seed, relevance="logistic"),
        "kmeans-fitted": FittedKMeansEncoder(k=cell.k, seed=seed),
        "effect-quantiles": EffectQuantileEncoder(k=cell.k),
        "term-pairs": TermPairEncoder(k=cell.k),
    }
    if ablation_cell(cell):
        encs["bottleneck-unweighted"] = BottleneckEncoder(k=cell.k, seed=seed, weighted=False)
    return encs


def run_replicate(args) -> list[dict]:
    cell, rep, master = args
    key = [master, cell.experiment, int(cell.purity * 100), int(cell.decoy), cell.n, cell.k, rep]
    seq = np.random.SeedSequence(key)
    added = np.random.SeedSequence(key + [1]).spawn(2 * len(SLOTS))
    rng = np.random.default_rng(seq)
    dgp = build_dgp(cell, master, rep)
    truth = dgp.truth()
    data, regime = dgp.sample(cell.n, rng)
    eval_data, eval_regime = dgp.sample(N_EVAL, rng)
    q_eval = dgp.exact_relevance(eval_data.features)
    lo_c, hi_c = tp_bounds_from_q(q_eval)
    full_lower, full_upper = float(lo_c.mean()), float(hi_c.mean())
    full_pn_ps = population_pn_ps(q_eval, None)
    exact = dgp.oracle_bounds()
    w_eval = pair_widths(q_eval, PI)
    narrowest = {f"best_width_{j}": best_term_pairs(w_eval, j)[1] for j in (2, 3, 4)}
    base = dict(experiment=cell.experiment, purity=cell.purity, decoy=cell.decoy, n=cell.n,
                k=cell.k, replicate=rep, true_pns=truth["PNS"], true_pn=truth["PN"],
                true_ps=truth["PS"], full_lower=full_lower, full_upper=full_upper,
                full_pn_lower=full_pn_ps["pn_lower"], full_pn_upper=full_pn_ps["pn_upper"],
                full_ps_lower=full_pn_ps["ps_lower"], full_ps_upper=full_pn_ps["ps_upper"],
                oracle_lower=exact.pns.lower, oracle_upper=exact.pns.upper,
                binding_patterns=binding_patterns(q_eval), **narrowest)
    rows = []

    feat_cols = np.arange(data.features.shape[1])
    eval_fold = np.random.default_rng(seq.spawn(2)[1]).integers(0, 2, size=N_EVAL)
    fixed_fold = _fold_labels(data, 2, np.random.default_rng(rep))
    n_profiles = int(eval_regime.max()) + 1 if cell.experiment == 2 else cell.k
    first = [("plain", "plain")] + list(first_encoders(cell, data.features.shape[1], rep).items())
    second = list(second_encoders(cell, rep).items())

    for name, enc in first + second:
        t0 = time.time()
        slot = SLOTS.index(name)
        in_first = name in dict(first)
        row = dict(base, estimator=name, error="")
        try:
            cf, fold_of, train_g = None, fixed_fold, None
            if name == "plain":
                g = np.ones((cell.n, 1))
                g_eval = np.ones((N_EVAL, 1))
            elif name == "random":
                rr = np.random.default_rng(seq.spawn(3)[2])
                g = np.eye(cell.k)[rr.integers(0, cell.k, size=cell.n)]
                g_eval = np.eye(cell.k)[rr.integers(0, cell.k, size=N_EVAL)]
            elif name == "oracle":
                g = np.eye(n_profiles)[regime]
                g_eval = np.eye(n_profiles)[eval_regime]
            else:
                if name == "bottleneck":
                    enc.feature_columns = feat_cols
                cf, fold_of = crossfit(data, enc, folds=2, seed=rep)
                g = cf.memberships(data, fold_of)
                g_eval = cf.memberships(eval_data, eval_fold)
                train_g = cf.training_memberships(data, fold_of)
            est, bad, k_used = stratified_from_memberships(data, g, alpha=0.5, min_weight=5.0)
            boots = _bootstrap_fixed(data, g, seq.spawn(4)[3] if in_first else added[2 * slot])
            se_lo, se_hi = boots.std(axis=0, ddof=1)
            ci = imbens_manski(est.pns.lower, est.pns.upper, se_lo, se_hi)
            sel = selected_term_bounds(data, g, fold_of, train_g, boot=BOOT,
                                       seed=added[2 * slot + 1])
            pooled_eval = apply_pooling(g, g_eval, 5.0)
            if name == "oracle":
                pop_lower, pop_upper = exact.pns.lower, exact.pns.upper
                info = cert = loss = float("nan")
                pop_pn_ps = dict(pn_lower=exact.pn.lower, pn_upper=exact.pn.upper,
                                 ps_lower=exact.ps.lower, ps_upper=exact.ps.upper)
            else:
                pop_lower, pop_upper, info = population_of_stratifier(q_eval, pooled_eval)
                cert = width_loss_certificate(info, PI)
                loss = (full_lower - pop_lower) + (pop_upper - full_upper)
                pop_pn_ps = population_pn_ps(q_eval, pooled_eval)
            row.update(
                pns_lower=est.pns.lower, pns_upper=est.pns.upper,
                pn_lower=est.pn.lower, pn_upper=est.pn.upper,
                ps_lower=est.ps.lower, ps_upper=est.ps.upper,
                se_lower=se_lo, se_upper=se_hi, ci_lower=ci[0], ci_upper=ci[1],
                strata=k_used, incompatible=bad,
                sel_lower=sel.lower, sel_upper=sel.upper,
                sel_se_lower=sel.se_lower, sel_se_upper=sel.se_upper,
                sel_ci_lower=sel.ci_lower, sel_ci_upper=sel.ci_upper,
                sel_avg_ci_lower=sel.avg_ci_lower, sel_avg_ci_upper=sel.avg_ci_upper,
                sel_strata=sel.strata,
                pop_lower=pop_lower, pop_upper=pop_upper,
                pop_pn_lower=pop_pn_ps["pn_lower"], pop_pn_upper=pop_pn_ps["pn_upper"],
                pop_ps_lower=pop_pn_ps["ps_lower"], pop_ps_upper=pop_pn_ps["ps_upper"],
                info_c_v_given_t=info, information_bound=cert, loss=loss,
            )
            if cell.experiment == 2 and name in LEARNED:
                row["ami_combined"] = adjusted_mutual_information(g.argmax(axis=1), regime)
                if name == "reach-practice":
                    ama, amp = [], []
                    for f, fitted in enumerate(cf.encoders):
                        sel_f = np.flatnonzero(fold_of == f)
                        ga, gm = fitted.regime_memberships(data.subset(sel_f))
                        ama.append(adjusted_mutual_information(ga.argmax(1), regime[sel_f] // 2))
                        amp.append(adjusted_mutual_information(gm.argmax(1), regime[sel_f] % 2))
                        row[f"k_a_fold{f}"] = fitted._sol.enc_a.shape[1]
                        row[f"k_m_fold{f}"] = fitted._sol.enc_m.shape[1]
                    row["ami_access"] = float(np.mean(ama))
                    row["ami_practice"] = float(np.mean(amp))
        except Exception as exc:  # recorded, never rerun (protocol section 4)
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc(limit=3)
        row["seconds"] = time.time() - t0
        rows.append(row)
    return rows


def apply_pooling(g: np.ndarray, g_eval: np.ndarray, min_weight: float) -> np.ndarray:
    """Merge the columns of ``g_eval`` exactly as ``pool_small_strata`` merges ``g``."""
    size = g.sum(axis=0)
    small = size < min_weight
    if small.sum() == 0:
        return g_eval
    big_cols = np.flatnonzero(~small)
    if big_cols.size == 0:
        return np.ones((g_eval.shape[0], 1))
    big = g_eval[:, big_cols].copy()
    pooled = g_eval[:, small].sum(axis=1)
    if g[:, small].sum() < min_weight:
        big[:, int(g[:, big_cols].sum(axis=0).argmax())] += pooled
        return big
    return np.concatenate([big, pooled[:, None]], axis=1)


def _bootstrap_fixed(data: Design, g: np.ndarray, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arms = [np.flatnonzero(data.arm == a) for a in np.unique(data.arm)]
    out = np.empty((BOOT, 2))
    for b in range(BOOT):
        idx = np.concatenate([rng.choice(a, size=a.size, replace=True) for a in arms])
        est, _, _ = stratified_from_memberships(data.subset(idx), g[idx], alpha=0.5,
                                                min_weight=5.0)
        out[b] = (est.pns.lower, est.pns.upper)
    return out


# ---------------------------------------------------------------------------
# Run


def cell_tag(cell: Cell) -> str:
    return (f"exp{cell.experiment}-p{int(cell.purity * 100):02d}-d{int(cell.decoy)}"
            f"-n{cell.n}-k{cell.k}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["1", "2", "3", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="replicates per cell (harness test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=MASTER_SEED,
                    help="master seed (harness tests use a seed other than the study's)")
    ap.add_argument("--only", default="",
                    help="comma-separated parts of cell tags; other cells are skipped (harness test)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s]
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    plans = []
    if args.experiment in ("1", "all"):
        plans.append(("exp1", cells_exp1(), args.limit or 200))
    if args.experiment in ("2", "all"):
        plans.append(("exp2", cells_exp2(), args.limit or 200))
    if args.experiment in ("3", "all"):
        plans.append(("exp3", cells_exp3(), args.limit or 200))
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for label, cells, reps in plans:
            cells = [c for c in cells if not only or any(s in cell_tag(c) for s in only)]
            if not cells:
                continue
            frames = []
            for i, cell in enumerate(cells):
                part = out / "parts" / f"{cell_tag(cell)}.csv"
                if part.exists():
                    df = pd.read_csv(part, dtype={"error": str})
                    if df["replicate"].nunique() == reps:
                        frames.append(df)
                        continue
                tasks = [(cell, r, args.seed) for r in range(reps)]
                rows = [row for res in ex.map(run_replicate, tasks, chunksize=2) for row in res]
                df = pd.DataFrame(rows)
                df.to_csv(part, index=False)
                frames.append(pd.read_csv(part, dtype={"error": str}))
                print(f"{label}: cell {i + 1}/{len(cells)} ({cell_tag(cell)}), "
                      f"{time.time() - t0:.0f}s", flush=True)
            pd.concat(frames, ignore_index=True).to_csv(
                out / "tables" / f"{label}-replicates.csv", index=False)


if __name__ == "__main__":
    main()
