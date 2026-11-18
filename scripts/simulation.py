"""Simulation study of bottleneck strata for bounds on probabilities of causation.

Implements PROTOCOL.md (Experiments 1 and 2). Usage:

    python simulation.py --experiment all --workers 8          # the study
    python simulation.py --experiment all --limit 2 --out tmp   # harness test only
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import gammaln

import ibstrata
from ibstrata.bounds.arms import width_loss_certificate
from ibstrata.simulate import make_regime_dgp, make_two_regime_dgp, purity_compositions
from ibstrata.stratify import (
    BottleneckEncoder,
    Design,
    KMeansEncoder,
    ReachPracticeEncoder,
    crossfit,
    imbens_manski,
    stratified_from_memberships,
)

MASTER_SEED = 20261117
N_EVAL = 20_000
BOOT = 200
PI = np.full(3, 1 / 3)
KAPPA_ARGS = dict(pi=PI)


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


def build_dgp(cell: Cell):
    if cell.experiment == 1:
        return make_regime_dgp(k=cell.k, d_signal=12, d_noise=4, separation=0.5,
                               confounding=1.0, type_probs=purity_compositions(cell.k, cell.purity),
                               decoy_clusters=3 if cell.decoy else 0,
                               d_decoy=12 if cell.decoy else 0, seed=MASTER_SEED % 1000)
    return make_two_regime_dgp(d_signal=8, d_noise=2, separation=0.6, purity=cell.purity,
                               decoy_clusters=3 if cell.decoy else 0,
                               d_decoy=12 if cell.decoy else 0, seed=MASTER_SEED % 1000)


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


def make_encoders(cell: Cell, d_total: int, seed: int):
    encs = {
        "oracle": None,
        "bottleneck": BottleneckEncoder(k=cell.k, seed=seed),
        "kmeans": KMeansEncoder(k=cell.k, seed=seed),
    }
    if cell.experiment == 1:
        encs["random"] = None
    else:
        access, practice = feature_groups(cell, d_total)
        encs["reach-practice"] = ReachPracticeEncoder(access, practice, k_a=2, k_m=2, seed=seed)
    return encs


def run_replicate(args) -> list[dict]:
    cell, rep = args
    seq = np.random.SeedSequence([MASTER_SEED, cell.experiment, int(cell.purity * 100),
                                  int(cell.decoy), cell.n, cell.k, rep])
    rng = np.random.default_rng(seq)
    dgp = build_dgp(cell)
    truth = dgp.truth()
    data, regime = dgp.sample(cell.n, rng)
    eval_data, eval_regime = dgp.sample(N_EVAL, rng)
    q_eval = dgp.exact_relevance(eval_data.features)
    lo_c, hi_c = tp_bounds_from_q(q_eval)
    full_lower, full_upper = float(lo_c.mean()), float(hi_c.mean())
    base = dict(experiment=cell.experiment, purity=cell.purity, decoy=cell.decoy, n=cell.n,
                k=cell.k, replicate=rep, true_pns=truth["PNS"], true_pn=truth["PN"],
                true_ps=truth["PS"], full_lower=full_lower, full_upper=full_upper)
    rows = []

    feat_cols = np.arange(data.features.shape[1])

    eval_fold = np.random.default_rng(seq.spawn(2)[1]).integers(0, 2, size=N_EVAL)
    encoders = make_encoders(cell, data.features.shape[1], rep)

    for name, enc in [("plain", "plain")] + list(encoders.items()):
        t0 = time.time()
        row = dict(base, estimator=name, error="")
        try:
            if name == "plain":
                g = np.ones((cell.n, 1))
                g_eval = np.ones((N_EVAL, 1))
            elif name == "random":
                rr = np.random.default_rng(seq.spawn(3)[2])
                g = np.eye(cell.k)[rr.integers(0, cell.k, size=cell.n)]
                g_eval = np.eye(cell.k)[rr.integers(0, cell.k, size=N_EVAL)]
            elif name == "oracle":
                g = np.eye(eval_regime.max() + 1 if cell.experiment == 2 else cell.k)[regime]
                g_eval = np.eye(g.shape[1])[eval_regime]
            else:
                if isinstance(enc, BottleneckEncoder):
                    enc.feature_columns = feat_cols
                cf, fold_of = crossfit(data, enc, folds=2, seed=rep)
                g = cf.memberships(data, fold_of)
                g_eval = cf.memberships(eval_data, eval_fold)
            est, bad, k_used = stratified_from_memberships(data, g, alpha=0.5, min_weight=5.0)
            boots = _bootstrap_fixed(data, g, seq.spawn(4)[3])
            se_lo, se_hi = boots.std(axis=0, ddof=1)
            ci = imbens_manski(est.pns.lower, est.pns.upper, se_lo, se_hi)
            pop_lower, pop_upper, info = population_of_stratifier(
                q_eval, apply_pooling(g, g_eval, 5.0))
            cert = width_loss_certificate(info, PI)
            row.update(
                pns_lower=est.pns.lower, pns_upper=est.pns.upper,
                pn_lower=est.pn.lower, pn_upper=est.pn.upper,
                ps_lower=est.ps.lower, ps_upper=est.ps.upper,
                se_lower=se_lo, se_upper=se_hi, ci_lower=ci[0], ci_upper=ci[1],
                strata=k_used, incompatible=bad,
                pop_lower=pop_lower, pop_upper=pop_upper, info_c_v_given_t=info,
                certificate=cert,
                loss=(full_lower - pop_lower) + (pop_upper - full_upper),
            )
            if cell.experiment == 2 and name in ("bottleneck", "kmeans", "reach-practice"):
                row["ami_combined"] = adjusted_mutual_information(g.argmax(axis=1), regime)
                if name == "reach-practice":
                    ama, amp = [], []
                    for f, fitted in enumerate(cf.encoders):
                        sel = np.flatnonzero(fold_of == f)
                        ga, gm = fitted.regime_memberships(data.subset(sel))
                        ama.append(adjusted_mutual_information(ga.argmax(1), regime[sel] // 2))
                        amp.append(adjusted_mutual_information(gm.argmax(1), regime[sel] % 2))
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
# Aggregation


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "purity", "decoy", "n", "k", "estimator"]
    ok = df[df["error"] == ""].copy()
    ok["width"] = ok["pns_upper"] - ok["pns_lower"]
    ok["covers"] = (ok["pns_lower"] <= ok["true_pns"]) & (ok["true_pns"] <= ok["pns_upper"])
    ok["ci_covers"] = (ok["ci_lower"] <= ok["true_pns"]) & (ok["true_pns"] <= ok["ci_upper"])
    ok["valid"] = ((ok["pop_lower"] <= ok["true_pns"] + 1e-3)
                   & (ok["true_pns"] <= ok["pop_upper"] + 1e-3))
    ok["err_lower"] = ok["pns_lower"] - ok["pop_lower"]
    ok["err_upper"] = ok["pns_upper"] - ok["pop_upper"]
    ok["pop_width"] = ok["pop_upper"] - ok["pop_lower"]
    ok["cert_ratio"] = np.where(ok["certificate"] > 1e-9, ok["loss"] / ok["certificate"], np.nan)
    g = ok.groupby(keys)
    out = g.agg(
        replicates=("replicate", "size"),
        width=("width", "mean"),
        pop_width=("pop_width", "mean"),
        err_lower=("err_lower", "mean"),
        err_upper=("err_upper", "mean"),
        covers=("covers", "mean"),
        ci_covers=("ci_covers", "mean"),
        valid=("valid", "mean"),
        cert_ratio_max=("cert_ratio", "max"),
        cert_ratio_median=("cert_ratio", "median"),
        strata=("strata", "mean"),
        incompatible=("incompatible", "mean"),
        seconds=("seconds", "mean"),
    ).reset_index()
    out["ci_covers_se"] = np.sqrt(out["ci_covers"] * (1 - out["ci_covers"]) / out["replicates"])
    fails = df[df["error"] != ""].groupby(keys).size().rename("failures")
    out = out.merge(fails, on=keys, how="left").fillna({"failures": 0})
    for col in ("ami_combined", "ami_access", "ami_practice"):
        if col in ok:
            out = out.merge(g[col].mean().rename(col).reset_index(), on=keys, how="left")
    return out


def environment(out: Path) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[3]).stdout.strip()
    except OSError:
        commit = ""
    info = dict(python=sys.version, platform=platform.platform(), ibstrata=ibstrata.__version__,
                numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                commit=commit, master_seed=MASTER_SEED, n_eval=N_EVAL, bootstrap=BOOT,
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "environment.json").write_text(json.dumps(info, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["1", "2", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="replicates per cell (harness test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    environment(out)
    plans = []
    if args.experiment in ("1", "all"):
        plans.append(("exp1", cells_exp1(), args.limit or 200))
    if args.experiment in ("2", "all"):
        plans.append(("exp2", cells_exp2(), args.limit or 100))
    for label, cells, reps in plans:
        tasks = [(c, r) for c in cells for r in range(reps)]
        rows = []
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            for i, res in enumerate(ex.map(run_replicate, tasks, chunksize=2)):
                rows.extend(res)
                if (i + 1) % 50 == 0:
                    print(f"{label}: {i + 1}/{len(tasks)} replicates, {time.time() - t0:.0f}s",
                          flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(out / "tables" / f"{label}-replicates.csv", index=False)
        summarize(df).to_csv(out / "tables" / f"{label}-cells.csv", index=False)
    info = json.loads((out / "environment.json").read_text())
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "environment.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
