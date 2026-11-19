"""Tables for the questions of the protocol.

Reads the replicate tables written by ``simulation.py`` and writes, into
``results/tables``: ``exp1-cells.csv``, ``exp2-cells.csv``, ``exp3-cells.csv`` and
``q1-validity.csv`` to ``q6-mixed-sign.csv``.

Usage:
    python summarize.py [results folder]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"
TABLES = RESULTS / "tables"
CELL = ["experiment", "purity", "decoy", "n", "k"]
PAIRS = [("bottleneck", "kmeans-weighted"), ("bottleneck", "bottleneck-unweighted"),
         ("bottleneck", "bottleneck-logistic"), ("bottleneck-logistic", "kmeans-fitted"),
         ("bottleneck-logistic", "effect-quantiles"), ("bottleneck-logistic", "term-pairs"),
         ("reach-practice", "bottleneck"),
         ("reach-practice", "bottleneck-logistic"), ("reach-practice", "kmeans-fitted")]
TOL = 1e-3  # Monte Carlo error of the evaluation sample


def load(exp: int) -> pd.DataFrame:
    df = pd.read_csv(TABLES / f"exp{exp}-replicates.csv", dtype={"error": str})
    df["error"] = df["error"].fillna("")
    ok = df[df["error"] == ""].copy()
    ok.attrs["failures"] = int((df["error"] != "").sum())
    oracle = ok["estimator"] == "oracle"
    # ends of the identified set: full covariates; for the true profiles, their exact bounds
    ok["end_lower"] = np.where(oracle, ok["oracle_lower"], ok["full_lower"])
    ok["end_upper"] = np.where(oracle, ok["oracle_upper"], ok["full_upper"])
    ok["pop_width"] = ok["pop_upper"] - ok["pop_lower"]
    ok["full_width"] = ok["full_upper"] - ok["full_lower"]
    ok["oracle_width"] = ok["oracle_upper"] - ok["oracle_lower"]
    # narrowest population width that as many strata as the cell uses can give
    ok["best_width"] = np.select([ok["k"] == 2, ok["k"] == 3], [ok["best_width_2"], ok["best_width_3"]],
                                 ok["best_width_4"])
    ok["pop_pn_width"] = ok["pop_pn_upper"] - ok["pop_pn_lower"]
    ok["pop_ps_width"] = ok["pop_ps_upper"] - ok["pop_ps_lower"]
    ok["full_pn_width"] = ok["full_pn_upper"] - ok["full_pn_lower"]
    ok["full_ps_width"] = ok["full_ps_upper"] - ok["full_ps_lower"]
    for q in ("pns", "pn", "ps"):
        lo = "pop_lower" if q == "pns" else f"pop_{q}_lower"
        hi = "pop_upper" if q == "pns" else f"pop_{q}_upper"
        ok[f"valid_{q}"] = (ok[lo] <= ok[f"true_{q}"] + TOL) & (ok[f"true_{q}"] <= ok[hi] + TOL)
    for tag, lo, hi, cl, ch in (("plug", "pns_lower", "pns_upper", "ci_lower", "ci_upper"),
                                ("sel", "sel_lower", "sel_upper", "sel_ci_lower", "sel_ci_upper"),
                                ("selavg", "sel_lower", "sel_upper", "sel_avg_ci_lower",
                                 "sel_avg_ci_upper")):
        ok[f"{tag}_width"] = ok[hi] - ok[lo]
        ok[f"{tag}_err_lower"] = ok[lo] - ok["pop_lower"]
        ok[f"{tag}_err_upper"] = ok[hi] - ok["pop_upper"]
        ok[f"{tag}_contains"] = (ok[lo] <= ok["true_pns"]) & (ok["true_pns"] <= ok[hi])
        ok[f"{tag}_ci_width"] = (ok[ch] - ok[cl]).clip(lower=0.0)
        ok[f"{tag}_covers_truth"] = (ok[cl] <= ok["true_pns"]) & (ok["true_pns"] <= ok[ch])
        ok[f"{tag}_covers_lower_end"] = (ok[cl] <= ok["end_lower"]) & (ok["end_lower"] <= ok[ch])
        ok[f"{tag}_covers_upper_end"] = (ok[cl] <= ok["end_upper"]) & (ok["end_upper"] <= ok[ch])
    ok["bound_ratio"] = np.where(ok["information_bound"] > 1e-9,
                                 ok["loss"] / ok["information_bound"], np.nan)
    return ok


MEANS = ["pop_lower", "pop_upper", "pop_width", "full_lower", "full_upper", "full_width",
         "oracle_width", "best_width", "best_width_2", "best_width_3", "best_width_4",
         "pop_pn_width", "pop_ps_width", "full_pn_width", "full_ps_width",
         "valid_pns", "valid_pn", "valid_ps", "binding_patterns", "strata", "sel_strata",
         "incompatible", "info_c_v_given_t", "information_bound", "loss", "seconds"] + [
    f"{tag}_{m}" for tag in ("plug", "sel")
    for m in ("width", "err_lower", "err_upper", "contains", "ci_width", "covers_truth",
              "covers_lower_end", "covers_upper_end")] + [
    f"selavg_{m}" for m in ("ci_width", "covers_truth", "covers_lower_end", "covers_upper_end")]


def per_cell(ok: pd.DataFrame) -> pd.DataFrame:
    g = ok.groupby(CELL + ["estimator"])
    out = g[MEANS].mean()
    out.insert(0, "replicates", g.size())
    for col in ("pop_width", "plug_width", "sel_width"):
        out[col + "_se"] = g[col].std(ddof=1) / np.sqrt(g.size())
    out = out.reset_index()
    plain = out[out["estimator"] == "plain"].set_index(CELL)["pop_width"].rename("plain_pop_width")
    out = out.join(plain, on=CELL)
    for ref, name in (("full_width", "share_of_full_gain"), ("oracle_width", "share_of_oracle_gain"),
                      ("best_width", "share_of_best_gain")):
        span = out["plain_pop_width"] - out[ref]
        out[name] = np.where(span > 1e-6, (out["plain_pop_width"] - out["pop_width"]) / span, np.nan)
    for col in ("ami_combined", "ami_access", "ami_practice"):
        if col in ok and ok[col].notna().any():
            out = out.merge(g[col].mean().reset_index(), on=CELL + ["estimator"], how="left")
    return out


def paired(ok: pd.DataFrame) -> pd.DataFrame:
    """Mean paired differences of the population widths of two stratifiers per cell."""
    wide = ok.pivot_table(index=CELL + ["replicate"], columns="estimator", values="pop_width")
    rows = []
    for a, b in PAIRS:
        if a not in wide or b not in wide:
            continue
        d = (wide[a] - wide[b]).dropna()
        for cell, v in d.groupby(level=list(range(len(CELL)))):
            rows.append(dict(zip(CELL, cell), first=a, second=b, replicates=len(v),
                             difference=v.mean(), se=v.std(ddof=1) / np.sqrt(len(v))))
    return pd.DataFrame(rows)


def main() -> None:
    exps = [e for e in (1, 2, 3) if (TABLES / f"exp{e}-replicates.csv").exists()]
    data = {e: load(e) for e in exps}
    cells = {e: per_cell(data[e]) for e in exps}
    for e in exps:
        cells[e].to_csv(TABLES / f"exp{e}-cells.csv", index=False)
    both = pd.concat(data.values(), ignore_index=True)
    allc = pd.concat(cells.values(), ignore_index=True)

    q1 = both.groupby(["experiment", "estimator"]).agg(
        replicates=("valid_pns", "size"), valid_pns=("valid_pns", "mean"),
        valid_pn=("valid_pn", "mean"), valid_ps=("valid_ps", "mean")).reset_index()
    q1.to_csv(TABLES / "q1-validity.csv", index=False)

    q2 = allc[CELL + ["estimator", "replicates", "pop_lower", "pop_upper", "pop_width",
                      "pop_width_se", "full_width", "oracle_width", "best_width",
                      "best_width_2", "best_width_3", "best_width_4", "share_of_full_gain",
                      "share_of_oracle_gain", "share_of_best_gain", "pop_pn_width", "full_pn_width",
                      "pop_ps_width", "full_ps_width", "binding_patterns"]]
    q2.to_csv(TABLES / "q2-widths.csv", index=False)
    pd.concat([paired(data[e]) for e in exps], ignore_index=True).to_csv(
        TABLES / "q2-paired.csv", index=False)

    q3_cols = [f"{tag}_{m}" for tag in ("plug", "sel")
               for m in ("width", "width_se", "err_lower", "err_upper", "contains", "ci_width",
                         "covers_truth", "covers_lower_end", "covers_upper_end")] + [
        f"selavg_{m}" for m in ("ci_width", "covers_truth", "covers_lower_end", "covers_upper_end")]
    allc[CELL + ["estimator", "replicates", "pop_width"] + q3_cols].to_csv(
        TABLES / "q3-finite-sample.csv", index=False)
    # coverage by group of cells: strata can narrow the bounds or not
    gain = (allc[allc["estimator"] == "plain"].set_index(CELL)["pop_width"]
            - allc[allc["estimator"] == "plain"].set_index(CELL)["full_width"])
    both = both.join((gain > 1e-3).rename("narrowing_available"), on=CELL)
    cov = [f"{tag}_{m}" for tag in ("plug", "sel")
           for m in ("contains", "covers_truth", "covers_lower_end", "covers_upper_end")] + [
        f"selavg_{m}" for m in ("covers_truth", "covers_lower_end", "covers_upper_end")]
    q3g = both.groupby(["experiment", "narrowing_available", "estimator", "n"]).agg(
        replicates=("replicate", "size"), **{c: (c, "mean") for c in cov}).reset_index()
    q3g.to_csv(TABLES / "q3-coverage.csv", index=False)

    lr = both[both["bound_ratio"].notna()].copy()
    plain_w = both[both["estimator"] == "plain"].set_index(CELL + ["replicate"])["pop_width"]
    lr = lr.join(plain_w.rename("plain_w"), on=CELL + ["replicate"])
    lr["largest_loss"] = lr["plain_w"] - lr["full_width"]
    lr["bound_below_largest_loss"] = lr["information_bound"] < lr["largest_loss"]
    q4 = lr.groupby(["experiment", "estimator"]).agg(
        replicates=("bound_ratio", "size"), ratio_max=("bound_ratio", "max"),
        ratio_median=("bound_ratio", "median"), bound_min=("information_bound", "min"),
        bound_mean=("information_bound", "mean"), loss_mean=("loss", "mean"),
        largest_loss_max=("largest_loss", "max"),
        bound_below_largest_loss=("bound_below_largest_loss", "mean")).reset_index()
    q4.to_csv(TABLES / "q4-information-bound.csv", index=False)

    if 2 in cells:
        keep = ["decoy", "n", "estimator", "replicates", "pop_width", "pop_width_se", "full_width",
                "oracle_width", "best_width", "share_of_full_gain", "share_of_oracle_gain",
                "share_of_best_gain", "plug_width", "sel_width", "sel_covers_truth",
                "sel_covers_lower_end", "sel_covers_upper_end"]
        keep += [c for c in ("ami_combined", "ami_access", "ami_practice") if c in cells[2]]
        cells[2][keep].to_csv(TABLES / "q5-two-group.csv", index=False)
    if 3 in cells:
        keep = ["decoy", "n", "estimator", "replicates", "pop_lower", "pop_upper", "pop_width",
                "pop_width_se", "full_lower", "full_upper", "full_width", "oracle_width",
                "best_width", "share_of_full_gain", "share_of_oracle_gain", "share_of_best_gain",
                "plug_width", "sel_width",
                "sel_covers_truth", "sel_covers_lower_end", "sel_covers_upper_end"]
        cells[3][keep].to_csv(TABLES / "q6-mixed-sign.csv", index=False)
    print({e: (len(data[e]), data[e].attrs["failures"]) for e in exps})


if __name__ == "__main__":
    main()
