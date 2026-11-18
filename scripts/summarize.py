"""Tables for the questions of the protocol and the headline numbers of the study.

Reads the replicate tables written by ``simulation.py`` and the exact oracle population
bounds written by ``oracle_population.py``, and writes
``results/tables/q1-validity.csv`` to ``q5-two-regimes.csv`` and ``results/summary.json``.

Usage:
    python summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
TABLES = RESULTS / "tables"
CELL = ["experiment", "purity", "decoy", "n", "k"]
LEARNED = ["bottleneck", "kmeans", "reach-practice", "random", "plain"]


def load(exp: int, oracle: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(TABLES / f"exp{exp}-replicates.csv", dtype={"error": str})
    df["error"] = df["error"].fillna("")
    ok = df[df["error"] == ""].copy()
    exact = oracle[oracle["experiment"] == exp].set_index(CELL)[["pns_lower", "pns_upper"]]
    exact.columns = ["exact_lower", "exact_upper"]
    ok = ok.join(exact, on=CELL)
    is_oracle = ok["estimator"] == "oracle"
    ok.loc[is_oracle, "pop_lower"] = ok.loc[is_oracle, "exact_lower"]
    ok.loc[is_oracle, "pop_upper"] = ok.loc[is_oracle, "exact_upper"]
    for col in ("certificate", "loss", "info_c_v_given_t"):
        ok.loc[is_oracle, col] = np.nan
    ok["width"] = ok["pns_upper"] - ok["pns_lower"]
    ok["pop_width"] = ok["pop_upper"] - ok["pop_lower"]
    ok["full_width"] = ok["full_upper"] - ok["full_lower"]
    ok["covers"] = (ok["pns_lower"] <= ok["true_pns"]) & (ok["true_pns"] <= ok["pns_upper"])
    ok["ci_covers"] = (ok["ci_lower"] <= ok["true_pns"]) & (ok["true_pns"] <= ok["ci_upper"])
    ok["valid"] = (ok["pop_lower"] <= ok["true_pns"] + 1e-3) & (ok["true_pns"] <= ok["pop_upper"] + 1e-3)
    ok["err_lower"] = ok["pns_lower"] - ok["pop_lower"]
    ok["err_upper"] = ok["pns_upper"] - ok["pop_upper"]
    ok["cert_ratio"] = np.where(ok["certificate"] > 1e-9, ok["loss"] / ok["certificate"], np.nan)
    ok.attrs["failures"] = int((df["error"] != "").sum())
    return ok


def per_cell(ok: pd.DataFrame) -> pd.DataFrame:
    g = ok.groupby(CELL + ["estimator"])
    out = g.agg(replicates=("replicate", "size"), width=("width", "mean"), pop_width=("pop_width", "mean"),
                full_width=("full_width", "mean"), err_lower=("err_lower", "mean"),
                err_upper=("err_upper", "mean"), covers=("covers", "mean"), ci_covers=("ci_covers", "mean"),
                valid=("valid", "mean")).reset_index()
    out["ci_covers_se"] = np.sqrt(out["ci_covers"] * (1 - out["ci_covers"]) / out["replicates"])
    ref = out.pivot_table(index=CELL, columns="estimator", values="pop_width")
    span = (ref["plain"] - ref["oracle"]).rename("span")
    out = out.join(span, on=CELL).join(ref["plain"].rename("plain_pop_width"), on=CELL)
    out["share_of_oracle_gain"] = np.where(out["span"] > 1e-6,
                                           (out["plain_pop_width"] - out["pop_width"]) / out["span"], np.nan)
    full_span = out["plain_pop_width"] - out["full_width"]
    out["share_of_full_gain"] = np.where(full_span > 1e-6,
                                         (out["plain_pop_width"] - out["pop_width"]) / full_span, np.nan)
    return out.drop(columns=["span", "plain_pop_width"])


def main() -> None:
    oracle = pd.read_csv(TABLES / "oracle-population.csv")
    e1, e2 = load(1, oracle), load(2, oracle)
    c1, c2 = per_cell(e1), per_cell(e2)

    both = pd.concat([e1, e2])
    q1 = both.groupby(["experiment", "estimator"]).agg(
        replicates=("valid", "size"), valid_share=("valid", "mean")).reset_index()
    q1.to_csv(TABLES / "q1-validity.csv", index=False)

    q2 = c1[CELL + ["estimator", "replicates", "width", "pop_width", "full_width", "share_of_oracle_gain",
                    "share_of_full_gain"]]
    q2.to_csv(TABLES / "q2-widths.csv", index=False)

    q3 = c1.groupby(["estimator", "n"]).agg(err_lower=("err_lower", "mean"), err_upper=("err_upper", "mean"),
                                            covers=("covers", "mean"), ci_covers_mean=("ci_covers", "mean"),
                                            ci_covers_min=("ci_covers", "min")).reset_index()
    q3.to_csv(TABLES / "q3-finite-sample.csv", index=False)

    lr = both[both["estimator"].isin(LEARNED) & both["cert_ratio"].notna()]
    q4 = lr.groupby(["experiment", "estimator"])["cert_ratio"].agg(
        replicates="size", ratio_max="max", ratio_q99=lambda s: s.quantile(0.99),
        ratio_median="median").reset_index()
    q4.to_csv(TABLES / "q4-certificate.csv", index=False)

    ami = e2.groupby(["decoy", "n", "estimator"])[["ami_combined", "ami_access", "ami_practice"]].mean()
    q5 = c2[["decoy", "n", "estimator", "replicates", "width", "pop_width", "full_width", "ci_covers",
             "share_of_oracle_gain", "share_of_full_gain"]].join(ami, on=["decoy", "n", "estimator"])
    q5.to_csv(TABLES / "q5-two-regimes.csv", index=False)

    def pick(df, **kw):
        sel = df
        for k, v in kw.items():
            sel = sel[sel[k] == v]
        return sel.set_index("estimator")

    head = {}
    head["failures"] = {"exp1": e1.attrs["failures"], "exp2": e2.attrs["failures"]}
    head["q1_min_valid_share"] = float(q1["valid_share"].min())
    head["q1_replicates"] = int(q1["replicates"].sum())
    for decoy in (False, True):
        s = pick(c1, purity=0.85, k=3, n=1500, decoy=decoy)
        head[f"q2_k3_purity085_n1500_decoy{int(decoy)}"] = {
            e: {"pop_width": float(s.loc[e, "pop_width"]), "width": float(s.loc[e, "width"]),
                "share_of_oracle_gain": float(s.loc[e, "share_of_oracle_gain"]),
                "share_of_full_gain": float(s.loc[e, "share_of_full_gain"])}
            for e in s.index}
        head[f"q2_k3_purity085_n1500_decoy{int(decoy)}"]["full_covariates_pop_width"] = float(s["full_width"].iloc[0])
    head["q3_ci_coverage_min"] = float(c1["ci_covers"].min())
    head["q3_ci_coverage_by_n"] = {int(n): float(v) for n, v in c1.groupby("n")["ci_covers"].mean().items()}
    head["q3_bounds_coverage_by_n"] = {int(n): float(v) for n, v in c1.groupby("n")["covers"].mean().items()}
    head["q4_ratio_max"] = float(q4["ratio_max"].max())
    for decoy in (False, True):
        s = q5[(q5["decoy"] == decoy) & (q5["n"] == 1500)].set_index("estimator")
        head[f"q5_n1500_decoy{int(decoy)}"] = {
            e: {k: (None if pd.isna(s.loc[e, k]) else float(s.loc[e, k]))
                for k in ("pop_width", "ci_covers", "share_of_full_gain", "ami_combined", "ami_access",
                          "ami_practice")}
            for e in s.index}
    head["q5_ci_coverage_min"] = float(c2["ci_covers"].min())
    (RESULTS / "summary.json").write_text(json.dumps(head, indent=2), encoding="utf-8")
    print(json.dumps(head, indent=2)[:3000])


if __name__ == "__main__":
    main()
