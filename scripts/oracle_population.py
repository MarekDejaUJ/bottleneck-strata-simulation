"""Exact population bounds of the oracle stratifier (strata = true regimes) for every cell.

The oracle's population bounds follow from the regime margins of the data-generating
process: the Tian-Pearl bounds of every regime, averaged with the regime probabilities.
They depend on the cell and not on the replicate. Output: ``results/tables/oracle-population.csv``.

Usage:
    python oracle_population.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import simulation as sim

OUT = Path(__file__).resolve().parents[1] / "results" / "tables" / "oracle-population.csv"


def main() -> None:
    rows = []
    for cell in sim.cells_exp1() + sim.cells_exp2():
        dgp = sim.build_dgp(cell)
        b = dgp.oracle_bounds()
        t = dgp.truth()
        rows.append(dict(experiment=cell.experiment, purity=cell.purity, decoy=cell.decoy, n=cell.n,
                         k=cell.k, true_pns=t["PNS"], true_pn=t["PN"], true_ps=t["PS"],
                         pns_lower=b.pns.lower, pns_upper=b.pns.upper, pns_width=b.pns.upper - b.pns.lower,
                         pn_lower=b.pn.lower, pn_upper=b.pn.upper, ps_lower=b.ps.lower, ps_upper=b.ps.upper,
                         valid=bool(b.pns.lower - 1e-12 <= t["PNS"] <= b.pns.upper + 1e-12)))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{len(rows)} cells written to {OUT}")


if __name__ == "__main__":
    main()
