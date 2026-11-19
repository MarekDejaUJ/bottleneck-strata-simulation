# ibstrata

Information-bottleneck strata for bounds on probabilities of causation.

`ibstrata` computes bounds on the probabilities of necessity and sufficiency (PNS), necessity (PN) and sufficiency (PS), and on their versions for multi-valued treatments and outcomes, from experimental and observational data collected in one population. The package compresses high-dimensional pre-treatment covariates into a small number of strata with a (multivariate) information bottleneck whose relevance variable is the arm-augmented outcome, and computes the bounds within the strata, with the plug-in estimate and with binding terms selected out of fold.

## Bounds available

| Function | Quantity | Source | Sharp |
|---|---|---|---|
| `lp_bounds` | any event on the choice and the potential outcomes, joint or conditional on observed variables | exact linear program over the joint of the choice and the potential outcomes | yes, given the margins |
| `tian_pearl` | PNS, PN, PS for binary treatment and outcome | Tian and Pearl (2000) | yes |
| `monotonic_point` | PNS, PN, PS under monotonicity | Tian and Pearl (2000) | point values |
| `li_pearl_theorem4`, `li_pearl_theorem5` | probability of preservation and probability of replacement, multi-valued treatment and outcome | Li and Pearl (2024), Theorems 4 and 5, as printed | valid; the lower bounds fall below the attainable value in part of the parameter space |
| `preservation_sharp`, `replacement_sharp` | the same two quantities | lower bounds of Shu, Wang and Li (2026), Theorem 4, with one hypothetical term; upper bounds of Li and Pearl (2024) | lower bounds attained for every number of treatment and outcome values |
| `li_pearl_theorem6`, `li_pearl_theorem7` | probability of substitute and probability of necessity with one hypothetical term | Li and Pearl (2024), Theorems 6 and 7 | equal to the exact linear program on the random margins of the tests |
| `shu_wang_li_pns` | PNS(k) for multi-valued treatment and outcome | Shu, Wang and Li (2026), Theorem 2 | sound; tight in the cases its authors checked |
| `stratified`, `stratified_lp`, `stratified_tian_pearl` | covariate-stratified bounds with soft weights | Mueller, Li and Pearl (2022), Theorem 4, extended to every stratifier | valid for every pre-treatment stratifier |

Every closed form is tested against the exact linear program on random distributions, and the worked examples of Li and Pearl (2024) are reproduced in the test suite.

## Example

```python
from ibstrata.bounds import Margins, tian_pearl, lp_bounds, Event

# Forced arms: outcome counts under treatment x0 and x1; free-choice arm: choice by outcome.
margins = Margins.from_counts(
    experimental_counts=[[140, 60], [80, 120]],
    observational_counts=[[90, 30], [20, 60]],
)
b = tian_pearl(margins, x=1, y=1)
print(b.pns, b.pn, b.ps)
print(lp_bounds(margins, Event(do={1: 1, 0: 0})))
```

## Installation

```sh
pip install ibstrata
```

Python 3.10 or later; the runtime dependencies are NumPy, SciPy, pandas and NetworkX.

## Versions

The package follows Semantic Versioning in the notation of PEP 440; `CHANGELOG.md` lists the changes of every version, and each release is tagged in Git as `ibstrata-vX.Y.Z`.

## Licence

GPL-3.0-or-later. Copyright Marek Deja, Jagiellonian University.
