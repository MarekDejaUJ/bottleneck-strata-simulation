# Bottleneck Strata Simulation

Data and code of the simulation study in Marek Deja, *Bottleneck strata for bounds on probabilities of causation from combined experimental and observational data* (manuscript). The study answers six questions fixed in `PROTOCOL.md` before the run: the validity of the population bounds of stratifiers learned from a sample, the width of the bounds and the component of a stratifier that carries the narrowing, the behavior of the estimated bounds and of the intervals in finite samples, the information bound on the width lost by compression, the two-group specification of the multivariate bottleneck, and latent profiles with effects of both signs.

## Versions

- **2.0.0** (this version): protocol version 2 and its run. Experiments 1 and 2 keep the simulated populations, the seeds and the stratifiers of version 1.0.0, and the rows of those stratifiers repeat the rows of version 1.0.0. Added: six stratifiers (the bottleneck with a logistic relevance model, the bottleneck without relevance weights, k-means on weighted covariates, k-means on fitted distributions, strata of pairs of terms, quantile groups of the fitted effect), the estimate with binding terms selected out of fold and its interval, the coverage of the intervals at the ends of the identified set, population bounds on PN and PS, the narrowest bounds of two, three and four strata, the exact population bounds of the true latent profiles in the replicate tables, 100 further replicates per cell of Experiment 2, and Experiment 3. Software: ibstrata 0.2.0.
- **1.0.0** (https://doi.org/10.5281/zenodo.24350575): protocol version 1 and its run, with ibstrata 0.1.0.

## Contents

| Path | Content |
|---|---|
| `PROTOCOL.md` | Protocol of the study, version 2, written on 18 November 2026 before the run |
| `scripts/simulation.py` | Experiments 1 to 3: simulated data, stratifiers, estimated and population bounds, intervals; writes the replicate tables |
| `scripts/summarize.py` | Cell tables and the tables of the six questions of the protocol |
| `results/tables/` | Result tables of the study |
| `ibstrata-0.2.0/` | ibstrata 0.2.0, the Python package that implements the method, with its tests |
| `LICENSE` | GNU General Public License, version 3 (scripts) |
| `LICENSE-DATA` | Creative Commons Attribution 4.0 International (protocol, README and tables) |

## Terms

The protocol, the scripts and the tables use the terms of the software. A regime is a latent profile of the paper. In Experiment 2 the access regime is the information reach profile and the practice regime is the information practice profile. The estimators are:

| Estimator | Stratifier |
|---|---|
| `plain` | no strata |
| `oracle` | the true latent profiles |
| `random` | strata drawn at random (Experiments 1 and 3) |
| `bottleneck` | the information bottleneck with the nearest-neighbor relevance model |
| `bottleneck-unweighted` | the same without relevance weights in the smoother (Experiment 1 with three profiles and purity above 0; Experiment 3) |
| `bottleneck-logistic` | the information bottleneck with the logistic relevance model |
| `kmeans` | outcome-blind k-means on all covariates |
| `kmeans-weighted` | k-means on the covariates scaled by the relevance weights |
| `kmeans-fitted` | k-means with the squared Euclidean distance on the fitted distributions of the logistic relevance model |
| `term-pairs` | strata of pairs of a lower and an upper term of the PNS bounds, chosen and assigned at the fitted distributions of the logistic relevance model |
| `effect-quantiles` | quantile groups of the fitted effect of the logistic relevance model |
| `reach-practice` | the two-group specification of the multivariate bottleneck (Experiment 2) |

## Design

Experiment 1 crosses purity (0, 0.6, 0.85), decoy structure (absent, present), sample size n (300, 600, 900, 1500) and the number k of latent profiles (2, 3): 48 cells with 200 replicates each and 99,200 rows in `exp1-replicates.csv`. Experiment 2 has a binary information reach profile and a binary information practice profile carried by separate groups of covariates, purity 0.85, decoy structure absent or present and n of 600, 900 or 1500: 6 cells with 200 replicates each, k = 4 strata and 12,000 rows in `exp2-replicates.csv`. Experiment 3 has four latent profiles with effects of both signs and without tied terms of the bounds, prototypes of the profiles drawn anew in every replicate, decoy structure absent or present and n of 600 or 1500: 4 cells with 200 replicates each, k = 4 strata and 8,800 rows in `exp3-replicates.csv`. Every replicate draws its data from a seed fixed by the cell and the replicate number (master seed 20261117), so a rerun with the same software on the same platform gives the same row.

## Tables

### exp1-replicates.csv, exp2-replicates.csv and exp3-replicates.csv

One row per cell, replicate and estimator.

| Column | Description |
|---|---|
| experiment, purity, decoy, n, k | cell (purity is 0 in Experiment 3, whose compositions are given in the protocol) |
| replicate | replicate number |
| true_pns, true_pn, true_ps | true PNS, PN and PS of the simulated population |
| full_lower, full_upper | population PNS bounds stratified by the full covariates, averaged over the evaluation sample |
| full_pn_lower, full_pn_upper, full_ps_lower, full_ps_upper | population bounds on PN and PS stratified by the full covariates |
| oracle_lower, oracle_upper | exact population PNS bounds stratified by the true latent profiles |
| binding_patterns | number of pairs of binding lower and upper term of the PNS bounds that hold at least 1% of the evaluation sample |
| best_width_2, best_width_3, best_width_4 | narrowest population width of the PNS bounds of two, three and four strata: the minimum over the sets of as many pairs of terms of the mean, over the evaluation sample, of the smallest width of a chosen pair |
| estimator | stratifier |
| error | message of a failed fit; empty otherwise |
| pns_lower, pns_upper, pn_lower, pn_upper, ps_lower, ps_upper | plug-in estimates of the bounds on PNS, PN and PS |
| se_lower, se_upper | bootstrap standard errors of the plug-in PNS endpoints (200 resamples within the design arms, memberships held fixed) |
| ci_lower, ci_upper | Imbens–Manski 95% interval for the PNS built on the plug-in endpoints |
| strata | number of strata of the plug-in estimate after merging strata with an expected size below five units |
| incompatible | number of strata whose estimated margins violate P(Y_x = y) ≥ P(X = x, Y = y) for some value x of the treatment and y of the outcome |
| sel_lower, sel_upper | PNS bounds with binding terms selected out of fold, averaged over the two folds |
| sel_se_lower, sel_se_upper | standard errors of the averaged endpoints from the bootstrap standard errors of the fold endpoints (200 resamples within folds and design arms, memberships and selected terms held fixed), with the folds treated as independent |
| sel_ci_lower, sel_ci_upper | 95% interval of the estimate with selected terms: the intersection of the Imbens–Manski intervals of the two folds, each at the level 0.975; a lower limit above the upper limit marks an empty intersection |
| sel_avg_ci_lower, sel_avg_ci_upper | secondary interval: Imbens–Manski 95% interval built on `sel_lower`, `sel_upper` and their standard errors |
| sel_strata | largest number of strata of a fold after merging strata with fewer than two units of a design arm |
| pop_lower, pop_upper | population PNS bounds of the stratifier, computed on an evaluation sample of 20,000 covariate vectors; for `oracle`, the exact bounds of the true latent profiles |
| pop_pn_lower, pop_pn_upper, pop_ps_lower, pop_ps_upper | population bounds of the stratifier on PN and PS |
| info_c_v_given_t | conditional mutual information I(C; V \| T) between the covariates and the augmented outcome given the strata, in nats, on the evaluation sample; empty for `oracle` |
| information_bound | kappa * sqrt(I(C; V \| T) / 2) with kappa = 6, the bound of the paper's theorem on the width lost by compression; empty for `oracle` |
| loss | width lost relative to the full covariates, (full_lower - pop_lower) + (pop_upper - full_upper); empty for `oracle` |
| seconds | running time of the estimator |
| ami_combined | Experiment 2: adjusted mutual information between the learned strata and the true combination of the two latent profiles |
| k_a_fold0, k_m_fold0, k_a_fold1, k_m_fold1 | Experiment 2, `reach-practice`: numbers of information reach and information practice strata of the encoders fitted for fold 0 and fold 1 |
| ami_access, ami_practice | Experiment 2, `reach-practice`: adjusted mutual information between the learned and the true information reach profile and between the learned and the true information practice profile, averaged over the folds |

### exp1-cells.csv, exp2-cells.csv and exp3-cells.csv

One row per cell and estimator, over the replicates without failure: `replicates`, the means of the replicate columns and of the derived columns below, and the Monte Carlo standard errors `pop_width_se`, `plug_width_se` and `sel_width_se`. The prefix `plug` marks the plug-in estimate and its interval, `sel` the estimate with selected terms and the intersection of the fold intervals, and `selavg` the secondary interval.

| Column | Description |
|---|---|
| pop_width, full_width, oracle_width | widths of the population PNS bounds of the stratifier, of the full covariates and of the true latent profiles |
| best_width | narrowest population width of as many strata as the cell uses (`best_width_2`, `best_width_3` or `best_width_4`) |
| plain_pop_width | population width without strata |
| pop_pn_width, pop_ps_width, full_pn_width, full_ps_width | widths of the population bounds on PN and PS |
| valid_pns, valid_pn, valid_ps | share of replicates whose population bounds contain the true value (tolerance 0.001 for the Monte Carlo error of the evaluation sample) |
| share_of_full_gain | (population width without strata - pop_width) / (population width without strata - full_width); empty when the two reference widths coincide |
| share_of_oracle_gain, share_of_best_gain | the same with oracle_width and with best_width in the place of full_width |
| plug_width, sel_width | mean width of the plug-in bounds and of the bounds with selected terms |
| plug_err_lower, plug_err_upper, sel_err_lower, sel_err_upper | mean signed error of the estimated endpoints against the population endpoints of the same stratifier |
| plug_contains, sel_contains | share of replicates whose estimated bounds contain the true PNS |
| plug_ci_width, sel_ci_width, selavg_ci_width | mean width of the 95% interval |
| plug_covers_truth, sel_covers_truth, selavg_covers_truth | share of replicates whose interval contains the true PNS |
| plug_covers_lower_end, plug_covers_upper_end, sel_covers_lower_end, sel_covers_upper_end, selavg_covers_lower_end, selavg_covers_upper_end | share of replicates whose interval contains the lower end and the upper end of the identified set: the population bounds of the full covariates, and for `oracle` the exact bounds of the true latent profiles |

### Tables of the questions

| Table | Content |
|---|---|
| `q1-validity.csv` | per experiment and estimator: replicates and the shares `valid_pns`, `valid_pn`, `valid_ps` |
| `q2-widths.csv` | per cell and estimator: population endpoints and width with its standard error, reference widths, shares of the narrowing, widths of the PN and PS bounds, number of binding patterns |
| `q2-paired.csv` | per cell and pair of estimators (`first`, `second`): mean paired difference of the population widths, first minus second, with its standard error |
| `q3-finite-sample.csv` | per cell and estimator: the finite-sample columns of the cell tables |
| `q3-coverage.csv` | per experiment, group of cells (`narrowing_available`: the full covariates narrow the bounds without strata), estimator and n: shares of replicates whose estimated bounds contain the true PNS and whose intervals contain the true PNS and the two ends of the identified set |
| `q4-information-bound.csv` | per experiment and estimator built from the covariates: largest and median ratio of the width lost to the information bound, smallest and mean bound, mean loss, largest possible loss (population width without strata minus population width of the full covariates), and the share of replicates in which the bound lies below the largest possible loss |
| `q5-two-group.csv` | Experiment 2 per cell and estimator: widths, shares, coverage of the interval with selected terms and adjusted mutual information |
| `q6-mixed-sign.csv` | Experiment 3 per cell and estimator: population endpoints, widths, shares and coverage of the interval with selected terms |

## Running the study

The study ran with Python 3.12, NumPy 2.5.5, SciPy 1.18.1, pandas 3.1.1 and ibstrata 0.2.0 (which also installs NetworkX).

```
python -m venv .venv
.venv\Scripts\activate             (Windows; Linux and macOS: source .venv/bin/activate)
pip install numpy==2.5.5 scipy==1.18.1 pandas==3.1.1
pip install ./ibstrata-0.2.0
python scripts/simulation.py --experiment all --workers 8
python scripts/summarize.py
```

The scripts write into `results/` and replace the stored tables, so a run in a copy of the folder keeps the stored tables for comparison. `simulation.py` writes every finished cell to `results/parts/` and continues an interrupted run with the missing cells. The full run took about six hours with 14 worker processes; `--limit 2` runs two replicates per cell, and `--seed` sets a master seed other than the study's. The fits of the logistic relevance model end in an iterative optimizer, so rows of the stratifiers built on that model can differ in late digits between platforms. The tests of ibstrata, which include the comparison of the bound functionals with the exact linear program, run with `pip install "./ibstrata-0.2.0[test]"` and `pytest ibstrata-0.2.0/tests`.

## License

The scripts are licensed under the GNU General Public License, version 3 or later (`LICENSE`), as is ibstrata 0.2.0 (`ibstrata-0.2.0/LICENSE`). The protocol, this README and the result tables are licensed under the Creative Commons Attribution 4.0 International license (`LICENSE-DATA`).
