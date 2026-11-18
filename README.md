# Bottleneck Strata Simulation

Data and code of the simulation study in Marek Deja, *Bottleneck strata for bounds on probabilities of causation from combined experimental and observational data* (manuscript). The study answers five questions fixed in `PROTOCOL.md` before any data were produced: the validity of the population bounds of stratifiers learned from a sample, the width of the bounds, the behavior of the estimated bounds and of the Imbens–Manski interval in finite samples, the certificate on the width lost by compression, and the two-group specification of the multivariate bottleneck.

## Contents

| Path | Content |
|---|---|
| `PROTOCOL.md` | Protocol of the study, written on 17 November 2026 before any data were produced |
| `scripts/simulation.py` | Experiments 1 and 2: simulated data, stratifiers, estimated and population bounds, intervals; writes the replicate and cell tables and `results/environment.json` (software versions, master seed, start and end times) |
| `scripts/oracle_population.py` | Exact population bounds of the true latent profiles for every cell; writes `results/tables/oracle-population.csv` |
| `scripts/summarize.py` | Tables of the five questions of the protocol and the headline numbers in `results/summary.json` |
| `results/tables/` | Result tables of the study |
| `ibstrata-0.1.0/` | ibstrata 0.1.0, the Python package that implements the method, with its tests |
| `LICENSE` | GNU General Public License, version 3 (scripts) |
| `LICENSE-DATA` | Creative Commons Attribution 4.0 International (protocol, README and tables) |

## Terms

The protocol, the scripts and the tables use the terms of the software. A regime is a latent profile of the paper. In Experiment 2 the access regime is the information reach profile and the practice regime is the information practice profile. The estimators are `plain` (no strata), `oracle` (strata given by the true latent profiles), `bottleneck` (the information bottleneck), `kmeans` (outcome-blind k-means on all covariates), `random` (strata drawn at random) and `reach-practice` (the two-group specification of the multivariate bottleneck, Experiment 2 only).

## Design

Experiment 1 crosses purity (0, 0.6, 0.85), decoy structure (absent, present), sample size n (300, 600, 900, 1500) and the number k of latent profiles (2, 3): 48 cells with 200 replicates each and 48,000 rows in `exp1-replicates.csv`. Experiment 2 has a binary information reach profile and a binary information practice profile carried by separate groups of covariates, purity 0.85, decoy structure absent or present and n of 600, 900 or 1500: 6 cells with 100 replicates each, k = 4 strata and 3,000 rows in `exp2-replicates.csv`. Every replicate draws its data from a seed fixed by the cell and the replicate number (master seed 20261117), so a rerun with the same software gives the same row.

## Tables

### exp1-replicates.csv and exp2-replicates.csv

One row per cell, replicate and estimator.

| Column | Description |
|---|---|
| experiment, purity, decoy, n, k | cell |
| replicate | replicate number |
| true_pns, true_pn, true_ps | true PNS, PN and PS of the simulated population |
| full_lower, full_upper | population PNS bounds stratified by the full covariates, averaged over the evaluation sample |
| estimator | stratifier |
| error | message of a failed fit; empty otherwise |
| pns_lower, pns_upper, pn_lower, pn_upper, ps_lower, ps_upper | estimated bounds on PNS, PN and PS |
| se_lower, se_upper | bootstrap standard errors of the estimated PNS endpoints (200 resamples within the design arms, memberships held fixed) |
| ci_lower, ci_upper | Imbens–Manski 95% interval for the PNS |
| strata | number of strata after merging strata with an expected size below five units |
| incompatible | number of strata whose estimated margins violate P(Y_x = y) ≥ P(X = x, Y = y) for some value x of the treatment and y of the outcome |
| pop_lower, pop_upper | population PNS bounds of the stratifier, computed on an evaluation sample of 20,000 covariate vectors |
| info_c_v_given_t | conditional mutual information I(C; V \| T) between the covariates and the augmented outcome given the strata, in nats, on the evaluation sample |
| certificate | kappa * sqrt(I(C; V \| T) / 2) with kappa = 6, the bound of the paper's theorem on the width lost by compression |
| loss | width lost relative to the full covariates, (full_lower - pop_lower) + (pop_upper - full_upper) |
| seconds | running time of the estimator |
| ami_combined | Experiment 2: adjusted mutual information between the learned strata and the true combination of the two latent profiles |
| k_a_fold0, k_m_fold0, k_a_fold1, k_m_fold1 | Experiment 2, `reach-practice`: numbers of information reach and information practice strata of the encoders fitted on fold 0 and fold 1 |
| ami_access, ami_practice | Experiment 2, `reach-practice`: adjusted mutual information between the learned and the true information reach profile and between the learned and the true information practice profile, averaged over the folds |

In the `oracle` rows, `pop_lower` and `pop_upper` are the bounds at the average of the covariate conditionals over the members of each true latent profile, which are the bounds of a soft stratification by the posterior probability of the profile given the covariates, and `info_c_v_given_t`, `certificate` and `loss` follow from the same averages. The exact population bounds of the true latent profiles are in `oracle-population.csv`, and `summarize.py` takes the population bounds of the oracle from that table.

### exp1-cells.csv and exp2-cells.csv

One row per cell and estimator, over the replicates without failure.

| Column | Description |
|---|---|
| experiment, purity, decoy, n, k | cell |
| estimator | stratifier |
| replicates | replicates without failure |
| width, pop_width | mean width of the estimated and of the population PNS bounds |
| err_lower, err_upper | mean signed error of the estimated lower and upper PNS endpoints against the population endpoints of the same stratifier |
| covers | share of replicates whose estimated bounds contain the true PNS |
| ci_covers, ci_covers_se | share of replicates whose Imbens–Manski 95% interval contains the true PNS, and its Monte Carlo standard error |
| valid | share of replicates whose population bounds contain the true PNS (tolerance 0.001 for the Monte Carlo error of the evaluation sample) |
| cert_ratio_max, cert_ratio_median | largest and median ratio of `loss` to `certificate` |
| strata, incompatible, seconds | means of the replicate columns |
| failures | number of failed fits |
| ami_combined, ami_access, ami_practice | Experiment 2: means of the replicate columns |

In the `oracle` rows, `pop_width`, `valid` and the certificate ratios refer to the soft stratification described above.

### q1-validity.csv

| Column | Description |
|---|---|
| experiment | 1 or 2 |
| estimator | stratifier |
| replicates | replicates without failure |
| valid_share | share of replicates whose population bounds contain the true PNS (tolerance 0.001) |

### q2-widths.csv (Experiment 1)

| Column | Description |
|---|---|
| experiment, purity, decoy, n, k | cell |
| estimator | stratifier |
| replicates | replicates without failure |
| width | mean width of the estimated bounds |
| pop_width | mean width of the population bounds of the stratifier (oracle: exact width from the margins of the latent profiles) |
| full_width | mean width of the population bounds stratified by the full covariates |
| share_of_oracle_gain | (plain population width - pop_width) / (plain population width - oracle population width); empty when the two reference widths coincide |
| share_of_full_gain | (plain population width - pop_width) / (plain population width - full_width); empty when the two reference widths coincide |

### q3-finite-sample.csv (Experiment 1, averaged over the cells of a sample size)

| Column | Description |
|---|---|
| estimator, n | stratifier and sample size |
| err_lower, err_upper | mean signed error of the estimated lower and upper endpoints against the population endpoints of the same stratifier |
| covers | share of replicates whose estimated bounds contain the true PNS |
| ci_covers_mean, ci_covers_min | mean and smallest cell share of replicates whose Imbens–Manski 95% interval contains the true PNS |

### q4-certificate.csv

| Column | Description |
|---|---|
| experiment, estimator | experiment and stratifier built from the covariates |
| replicates | replicates with a positive certificate |
| ratio_max, ratio_q99, ratio_median | largest, 99th percentile and median of the ratio of the width lost relative to the full covariates to the certificate kappa * sqrt(I(C; V \| T) / 2) |

### q5-two-regimes.csv (Experiment 2)

| Column | Description |
|---|---|
| decoy, n, estimator | cell and stratifier |
| replicates, width, pop_width, full_width, share_of_oracle_gain, share_of_full_gain | as in `q2-widths.csv` |
| ci_covers | share of replicates whose Imbens–Manski 95% interval contains the true PNS |
| ami_combined | adjusted mutual information between the learned strata and the true combination of the two latent profiles |
| ami_access, ami_practice | adjusted mutual information between the learned and the true information reach profile and between the learned and the true information practice profile |

### oracle-population.csv

Exact population bounds of the true latent profiles for every cell (`experiment`, `purity`, `decoy`, `n`, `k`): true PNS, PN and PS; lower and upper bounds on PNS, PN and PS; width of the PNS bounds (`pns_width`); `valid`, whether the PNS bounds contain the true PNS.

## Running the study

The study ran with Python 3.12, NumPy 2.5.5, SciPy 1.18.1, pandas 3.1.1 and ibstrata 0.1.0 (which also installs NetworkX).

```
python -m venv .venv
.venv\Scripts\activate             (Windows; Linux and macOS: source .venv/bin/activate)
pip install numpy==2.5.5 scipy==1.18.1 pandas==3.1.1
pip install ./ibstrata-0.1.0
python scripts/simulation.py --experiment all --workers 8
python scripts/oracle_population.py
python scripts/summarize.py
```

The scripts write into `results/` and replace the stored tables, so a run in a copy of the folder keeps the stored tables for comparison. The full run takes about four hours with eight worker processes; `--limit 2` runs two replicates per cell. The tests of ibstrata, which include the comparison of the bound functionals with the exact linear program, run with `pip install "./ibstrata-0.1.0[test]"` and `pytest ibstrata-0.1.0/tests`.

## License

The scripts are licensed under the GNU General Public License, version 3 or later (`LICENSE`), as is ibstrata 0.1.0 (`ibstrata-0.1.0/LICENSE`). The protocol, this README and the result tables are licensed under the Creative Commons Attribution 4.0 International license (`LICENSE-DATA`).
