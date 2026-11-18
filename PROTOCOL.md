# Method validation: bottleneck strata for bounds on probabilities of causation

Written 17 November 2026, before any data of the study were produced. Paper: `papers/method`. Script: `scripts/simulation.py`. Package: `ibstrata` at the commit recorded in `results/environment.json`.

## 1. Questions

- **Q1 Validity.** Do the population bounds of a stratifier learned from a sample contain the true PNS, PN and PS? Criterion: Propositions 1 and 2 of the method (stratified bounds are valid for every stratifier that depends on pre-treatment covariates only). Measure: share of replicates in which the population bounds of the learned stratifier contain the true value, computed on an evaluation sample with exact covariate-level conditionals; expected value 1 up to the Monte Carlo error of the evaluation sample.
- **Q2 Tightness.** How wide are the estimated PNS bounds of each stratifier? Measure: mean width of the estimated bounds per scenario, sample size and stratifier; width relative to the bounds without covariates and to the bounds stratified by the true regime (oracle).
- **Q3 Finite-sample behavior.** How far do estimated endpoints fall from the population endpoints of the same learned stratifier, and how often do intervals cover the true PNS? Measures: mean signed error of each endpoint; share of replicates in which the estimated bounds contain the true PNS; share in which the Imbens-Manski 95% interval (Imbens and Manski, 2004) contains the true PNS. Criterion for the interval: nominal coverage of 0.95 or more.
- **Q4 Information certificate.** Is the width lost by a learned stratifier relative to the full covariates below the bound of Theorem 1, and by how much? Measure: ratio of the loss (L_C - L_T) + (U_T - U_C) to kappa * sqrt(I(C; V | T) / 2), with both terms computed on the evaluation sample.
- **Q5 Two regimes.** When the response types depend on an access regime and a practice regime carried by two separate feature groups, how do product strata of the reach-practice bottleneck compare with a single bottleneck of the same number of strata? Measures: width and interval coverage as in Q2 and Q3; adjusted mutual information between each learned regime and its true regime.

## 2. Software, frozen

| Item | Version | Source |
|---|---|---|
| Python | 3.12 (Windows, CPython) | python.org |
| ibstrata | commit in `results/environment.json` | this repository |
| NumPy, SciPy, pandas, NetworkX | versions in `results/environment.json` | PyPI |

The script writes the versions and the package commit to `results/environment.json` before the first replicate.

## 3. Material: data-generating processes

All data are simulated by `ibstrata.simulate.make_regime_dgp`. Common settings: binary treatment and outcome, treatment of interest x = 1 and outcome y = 1; arms with probabilities (1/3, 1/3, 1/3) for the forced x' arm, the forced x arm and the free-choice arm; 12 binary signal features with regime prototypes and separation 0.5; 4 binary noise features with probability 1/2; choice logits spread evenly from -0.8 to 0.8 across regimes; confounding 1.0 (the logit of choosing x rises by 1 for helped units and falls by 1 for hurt units); regimes equally likely.

### Experiment 1 (Q1-Q4): factors

| Factor | Levels |
|---|---|
| Purity of response types within regimes | 0 (all regimes share one composition), 0.6, 0.85 |
| Decoy feature structure | absent; present (3 decoy clusters on 12 extra features, separation 0.9, independent of regimes and response types) |
| Sample size n | 300, 600, 900, 1500 |
| Number of regimes K | 2, 3 |

48 cells, 200 replicates per cell.

Response-type compositions (never, helped, hurt, always). Hurt is 0.03 in every regime. With purity rho > 0, regime r has a dominant type from the cycle (always, helped, never) in that order, with share rho, and the remaining 0.97 - rho split equally between the two other non-hurt types. With rho = 0 every regime has (0.30, 0.30, 0.03, 0.37).

### Experiment 2 (Q5)

Two binary regimes: access regime R_A carried by 8 access signal features and practice regime R_M carried by 8 practice signal features (separation 0.6 each, 2 noise features per group). Response types depend on (R_A, R_M) with purity 0.85, dominant types (never, helped, helped, always) for (0, 0), (0, 1), (1, 0), (1, 1). Factors: n in {600, 900, 1500}; decoy structure absent or present (as in Experiment 1, added to the access group). 100 replicates per cell.

## 4. Procedures and measures

### Estimators (both experiments)

All stratified estimators use two-fold cross-fitting within arms with pooled out-of-fold memberships (`honest_bounds`, `mode="pooled"`), a pseudo-count of 0.5 per cell of every stratum, and pooling of strata with expected size below 5 persons. The strata of the two fold encoders are aligned (Hungarian assignment on the agreement of their memberships over all persons, `ibstrata.stratify.crossfit`) and merged, so a learned stratifier has as many strata as the oracle. Amendment of 17 November 2026, made after a harness test and before the study run: the first version of this section kept the strata of the two fold encoders apart, which doubled the number of strata of learned stratifiers relative to the oracle.

- **plain**: Tian-Pearl bounds from the pooled margins.
- **oracle**: strata given by the true regime of every person.
- **bottleneck**: `BottleneckEncoder` with K strata (Experiment 2: 4 strata), relevance-weighted nearest-neighbour smoother with sqrt(n_train) neighbours (at least 10), arm-balanced relevance with pseudo-count 0.05, annealing over 50 betas from 0.5 to 3000 on a geometric grid, seed fixed per replicate.
- **kmeans**: outcome-blind k-means on all features with K clusters (Experiment 2: 4), 10 starts.
- **random**: K strata drawn uniformly at random for every person (a stratifier unrelated to the outcome).
- **reach-practice** (Experiment 2 only): `ReachPracticeEncoder` with 2 x 2 product strata from the access and the practice feature groups.

### Inference

For every estimator and replicate, 200 bootstrap resamples within arms with the memberships held fixed give standard errors of the PNS endpoints; the Imbens-Manski 95% interval uses them.

### Population quantities

For every replicate an evaluation sample of 20,000 covariate vectors is drawn from the same process. Exact conditionals q_c = sum_r P(r | c) q_r follow from the feature likelihoods and the regime margins (features are independent of the response types and of the choice given the regime). Full-covariate bounds L_C, U_C are the averages of the Tian-Pearl bounds at q_c; the population bounds of a learned stratifier average the bounds at q_t = sum_c w_ct q_c / sum_c w_ct over the evaluation sample with the stratifier's memberships (both fold encoders, each applied to a random half); I(C; V | T) = sum P(t, c) D(q_c || q_t) over the evaluation sample; kappa = 6 for the Tian-Pearl PNS bounds with equal allocation.

### Decision rules

A replicate in which an encoder fails (exception) is recorded with the error message and excluded from the means of that estimator only; the count of failures is reported per cell. No replicate is rerun.

## 5. Outputs

- `results/environment.json`: versions, commit, master seed, start and end times.
- `results/tables/exp1-replicates.csv`: one row per cell, replicate and estimator: factors, estimated PNS, PN, PS endpoints, Imbens-Manski interval, endpoint standard errors, strata, incompatible strata, population bounds of the learned stratifier, L_C, U_C, I(C; V | T), certificate, run time, error.
- `results/tables/exp1-cells.csv`: per cell and estimator: mean width, endpoint errors, coverage of the true PNS by the bounds and by the interval with Monte Carlo standard errors, validity share (Q1), certificate ratio summaries (Q4), failure count.
- `results/tables/exp2-replicates.csv`, `results/tables/exp2-cells.csv`: the same for Experiment 2 with the adjusted mutual information of the learned regimes.
- `results/summary.json`: headline numbers for the paper.
- `results/SUMMARY.md`: the tables read in prose.

## 6. Known limitations, stated in advance

- Binary treatment and outcome; the nonbinary bounds are verified separately against the exact linear program in the package tests and are not part of the simulation.
- The data-generating process makes covariates informative only through a latent regime; real covariates may carry graded heterogeneity without regimes.
- The interval treats endpoint estimates as approximately normal and ignores the variability of the learned encoder (memberships fixed in the bootstrap); undercoverage from either source is a result of Q3.
- Plug-in endpoints of maxima and minima are biased toward each other in finite samples; the study measures the bias and does not apply an intersection-bound correction.
- Equal allocation only; the certificate constant grows as the smallest arm probability falls.
