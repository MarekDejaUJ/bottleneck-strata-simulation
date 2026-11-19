# Method validation: bottleneck strata for bounds on probabilities of causation

Protocol, version 2. Written 18 November 2026, before the run of the study. Script: `scripts/simulation.py`. Package: `ibstrata` 0.2.0.

Version 1 of the protocol (17 November 2026) and the tables of its run are archived as release 1.0.0 of the study (https://doi.org/10.5281/zenodo.24350575). Version 2 was written with the tables of version 1 known. Version 2 keeps the data-generating processes, the seeds and the estimators of Experiments 1 and 2, so the rows of the estimators of version 1 repeat, and adds six stratifiers, the estimate with binding terms selected out of fold and its interval, population bounds on PN and PS, the narrowest bounds of a given number of strata, the coverage of the intervals at the ends of the identified set, the comparison of the information bound with the largest possible loss, 100 further replicates per cell of Experiment 2, and Experiment 3. The added stratifiers and the added estimate were developed on harness runs with seeds other than the study's (option `--seed`); no harness row enters a table of the study.

## 1. Questions

- **Q1 Validity.** Do the population bounds of every stratifier contain the true PNS, PN and PS? Criterion: Propositions 1 and 4 of the method (stratified bounds are valid for every stratifier that depends on pre-treatment covariates only). Measure: share of replicates in which the population bounds of the stratifier contain the true value, computed on an evaluation sample with exact covariate-level conditionals, with a tolerance of 0.001 for the Monte Carlo error of the evaluation sample; expected value 1. The measure checks the bound functionals and the population quantities of the script.
- **Q2 Tightness.** How wide are the population PNS bounds of each stratifier, and which component of a stratifier carries the narrowing? Measures: mean population width per cell and stratifier with its Monte Carlo standard error; share of the narrowing from the bounds without covariates to the bounds of the full covariates, to the narrowest bounds of as many strata as the cell uses (Proposition 3) and to the bounds of the true regimes; mean paired differences of the population widths, with standard errors, for the pairs bottleneck and kmeans-weighted (bottleneck against clustering of the weighted features), bottleneck and bottleneck-unweighted (relevance weights), bottleneck and bottleneck-logistic (relevance model), bottleneck-logistic and kmeans-fitted (Kullback-Leibler against Euclidean clustering of the same fitted distributions), bottleneck-logistic and term-pairs (information distortion against the exact distortion of the PNS bounds on the same fitted distributions), bottleneck-logistic and effect-quantiles (whole augmented outcome against one predicted quantity); population widths of the PN and PS bounds; the narrowest population widths of 2, 3 and 4 strata; number of binding patterns of the full covariates (pairs of binding lower and upper term that hold at least 1% of the evaluation sample).
- **Q3 Finite-sample behavior.** For the plug-in estimate and for the estimate with binding terms selected out of fold: mean estimated width; mean signed error of each endpoint against the population endpoint of the same stratifier; share of replicates in which the estimated bounds contain the true PNS; share of replicates in which the 95% interval contains (a) the true PNS, (b) the lower end and (c) the upper end of the identified set. The identified set is the interval of the population bounds of the full covariates; for the oracle, whose strata are not a function of the covariates, the exact bounds of the true regimes. A population with the same distribution of all observed variables and a PNS at either end of the identified set exists, because the Tian-Pearl bounds are attained within every covariate value, and every estimator behaves in that population as in the simulated one; the share of intervals that contain an end is therefore the coverage of the interval in that population. Criterion for an interval: 0.95 or more at the true value and at both ends.
- **Q4 Information bound.** How does the width lost by a stratifier relative to the full covariates compare with the bound of Theorem 1, and how does the bound compare with the largest loss possible in the cell? Measures: ratio of the loss (L_C - L_T) + (U_T - U_C) to kappa * sqrt(I(C; V | T) / 2), both on the evaluation sample; share of replicates in which the bound lies below the largest possible loss, the population width without covariates minus the population width of the full covariates.
- **Q5 Two regimes.** When the response types depend on an access regime and a practice regime carried by two separate feature groups, how do the product strata of the reach-practice bottleneck compare with single stratifiers of the same number of strata? Measures: as in Q2 and Q3, with the paired differences reach-practice against bottleneck, bottleneck-logistic and kmeans-fitted; adjusted mutual information between each learned regime and its true regime.
- **Q6 Regimes with effects of both signs.** Do the answers to Q2 and Q3 hold when the regimes differ in the sign of the effect, no regime has tied terms of a bound, and the regime prototypes change from replicate to replicate? Measures: as in Q2 and Q3, with the lower and the upper population endpoints reported separately.

## 2. Software, frozen

| Item | Version |
|---|---|
| Python | 3.12 (Windows, CPython) |
| ibstrata | 0.2.0 |
| NumPy | 2.5.5 |
| SciPy | 1.18.1 |
| pandas | 3.1.1 |

## 3. Material: data-generating processes

All data are simulated by `ibstrata.simulate`. Common settings: binary treatment and outcome, treatment of interest x = 1 and outcome y = 1; arms with probabilities (1/3, 1/3, 1/3) for the forced x' arm, the forced x arm and the free-choice arm; binary signal features that equal the prototype of the regime with probability (1 + separation) / 2; binary noise features with probability 1/2; confounding 1.0 (the logit of choosing x rises by 1 for helped units and falls by 1 for hurt units); regimes equally likely. The decoy structure adds 3 latent clusters, independent of the regimes and of the response types, that shift 12 further features with separation 0.9.

### Experiment 1 (Q1-Q4): factors

| Factor | Levels |
|---|---|
| Purity of response types within regimes | 0 (all regimes share one composition), 0.6, 0.85 |
| Decoy feature structure | absent; present |
| Sample size n | 300, 600, 900, 1500 |
| Number of regimes K | 2, 3 |

48 cells, 200 replicates per cell. 12 signal features with separation 0.5 and 4 noise features; choice logits spread evenly from -0.8 to 0.8 across regimes; one draw of regime prototypes for all replicates.

Response-type compositions (never, helped, hurt, always). Hurt is 0.03 in every regime. With purity rho > 0, regime r has a dominant type from the cycle (always, helped, never) in that order, with share rho, and the remaining 0.97 - rho split equally between the two other non-hurt types. With rho = 0 every regime has (0.30, 0.30, 0.03, 0.37). In the helped-dominant regime the never and the always units have equal shares and equal choice probabilities, so the four terms of the upper PNS bound coincide in that regime.

### Experiment 2 (Q5)

Two binary regimes: access regime R_A carried by 8 access signal features and practice regime R_M carried by 8 practice signal features (separation 0.6 each, 2 noise features per group). Response types depend on (R_A, R_M) with purity 0.85, dominant types (never, helped, helped, always) for (0, 0), (0, 1), (1, 0), (1, 1). Factors: n in {600, 900, 1500}; decoy structure absent or present (added to the access group). 6 cells, 200 replicates per cell, 4 strata.

### Experiment 3 (Q6)

Four regimes with the compositions (never, helped, hurt, always) = (0.08, 0.66, 0.05, 0.21), (0.25, 0.05, 0.52, 0.18), (0.58, 0.10, 0.06, 0.26) and (0.12, 0.20, 0.10, 0.58) and the choice logits 0.7, -0.6, -0.1 and 0.3. The effect is positive in the first, third and fourth regime and negative in the second; the binding lower terms differ across the regimes, and no regime has tied terms. 12 signal features with separation 0.5 and 4 noise features; the regime prototypes are drawn anew for every replicate. Factors: n in {600, 1500}; decoy structure absent or present. 4 cells, 200 replicates per cell, 4 strata.

## 4. Procedures and measures

### Stratifiers

Every learned stratifier is fitted with two-fold cross-fitting within arms (`ibstrata.stratify.crossfit`): the encoder of a fold is fitted on the other fold. The number of strata is K in Experiment 1 and 4 in Experiments 2 and 3.

- **plain**: one stratum.
- **oracle**: strata given by the true regime of every person.
- **random**: K strata drawn uniformly at random for every person (Experiments 1 and 3).
- **bottleneck**: `BottleneckEncoder` with the nearest-neighbour relevance model: relevance-weighted smoother with sqrt(n_fit) neighbours (at least 10), n_fit the number of persons the encoder is fitted on; arm-balanced relevance with pseudo-count 0.05; annealing over 50 betas from 0.5 to 3000 on a geometric grid; the solution with the chosen number of strata at the last beta; a new person receives the average assignment of the nearest neighbours.
- **bottleneck-unweighted**: the bottleneck without relevance weights in the smoother. Runs in the cells of Experiment 1 with three regimes and purity above 0 and in Experiment 3.
- **bottleneck-logistic**: `BottleneckEncoder` with the logistic relevance model: within every arm, a multinomial logistic regression of the observed cell on the standardized features, fitted by maximum likelihood with the penalty of half the squared norm of the coefficients and 0.05 pseudo-observations of every class at the mean of the features; the same annealing; a new person is assigned by the stationary equation of the bottleneck at the fitted distribution of the person.
- **kmeans**: outcome-blind k-means on all features, Lloyd iterations from centers drawn at random among the persons, best of 10 starts.
- **kmeans-weighted**: k-means on the features scaled by the square roots of the relevance weights of the bottleneck.
- **kmeans-fitted**: k-means with the squared Euclidean distance on the fitted distributions of the logistic relevance model.
- **term-pairs**: `TermPairEncoder`: the pairs of a lower and an upper term of the Tian-Pearl PNS bounds, as many as strata, with the smallest mean width of the narrowest chosen pair at the fitted distributions of the logistic relevance model (all sets of pairs enumerated); a person belongs to the pair with the smallest width at the fitted distribution of the person.
- **effect-quantiles**: quantile groups of the fitted effect P(y_x | c) - P(y_x' | c) of the logistic relevance model, with cut points at the quantiles of the persons the model is fitted on.
- **reach-practice** (Experiment 2): `ReachPracticeEncoder` with 2 x 2 product strata from the access and the practice feature groups.

### Estimates

- **Plug-in.** The strata of the two fold encoders are aligned (Hungarian assignment on the agreement of their memberships over all persons) and merged; margins of every stratum with a pseudo-count of 0.5 per cell; strata with expected size below 5 persons are pooled; Tian-Pearl bounds at the margins of every stratum.
- **Selected out of fold.** `ibstrata.stratify.selected_term_bounds`. For every fold and every stratum of the encoder of the fold, the lower term and the upper term of the Tian-Pearl PNS bounds that bind at the margins of the stratum among the persons outside the fold, with the margins shrunk toward the pooled margins of the persons outside the fold by the weight of 20 persons per arm; the selected terms evaluated at the margins of the stratum among the persons of the fold, without pseudo-counts; strata with fewer than 2 persons of an arm in the fold are merged; the bounds of the two folds are averaged. For plain, oracle and random the folds are the folds of the cross-fitting of the same replicate.

### Inference

- **Plug-in interval.** 200 bootstrap resamples within arms with the memberships held fixed give the standard errors of the plug-in PNS endpoints; Imbens-Manski 95% interval (Imbens and Manski, 2004).
- **Interval of the estimate selected out of fold.** For every fold, 200 bootstrap resamples of the persons of the fold within arms, with memberships and selected terms held fixed, give the standard errors of the fold endpoints; the Imbens-Manski interval of every fold at the level 0.975; the interval is the intersection of the two fold intervals, with a coverage of at least 0.95 by the union bound whatever the dependence between the folds. An empty intersection counts as an interval that contains no value.
- **Secondary.** The Imbens-Manski 95% interval built on the averages of the fold endpoints with standard errors that treat the folds as independent, recorded to measure what the assumption of independent folds costs.

### Population quantities

For every replicate an evaluation sample of 20,000 covariate vectors is drawn from the same process. Exact conditionals q_c = sum_r P(r | c) q_r follow from the feature likelihoods and the regime margins (features are independent of the response types and of the choice given the regime). Full-covariate bounds L_C, U_C are the averages of the Tian-Pearl bounds at q_c; the population bounds of a stratifier built from the covariates average the bounds at q_t = sum_c w_ct q_c / sum_c w_ct over the evaluation sample with the stratifier's memberships (both fold encoders, each applied to a random half); the population bounds of the oracle follow exactly from the regime margins. The population bounds on PN and PS average the Tian-Pearl bounds on the numerators P(y'_x', x, y) and P(y_x, x', y') and divide by the population probabilities of the conditioning cells. The narrowest population width of k strata, k = 2, 3, 4, is the minimum over the sets of k pairs of terms of the mean, over the evaluation sample, of the smallest width of a chosen pair at q_c (`ibstrata.stratify.best_term_pairs`). I(C; V | T) = sum P(t, c) D(q_c || q_t) over the evaluation sample; kappa = 6 for the Tian-Pearl PNS bounds with equal allocation.

### Seeds

Master seed 20261117. The data of a replicate come from the seed sequence (master seed, experiment, purity, decoy, n, K, replicate), the bootstrap of the estimators of version 1 from the children of that sequence in the order of version 1, and the bootstraps of the added estimators and estimates from the children of the same sequence extended by the entry 1.

### Decision rules

A replicate in which an encoder fails (exception) is recorded with the error message and excluded from the means of that estimator only; the count of failures is reported per experiment. No replicate is rerun.

## 5. Outputs

- `results/tables/exp1-replicates.csv`, `exp2-replicates.csv`, `exp3-replicates.csv`: one row per cell, replicate and estimator.
- `results/tables/exp1-cells.csv`, `exp2-cells.csv`, `exp3-cells.csv`: means per cell and estimator.
- `results/tables/q1-validity.csv`, `q2-widths.csv`, `q2-paired.csv`, `q3-finite-sample.csv`, `q3-coverage.csv`, `q4-information-bound.csv`, `q5-two-group.csv`, `q6-mixed-sign.csv`: tables of the six questions (`scripts/summarize.py`).

## 6. Known limitations, stated in advance

- Binary treatment and outcome; the nonbinary bounds are verified separately against the exact linear program in the package tests and are not part of the simulation.
- The data-generating processes make covariates informative only through latent regimes; real covariates may carry graded heterogeneity without regimes. The features are independent binary variables given the regime, a structure that suits the logistic relevance model.
- The comparison covers stratifiers; estimators of covariate-assisted bounds without strata are not part of the study.
- The intervals treat endpoint estimates as approximately normal and hold the fitted encoders fixed in the bootstrap; undercoverage from either source is a result of Q3. In strata with response types of high purity the cell proportions lie near 0 and 1, where the normal approximation of a proportion is least accurate.
- Plug-in endpoints of maxima and minima are biased toward each other in finite samples; the study measures the bias and compares the plug-in estimate with the estimate with binding terms selected out of fold.
- The bound of Theorem 1 needs the covariate-level conditionals and is computed only because the simulation knows them.
- Equal allocation only; the constant of Theorem 1 grows as the smallest arm probability falls.
