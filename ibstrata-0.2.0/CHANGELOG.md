# Changelog

Versions follow Semantic Versioning (<https://semver.org/>) in the notation of PEP 440: the third number changes with fixes, the second with new features, the first with a change that breaks earlier code. Before 1.0.0 a new second number can also change an interface, and its entry says so. Development versions between releases end in `.devN`. Every release is tagged in Git as `ibstrata-vX.Y.Z`.

## 0.2.0 (2026-11-18)

- `ibstrata.simulate.TapRegimeDGP` and `TapTaskModel`: reach and practice regimes that persons express in tap-order blocks (Thurstone-type orders on the blocks of a design, with an anchor card), with response types, choice and outcome by the combined regime; the features equal the statistics of `ibstrata.orders`. `dominant_type_probs` builds type compositions from a table of dominant types.
- `ibstrata.relevance`: multinomial logistic regression by penalized maximum likelihood (`fit_softmax`).
- `ibstrata.stratify.LogisticRelevance`: relevance model that fits the observed cell on the features within every design arm and assembles the arms with the design probabilities.
- `BottleneckEncoder(relevance="logistic")`: the bottleneck on the fitted distributions of the logistic relevance model; a new person is assigned by the stationary equation at the fitted distribution of the person. The default `relevance="neighbors"` is the encoder of 0.1.0.
- Comparison stratifiers: `KMeansEncoder(weighted=True)` (k-means on the features scaled by the relevance weights), `FittedKMeansEncoder` (k-means on the fitted distributions) and `EffectQuantileEncoder` (quantile groups of the fitted effect).
- `selected_term_bounds`: stratified Tian-Pearl PNS bounds with the binding term of every stratum selected out of fold (stratum margins of the persons outside the fold, shrunk toward their pooled margins), fold bounds averaged into the estimate, and the intersection of the Imbens-Manski intervals of the folds as the interval, valid whatever the dependence between the folds; `pns_terms` gives the terms of the bounds; `CrossFit.training_memberships` gives the memberships of the persons every fold encoder was fitted on.
- `pair_widths`, `best_term_pairs` and `TermPairEncoder`: the widths of the sixteen pairs of a lower and an upper term of the PNS bounds, the set of k pairs that gives the narrowest bounds of k strata, and the stratifier that assigns every person to the narrowest of the chosen pairs at the fitted distribution of the person.
- Documentation: the lower bounds of `preservation_sharp` and `replacement_sharp` are the lower bounds of Shu, Wang and Li (2026, Theorem 4) with one hypothetical term; the README table states the status of every single-term bound.
- Tests: the exact identity for the width lost by a stratifier, the partition by binding terms, the narrowest bounds of k strata against the enumeration of all partitions, the logistic relevance model, the added encoders and the selected-term bounds with their interval.

## 0.1.0 (2026-11-17)

First numbered version.

- `ibstrata.bounds`: margins of combined experimental and observational data with the compatibility check and the weighted l1 projection onto compatible margins; events of the counterfactual language; exact bounds by linear programming over the joint of choice and potential outcomes, with feasible endpoint witnesses; Tian-Pearl bounds on PNS, PN and PS; monotonicity; the bounds of Li and Pearl (2024, Theorems 4-7, with the attained lower bounds of Theorems 4 and 5); the closed-form PNS bounds of Shu, Wang and Li (2026, Theorem 2); stratified bounds with soft weights; the arm-augmented outcome; the tightness constant and the certificate of Theorem 1.
- `ibstrata.info`: entropy, divergence and (conditional) mutual information of discrete distributions.
- `ibstrata.ib`: the information bottleneck by iteration, deterministic annealing, agglomeration and the deterministic variant.
- `ibstrata.mib`: the multivariate information bottleneck (the generic engine of Theorem 7.1 of Friedman et al., 2001) and the reach-practice specification on sparse nearest-neighbor smoothers.
- `ibstrata.stratify`: the design of a two-stage randomized preference study, arm-balanced smoothed relevance, bottleneck, reach-practice, fixed and k-means encoders, pooling of small strata, honest cross-fitting with aligned fold strata, bias correction, the bootstrap within arms and Imbens-Manski intervals.
- `ibstrata.simulate`: populations of regimes with declared response types, decoy features and exact posteriors.
- `ibstrata.io`: the reader of Card Surveys exports (format `cardsurveys-export-1`, folder or ZIP).
- `ibstrata.orders`: person-level statistics of tap-order blocks: anchor shares, precedence shares, Copeland orders, consistency, cyclic triads, Kendall's tau-b against a reference order and the calibration of gate answers.
