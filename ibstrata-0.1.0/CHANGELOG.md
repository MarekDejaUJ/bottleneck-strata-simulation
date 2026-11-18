# Changelog

Versions follow Semantic Versioning (<https://semver.org/>) in the notation of PEP 440: the third number changes with fixes, the second with new features, the first with a change that breaks earlier code. Before 1.0.0 a new second number can also change an interface, and its entry says so. Development versions between releases end in `.devN`. Every release is tagged in Git as `ibstrata-vX.Y.Z`.

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
