# Scientific basis for the hybrid selective evolution

## Reused methods

The directional component reuses the Additive Noise Model implementation from
the `causal-learn` project rather than introducing another causal-discovery
algorithm. The package documentation defines the forward and backward
independence p-values and cites Hoyer et al. (2009). The method is appropriate
as an assumption-conditional direction test. It is not a universal causal
identifier.

The policy follows the selective-classification risk--coverage formulation. A
classifier may abstain on cases with high estimated error instead of converting
every score into a decision. The evaluation therefore reports both selective
risk and coverage, together with class-specific false authorization. Raw
accuracy alone would reward a conservative system that refuses nearly every
case.

Probability calibration uses standard isotonic regression. Logistic fitting
and isotonic calibration come from scikit-learn. MouseClaimBench contributes
the evidence representation, non-overridable claim semantics, temporal data
partitioning, and claim-aware evaluation. It does not claim novelty for ANM,
logistic regression, isotonic calibration, or selective classification.

## Why the combination is scientifically useful

A pure rule system is transparent but can amplify an unreliable binary
diagnostic. A pure classifier can exploit context but may authorize a claim
whose defining evidence type was never observed. The hybrid separates these
responsibilities. Statistical evidence determines confidence. Knowledge rules
define which evidence transitions are semantically legal. Abstention preserves
uncertainty when neither side is sufficiently supported.

The central falsifiable hypothesis is not that rules always beat learning. It
is that semantic constraints can be added without destroying useful coverage
or calibrated risk. This is narrower than the rejected version-1 hypothesis and
better aligned with a knowledge-based systems contribution.

## Known threats before execution

ANM is expected to degrade under confounding, selection, measurement error,
heteroscedasticity, and mechanisms outside additive noise. Isotonic calibration
can overfit when a claim has few calibration examples. Synthetic regimes cannot
represent the full distribution of scientific workflows. The knowledge profile
remains author-proposed and lacks independent expert validation. These are
limitations of the study design, not post-hoc explanations to be added only if
the endpoint fails.

## Primary sources

- `causal-learn` ANM documentation:
  https://causal-learn.readthedocs.io/en/latest/search_methods_index/Causal%20discovery%20methods%20based%20on%20constrained%20functional%20causal%20models/anm.html
- `causal-learn` software paper, JMLR 25 (2024):
  https://jmlr.org/papers/volume25/23-0970/23-0970.pdf
- Hoyer et al., nonlinear causal discovery with additive noise models:
  https://proceedings.neurips.cc/paper_files/paper/2008/file/f7664060cc52bc6f3d620bcedc94a4b6-Paper.pdf
- Geifman and El-Yaniv, selective classification:
  https://proceedings.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf

