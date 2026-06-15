"""Bayesian model-comparison + design-stage power study for HZO crystallization kinetics.

Prototype modules supporting the plan's REVISION 1 #1 go/no-go gate:
- forward_models: candidate kinetic models (alpha(u; theta))
- evidence: grid-quadrature marginal likelihood / Bayes factor (small-n safe)
- power_study: preposterior P(correct model selection) vs n and noise
"""
