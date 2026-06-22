"""Crystallization-boundary coordinate-discovery framework.

Given flash-anneal shots with crystallinity outcomes, determine which coordinate chart most
simply explains the crystallization boundary (chart comparison by log marginal likelihood)
and run a design-stage power study across measurement-readout types.

Modules:
  synthetic.py  -- forward model: (V,t) -> trace -> descriptors -> crystallinity readout
  charts.py     -- candidate coordinate charts on the (V,t) manifold
  compare.py    -- fit one GP per chart, score by LML, tempered weights
  power.py      -- Monte-Carlo power study across sample size, noise, and readout type
"""
