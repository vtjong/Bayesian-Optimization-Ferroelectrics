"""Crystallization-boundary coordinate-discovery framework (rebuild of the boss's fla-ml).

Layer 3 of the four-layer plan: given shots with crystallinity outcomes, determine which
coordinate chart most simply explains the boundary (chart comparison by log marginal
likelihood) and run a design-stage power study across readout types.

Modules (mirroring the boss's file names so they map 1:1 to his framework):
  synthetic.py  -- forward model: (V,t) -> trace -> descriptors -> crystallinity readout
  charts.py     -- candidate coordinate charts on the (V,t) manifold
  compare.py    -- fit one GP per chart, score by LML, tempered weights
  power.py      -- Monte-Carlo power study across sample size, noise, and readout type
"""
