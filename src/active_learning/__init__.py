"""The learner: what we infer from measurements, and what to fire next.

These modules see coordinates in the design box and readings taken there. They never see a
temperature, a cooling law or an activation energy. That restriction is the point -- it is what
lets the boundary they report count as evidence about the film rather than an echo of the thermal
model in ``physics`` -- and ``tests/test_layering.py`` asserts it over the transitive import
closure, because a single convenience import would undo it silently.

  surrogate.py    exact GP on the LATENT logit field, Matern 5/2 ARD, MAP hyperparameters
  acquisition.py  acquisition scores, and fantasy-conditioned selection of a whole batch

Fitting happens on the logit of a self-normalized reading because the readout has no calibrated
zero or gain, so only the SHAPE of the response across conditions is trustworthy.
"""
