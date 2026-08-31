"""The forward model: what the tool does to the film.

Given a condition, these modules predict an outcome. That is exactly the knowledge the learner in
``active_learning`` must NOT have -- a boundary found with the help of this package would be partly
a re-drawing of our own assumptions rather than evidence about the film -- so nothing here is
importable from there, and ``tests/test_layering.py`` enforces it.

  constants.py      every number the models depend on, grouped by what kind of claim it makes
  thermal_model.py  (V, t) -> peak temperature and the full transient, over five candidate shapes

carries an ensemble rather than a model: the shape sets the geometry of the boundary, and the
campaign is built to stay informative whichever candidate turns out to be right.
"""
