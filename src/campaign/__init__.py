"""Running the real experiment: what was fired, what came back, and what it says.

  plan.py       build a batch of conditions and write it as a plan the lab can fire
  results.py    ingest measured outcomes, refusing to guess when the record is ambiguous
  reporting.py  put a fitted boundary back into volts, milliseconds and degrees

``reporting`` is the only module in the repository that imports both the learner and the physics,
and it does so in that order: the boundary is FOUND without the thermal model, then LABELLED with
the temperature that model predicts. Keeping the two steps in separate packages is what makes the
separation checkable rather than merely claimed.
"""
