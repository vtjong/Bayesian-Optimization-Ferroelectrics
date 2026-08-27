"""Evidence for the design decisions, as opposed to the machinery that executes them.

Nothing here runs during a real campaign. These modules exist to answer "how do you know that
design was worth firing" with a number rather than an argument -- by building worlds in which the
campaign's assumptions are wrong and measuring what each candidate design would still conclude.

  worlds.py       randomized worlds that hold the 30 measured temperatures exact and vary the rest
  adversarial.py  ground truths the seed design deliberately did NOT come from
  designs.py      candidate seed designs, including ones using only the measured table
  evaluate.py     what a design would let us conclude, and how wrong that could be
  picker.py       the earlier synthetic-study stack: sklearn GP, entropy acquisitions, and the
                  heteroscedastic readout-noise model used to generate synthetic readings

``picker`` is deliberately NOT merged into ``active_learning``. It produced the numbers the design
studies report, and quietly rewriting it onto the newer surrogate would invalidate those numbers
while looking like a refactor. Replacing it is a re-analysis to be run and stated as one.
"""
