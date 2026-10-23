"""Electrical features of the flashed HZO films from the Keithley 4200A exports.

One round of measurements is a folder of ``Sample<NN>/`` folders, each holding the PUND, P-V and
endurance workbooks of ten or so capacitors. The pipeline runs in phases, one module each:

1. ``locate``    find the workbooks, read their Settings, and group runs per (capacitor, state).
2. ``pund``, ``pv``, ``endurance``    one run -> its features and the traces the QC report plots.
3. ``aggregate``    per-capacitor rows -> per-sample statistics with repeat scatter.
4. ``report``    the QC page with every capacitor of every sample overlaid.
5. ``pipeline``    runs the phases for one round; ``run_electrical_features.py`` is the
   command line.

The group's summary sheet is the regression target: its Psw, Pr and kappa definitions are
reproduced here and pinned by ``tests/test_electrical_round2.py``.
"""
