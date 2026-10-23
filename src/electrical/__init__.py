"""Electrical features of the flashed HZO films, from the Keithley 4200A exports.

One round of measurements is a folder of ``Sample<NN>/`` folders, each holding the PUND, P-V
and endurance workbooks of ten or so capacitors. ``process_round`` runs the steps in order:

1. ``find_measurement_files``      find the exports; sample, capacitor and kind from their names.
2. ``read_keithley_export``        open each workbook once: Settings blocks and run sheets.
3. ``read_measurement_runs``       read the sheets as runs; pair them per capacitor and state.
4. ``extract_pund_features``       Psw, O / T charge split, coercive voltage, leakage, κ.
   ``extract_pv_features``         Pr, Ec, pinching, κ from the Q-V slope.
   ``extract_endurance_features``  wake-up and fatigue over 10⁶ cycles.
5. ``build_feature_tables``        per-capacitor and per-sample tables; the campaign table.
6. ``flag_issues``                 problems found on the way, and outlying capacitors.
7. ``write_qc_report``             the QC page with every capacitor overlaid.

``convert_units`` holds the capacitor geometry and the charge and capacitance conversions the
extractors share. The group's summary sheet is the regression target: its Psw, Pr and κ
definitions are reproduced here and pinned by ``tests/test_electrical_round2.py``.
"""
