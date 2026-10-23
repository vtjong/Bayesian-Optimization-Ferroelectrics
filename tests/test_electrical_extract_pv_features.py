"""Pr, Ec+ and Ec- against the lines the group's script draws on its Q-V plots.

The targets are read off the group's ``Analysis_Plots/PV_RxCy.png`` (dotted lines for Ec, the
marked V = 0 point for Pr), so they carry the reading error of a plot, about 0.03.
"""

import numpy as np
import pytest

from electrical.extract_pv_features import extract_pv_features
from electrical.find_measurement_files import parse_file_name
from electrical.read_measurement_runs import read_runs
from paths import round_raw_dir

PLOT_READING = 0.03


def _features(round_number: int, sample: str, name: str):
    runs = read_runs(parse_file_name(round_raw_dir(round_number) / sample / name))
    return [extract_pv_features(run)[0] for run in runs]


@pytest.mark.parametrize(
    ("round_number", "sample", "name", "group_lines"),
    [
        (2, "Sample16", "PV_pris_1e6_R9C9_30um.xls", [(1.18, -0.75), (1.32, -0.92)]),
        (4, "Sample26", "PV_pris_1e6_R9C9_30um.xls", [(1.22, -0.57), (1.35, -0.82)]),
        (3, "Sample25", "PV_pris_1e6_R9C9_30um.xls", [(0.27, -0.12), (0.60, -0.30)]),
    ],
)
def test_coercive_voltages_sit_on_the_groups_lines(round_number, sample, name, group_lines):
    for features, (ec_pos, ec_neg) in zip(_features(round_number, sample, name), group_lines):
        assert features.coercive.ec_pos_v == pytest.approx(ec_pos, abs=PLOT_READING)
        assert features.coercive.ec_neg_v == pytest.approx(ec_neg, abs=PLOT_READING)


def test_pr_is_read_where_the_group_marks_it():
    pristine, cycled = _features(2, "Sample16", "PV_pris_1e6_R9C9_30um.xls")
    assert pristine.remanence.pr_uc_cm2 == pytest.approx(10.2, abs=0.3)
    assert cycled.remanence.pr_uc_cm2 == pytest.approx(15.4, abs=0.3)


def test_a_linear_film_whose_charge_never_crosses_zero_on_the_first_rise_has_no_ec_pos():
    (features,) = _features(2, "Sample14", "PV_pris_R9C9_30um.xls")
    assert np.isnan(features.coercive.ec_pos_v)
    assert features.coercive.ec_neg_v == pytest.approx(0.17, abs=PLOT_READING)
