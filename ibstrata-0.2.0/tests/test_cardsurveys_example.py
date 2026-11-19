"""The example export of the Card Surveys tap-order demonstration (40 simulated respondents),
read against the shares of pair observations computed by the R package eidemetrics
(``read_cardsurveys(..., duplicates = TRUE)``), stored in ``cardsurveys_tap_shares.csv``."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from ibstrata.io import read_cardsurveys
from ibstrata.orders import tap_task

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def shares():
    s = pd.read_csv(DATA / "cardsurveys_tap_shares.csv")
    return s.set_index(["response_uuid", "judgment"])["share"]


def test_anchor_and_precedence_shares_agree_with_the_r_reader(shares):
    ex = read_cardsurveys(DATA / "cardsurveys_tap").submitted()
    task = tap_task(ex, "value")
    assert task.anchor == "ANCHOR"
    anchor = task.anchor_share()
    assert len(anchor) == 40
    for rid, row in anchor.iterrows():
        for item, v in row.items():
            assert v == pytest.approx(shares[(rid, f"{item}<ANCHOR")])
    prec = task.precedence()
    for rid, row in prec.iterrows():
        for col, v in row.items():
            a, b = col.split(">")
            assert v == pytest.approx(shares[(rid, f"{a}<{b}")])


def test_zip_and_folder_exports_are_read_alike(tmp_path):
    archive = shutil.make_archive(str(tmp_path / "tap"), "zip", root_dir=DATA,
                                  base_dir="cardsurveys_tap")
    folder = read_cardsurveys(DATA / "cardsurveys_tap")
    packed = read_cardsurveys(archive)
    assert folder.description["format"] == packed.description["format"] == "cardsurveys-export-1"
    pd.testing.assert_frame_equal(folder.blocks, packed.blocks)
    assert folder.module("value")["anchor"]["code"] == "ANCHOR"
