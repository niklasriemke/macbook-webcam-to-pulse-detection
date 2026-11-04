"""Regressionstest mit einer echten Aufnahme gegen eine Whoop als Referenz.

messungen/roh.csv: MacBook Air, abends, Raumlicht, ruhig sitzend (pulse.py --record).
messungen/whoop_referenz.csv: Live-Herzfrequenz der Whoop, aus einer
Bildschirmaufnahme abgelesen. Die Handy-Uhr ging einige Sekunden vor; geprüft
wird deshalb über einen Bereich plausibler Versätze.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auswertung import compare, load_recording, load_reference, replay  # noqa: E402

REC = ROOT / "messungen" / "roh.csv"
REF = ROOT / "messungen" / "whoop_referenz.csv"
pytestmark = pytest.mark.skipif(not (REC.exists() and REF.exists()), reason="Messdaten fehlen")


@pytest.fixture(scope="module")
def rec():
    return load_recording(str(REC))


@pytest.mark.parametrize("offset", [0, 3, 6])
def test_default_settings_match_whoop(rec, offset):
    res = compare(replay(rec), *load_reference(str(REF), rec, offset))
    assert res["n"] >= 60
    assert res["mae"] < 5.0, res
    assert res["treffer_5"] >= 0.8, res


def test_green_beats_previous_pos_default(rec):
    ref = load_reference(str(REF), rec, 3)
    green = compare(replay(rec, "green", 20), *ref)
    old = compare(replay(rec, "pos", 10, track=True), *ref)
    assert green["mae"] < old["mae"]
