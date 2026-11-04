import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rppg import PulseEstimator  # noqa: E402

FPS = 30.0
BASE_RGB = np.array([180.0, 120.0, 100.0])  # typischer Hautton
PULSE_DIR = np.array([0.33, 0.77, 0.53])  # Puls wirkt vor allem auf Grün (Hämoglobin)


def synthetic_skin(
    bpm: float,
    seconds: float = 15.0,
    seed: int = 0,
    flicker: bool = True,
    channel_mismatch=(0.0, 0.0, 0.0),
):
    """Simuliert die mittlere Hautfarbe pro Kamerabild.

    Enthält: Puls (~0,2 %), langsame Helligkeitsdrift, optional Lichtflackern
    im Pulsband, Sensorrauschen, unregelmäßige Bildrate und optional eine
    nicht-proportionale Kanalantwort auf Helligkeit (wie bei echten Kameras).
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    t = np.cumsum(rng.uniform(0.8, 1.2, n) / FPS)
    pulse = 0.002 * np.sin(2 * np.pi * bpm / 60 * t)
    illum = 0.03 * np.sin(2 * np.pi * 0.25 * t)
    if flicker:
        illum = illum + 0.01 * np.sin(2 * np.pi * 1.7 * t)  # 102 BPM, 5x stärker als der Puls
    gain = 1 + illum[:, None] * (1 + np.asarray(channel_mismatch))
    rgb = BASE_RGB * gain * (1 + pulse[:, None] * PULSE_DIR)
    rgb += rng.normal(0, 0.15, rgb.shape)
    return t, rgb


def run(est: PulseEstimator, t, rgb):
    result = None
    for i, (ti, ci) in enumerate(zip(t, rgb)):
        est.push(ti, ci)
        if i % 15 == 0:
            result = est.estimate() or result
    return est.estimate() or result


@pytest.mark.parametrize("bpm", [50, 65, 72, 90, 120, 150])
def test_estimates_heart_rate_despite_flicker(bpm):
    t, rgb = synthetic_skin(bpm, seed=bpm)
    result = run(PulseEstimator(window_sec=10), t, rgb)
    assert result is not None
    assert abs(result.bpm - bpm) < 2.0, result
    assert result.snr_db > 2


@pytest.mark.parametrize("bpm", [55, 75, 110])
def test_robust_to_nonproportional_channel_response(bpm):
    # Regression: Original-POS lag hier bei ~54 statt 75 BPM, weil die Reste
    # der Helligkeitsdrift das alpha-Gewicht verfälscht haben.
    t, rgb = synthetic_skin(bpm, seed=bpm, flicker=False, channel_mismatch=(-0.08, 0.11, -0.03))
    result = run(PulseEstimator(window_sec=10), t, rgb)
    assert result is not None
    assert abs(result.bpm - bpm) < 2.0, result


def test_green_method_works_in_steady_light():
    t, rgb = synthetic_skin(80, flicker=False)
    result = run(PulseEstimator(method="green"), t, rgb)
    assert abs(result.bpm - 80) < 2.0, result


def test_pure_noise_has_low_snr():
    rng = np.random.default_rng(1)
    t = np.arange(450) / FPS
    rgb = BASE_RGB + rng.normal(0, 0.3, (450, 3))
    result = run(PulseEstimator(), t, rgb)
    assert result.snr_db < 0


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        PulseEstimator(method="magic")


def test_no_estimate_before_min_duration():
    t, rgb = synthetic_skin(70, seconds=3)
    est = PulseEstimator(window_sec=10, min_sec=5)
    for ti, ci in zip(t, rgb):
        est.push(ti, ci)
    assert est.estimate() is None


def test_window_is_bounded():
    t, rgb = synthetic_skin(70, seconds=30)
    est = PulseEstimator(window_sec=10)
    for ti, ci in zip(t, rgb):
        est.push(ti, ci)
    assert est.buffered_sec <= 10.0


def test_reset_clears_state():
    t, rgb = synthetic_skin(70)
    est = PulseEstimator()
    for ti, ci in zip(t, rgb):
        est.push(ti, ci)
    assert est.estimate() is not None
    est.reset()
    assert est.buffered_sec == 0
    assert est.estimate() is None
