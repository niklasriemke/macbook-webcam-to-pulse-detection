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
def test_pos_estimates_heart_rate_despite_flicker(bpm):
    # Lichtflackern im Pulsband: die Stärke von POS (der Grünkanal scheitert hier)
    t, rgb = synthetic_skin(bpm, seed=bpm)
    result = run(PulseEstimator(window_sec=10, method="pos"), t, rgb)
    assert result is not None
    assert abs(result.bpm - bpm) < 2.0, result
    assert result.snr_db > 2


@pytest.mark.parametrize("bpm", [55, 75, 110])
def test_pos_robust_to_nonproportional_channel_response(bpm):
    # Regression: Original-POS lag hier bei ~54 statt 75 BPM, weil die Reste
    # der Helligkeitsdrift das alpha-Gewicht verfälscht haben.
    t, rgb = synthetic_skin(bpm, seed=bpm, flicker=False, channel_mismatch=(-0.08, 0.11, -0.03))
    result = run(PulseEstimator(window_sec=10, method="pos"), t, rgb)
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


def test_pos_region_fusion_not_worse_than_pooled_average():
    # Wie im Programm: [alle Pixel, Stirn, Wange links, Wange rechts]. In den
    # meisten Fällen stört ein periodisches Artefakt mit Farbanteil (z. B.
    # wippende Haarsträhne) nur die Stirn – das kann POS nicht wegrechnen.
    hits_fused = hits_pooled = 0
    for seed in range(3):
        rng = np.random.default_rng(seed)
        for bpm in (58, 85):
            t, clean = synthetic_skin(bpm, seconds=12, seed=seed * 10 + bpm, flicker=False)
            for d in ([-0.6, 0.2, 0.9], [0.5, -0.8, 0.2], [1.0, 0.3, -0.5]):
                art = 0.004 * np.sin(2 * np.pi * rng.uniform(0.9, 2.5) * t)
                forehead = clean * (1 + art[:, None] * np.array(d))
                cl = clean + rng.normal(0, 0.4, clean.shape)
                cr = clean + rng.normal(0, 0.4, clean.shape)
                pooled = (forehead + cl + cr) / 3
                regions = np.stack([pooled, forehead, cl, cr], axis=1)
                hits_fused += abs(run(PulseEstimator(method="pos"), t, regions).bpm - bpm) < 3
                hits_pooled += abs(run(PulseEstimator(method="pos"), t, pooled).bpm - bpm) < 3
    assert hits_fused >= hits_pooled, (hits_fused, hits_pooled)


def test_tracking_prefers_peak_near_previous_estimate():
    from rppg import pick_peak

    freqs = np.linspace(0.7, 3.5, 500)
    spec = np.exp(-((freqs - 1.2) ** 2) / 0.001) * 0.8 + np.exp(-((freqs - 0.9) ** 2) / 0.001)
    spec /= spec.sum()
    free, _ = pick_peak(freqs, spec, 300, 30.0)
    tracked, _ = pick_peak(freqs, spec, 300, 30.0, prev_bpm=72.0)
    assert abs(free - 54) < 1  # stärkster Peak (Artefakt bei 0,9 Hz)
    assert abs(tracked - 72) < 1  # fast gleich starker Peak beim bisherigen Wert
    # Ein klar dominanter neuer Peak setzt sich trotzdem durch
    spec2 = np.exp(-((freqs - 1.2) ** 2) / 0.001) * 0.3 + np.exp(-((freqs - 0.9) ** 2) / 0.001)
    assert abs(pick_peak(freqs, spec2 / spec2.sum(), 300, 30.0, prev_bpm=72.0)[0] - 54) < 1


def test_min_duration_never_exceeds_window():
    t, rgb = synthetic_skin(70, seconds=15, flicker=False)
    est = PulseEstimator(window_sec=10, min_sec=10)
    for ti, ci in zip(t, rgb):
        est.push(ti, ci)
    assert est.estimate() is not None


def test_exposure_jump_restarts_buffer_but_keeps_display_value():
    t, rgb = synthetic_skin(70, seconds=30, flicker=False)
    rgb = rgb.copy()
    rgb[t > 15] *= 1.4  # Belichtungsautomatik springt
    est = PulseEstimator(window_sec=20, min_sec=10)
    before = None
    for ti, ci in zip(t, rgb):
        est.push(ti, ci)
        if before is None and ti > 14.5:
            before = est.estimate()
    assert est.restarts == 1
    after = est.estimate()  # nur Daten nach dem Sprung im Puffer
    assert est.buffered_sec < 15.5
    assert abs(before.bpm - 70) < 2 and abs(after.bpm - 70) < 2


def test_gap_restarts_buffer():
    t, rgb = synthetic_skin(70, seconds=20, flicker=False)
    keep = (t < 8) | (t > 10)  # 2 s Aussetzer
    est = PulseEstimator()
    for ti, ci in zip(t[keep], rgb[keep]):
        est.push(ti, ci)
    assert est.restarts == 1
