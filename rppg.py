"""Remote photoplethysmography (rPPG): Puls aus Farbschwankungen der Haut.

Bei jedem Herzschlag ändert sich das Blutvolumen in der Haut minimal, wodurch
sich ihre Farbe (vor allem im Grünkanal) um Bruchteile eines Prozents ändert.
Dieses Modul nimmt die mittlere RGB-Farbe einer Hautregion pro Kamerabild
entgegen und schätzt daraus die Herzfrequenz.

Standardverfahren ist der POS-Algorithmus ("Plane-Orthogonal-to-Skin") von
Wang, den Brinker, Stuijk & de Haan (2017), IEEE TBME 64(7). Er projiziert die
normierten RGB-Signale auf eine Ebene, die orthogonal zum Hautton liegt, und
unterdrückt dadurch Helligkeitsänderungen (Bewegung, Lichtflackern) deutlich
besser als ein reines Grünkanal-Signal. Details zur hier verwendeten Variante
stehen in `pos()`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

MIN_BPM = 42.0
MAX_BPM = 210.0

# Projektionsmatrix aus dem POS-Paper (Zeilen: S1 = G - B, S2 = -2R + G + B)
_POS_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpoliert unregelmäßig abgetastete Samples auf ein festes Raster.

    Webcams liefern selten exakt konstante Bildraten, die FFT setzt aber
    gleichmäßige Abtastung voraus.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    xu = np.column_stack([np.interp(tu, t, x[:, c]) for c in range(x.shape[1])])
    return tu, xu


def _moving_mean(x: np.ndarray, l: int) -> np.ndarray:
    kernel = np.ones(l) / l
    pad = (l // 2, l - 1 - l // 2)
    return np.column_stack([np.convolve(np.pad(x[:, c], pad, mode="edge"), kernel, "valid") for c in range(x.shape[1])])


def pos(rgb: np.ndarray, fs: float, win_sec: float = 1.6) -> np.ndarray:
    """POS-Pulssignal aus einer (N, 3)-Folge mittlerer RGB-Werte.

    Abweichung vom Original-Paper: Die beiden Projektionen werden vor der
    Bestimmung des Mischfaktors alpha bandpassgefiltert (wie bei CHROM).
    Echte Kameras reagieren wegen Gamma-Kurve, Weißabgleich und Kompression
    nicht exakt proportional in allen Kanälen auf Helligkeitsänderungen. Die
    Reste langsamer Lichtschwankungen sind dann so groß wie der Puls selbst
    und würden alpha sonst verfälschen.
    """
    n = rgb.shape[0]
    l = max(2, int(round(win_sec * fs)))
    if np.any(rgb <= 0) or n < l:
        return np.zeros(n)
    cn = rgb / _moving_mean(rgb, l) - 1.0  # zeitliche Normierung
    s = cn @ _POS_PROJECTION.T  # (n, 2)
    s = np.column_stack([bandpass(s[:, 0], fs), bandpass(s[:, 1], fs)])
    h = np.zeros(n)
    for start in range(0, n - l + 1):
        seg = s[start : start + l]
        std2 = seg[:, 1].std()
        alpha = seg[:, 0].std() / std2 if std2 > 1e-12 else 0.0
        p = seg[:, 0] + alpha * seg[:, 1]
        h[start : start + l] += p - p.mean()
    return h


def green(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Klassisches Grünkanal-Signal (Verkruysse et al. 2008).

    Bei ruhigem, gleichmäßigem Licht oft sehr sauber, aber anfällig für
    Bewegung und Helligkeitsschwankungen.
    """
    g = rgb[:, 1]
    return g / g.mean() - 1.0


METHODS = {"pos": pos, "green": green}


def bandpass(x: np.ndarray, fs: float, lo_bpm: float = MIN_BPM, hi_bpm: float = MAX_BPM) -> np.ndarray:
    """Butterworth-Bandpass auf den plausiblen Herzfrequenzbereich."""
    nyq = fs / 2.0
    hi = min(hi_bpm / 60.0, 0.95 * nyq)
    b, a = butter(3, [lo_bpm / 60.0, hi], btype="band", fs=fs)
    return filtfilt(b, a, x)


def spectrum_peak(x: np.ndarray, fs: float) -> tuple[float, float]:
    """Dominante Frequenz im Pulsband als (bpm, SNR in dB).

    Der SNR folgt de Haan & Jeanne (2013): Leistung in einem schmalen Fenster
    um die Grundfrequenz und ihre erste Oberwelle im Verhältnis zur restlichen
    Leistung im Pulsband. Werte > 0 dB deuten auf ein brauchbares Signal hin.
    """
    x = x - x.mean()
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(max(n * 8, 2048))))
    spec = np.abs(np.fft.rfft(x * np.hanning(n), nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)

    band = (freqs >= MIN_BPM / 60.0) & (freqs <= MAX_BPM / 60.0)
    if not band.any() or spec[band].sum() <= 0:
        return float("nan"), float("-inf")
    idx = np.flatnonzero(band)
    k = idx[np.argmax(spec[idx])]

    # Parabolische Interpolation für Auflösung unterhalb des FFT-Rasters
    f_peak = freqs[k]
    if 0 < k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = a - 2 * b + c
        if denom != 0:
            f_peak = freqs[k] + 0.5 * (a - c) / denom * (freqs[1] - freqs[0])

    # Hauptkeule des Hann-Fensters ist ±2/T breit – so viel gehört zum Peak
    tol = max(0.1, 2.0 * fs / n)  # Hz
    signal_mask = (np.abs(freqs - f_peak) <= tol) | (np.abs(freqs - 2 * f_peak) <= 2 * tol)
    sig = spec[band & signal_mask].sum()
    noise = spec[band & ~signal_mask].sum()
    snr = 10 * np.log10(sig / noise) if noise > 0 else float("inf")
    return float(f_peak * 60.0), float(snr)


@dataclass
class PulseResult:
    bpm: float  # geglätteter Wert für die Anzeige
    raw_bpm: float  # Schätzung nur aus dem aktuellen Fenster
    snr_db: float
    signal: np.ndarray  # gefiltertes Pulssignal (für die Kurve)
    fs: float


class PulseEstimator:
    """Sammelt RGB-Samples und schätzt fortlaufend die Herzfrequenz.

    window_sec: Länge des Analysefensters. Länger = stabiler, aber träger.
    min_sec:    Mindestdauer, bevor die erste Schätzung ausgegeben wird.
    min_snr_db: Schätzungen darunter fließen nicht in den geglätteten Wert ein.
    """

    def __init__(
        self,
        window_sec: float = 10.0,
        min_sec: float = 8.0,
        fs: float = 30.0,
        min_snr_db: float = -3.0,
        smoothing: int = 7,
        method: str = "pos",
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"Unbekannte Methode {method!r}, erlaubt: {', '.join(METHODS)}")
        self._extract = METHODS[method]
        self.window_sec = window_sec
        self.min_sec = min_sec
        self.fs = fs
        self.min_snr_db = min_snr_db
        self._t: deque[float] = deque()
        self._rgb: deque[np.ndarray] = deque()
        self._history: deque[float] = deque(maxlen=smoothing)

    def reset(self) -> None:
        self._t.clear()
        self._rgb.clear()
        self._history.clear()

    @property
    def buffered_sec(self) -> float:
        return self._t[-1] - self._t[0] if len(self._t) > 1 else 0.0

    def push(self, t: float, rgb) -> None:
        """Fügt ein Sample hinzu: Zeitstempel in Sekunden, mittleres (R, G, B)."""
        if self._t and t <= self._t[-1]:
            return  # doppelte/rückwärtslaufende Zeitstempel ignorieren
        self._t.append(float(t))
        self._rgb.append(np.asarray(rgb, dtype=float))
        while self._t and self._t[-1] - self._t[0] > self.window_sec:
            self._t.popleft()
            self._rgb.popleft()

    def estimate(self) -> PulseResult | None:
        if self.buffered_sec < self.min_sec:
            return None
        _, rgb = resample_uniform(np.array(self._t), np.vstack(self._rgb), self.fs)
        if len(rgb) < int(2 * self.fs):
            return None
        sig = bandpass(self._extract(rgb, self.fs), self.fs)
        raw_bpm, snr = spectrum_peak(sig, self.fs)
        if np.isnan(raw_bpm):
            return None
        if snr >= self.min_snr_db:
            self._history.append(raw_bpm)
        smoothed = float(np.median(self._history)) if self._history else float("nan")
        return PulseResult(bpm=smoothed, raw_bpm=raw_bpm, snr_db=snr, signal=sig, fs=self.fs)
