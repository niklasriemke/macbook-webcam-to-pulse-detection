"""Remote photoplethysmography (rPPG): Puls aus Farbschwankungen der Haut.

Bei jedem Herzschlag ändert sich das Blutvolumen in der Haut minimal, wodurch
sich ihre Farbe (vor allem im Grünkanal) um Bruchteile eines Prozents ändert.
Dieses Modul nimmt die mittlere RGB-Farbe einer Hautregion pro Kamerabild
entgegen und schätzt daraus die Herzfrequenz.

Zwei Verfahren stehen zur Wahl:

- `green` (Standard): der Grünkanal, in dem das Hämoglobin am stärksten
  absorbiert. In der Vergleichsmessung gegen eine Whoop (MacBook Air, Abendlicht,
  ruhig sitzend; siehe tests/test_real_recording.py) war er klar am genauesten,
  weil Rot- und vor allem Blaukanal bei wenig Licht stark rauschen.
- `pos`: POS-Algorithmus ("Plane-Orthogonal-to-Skin") von Wang, den Brinker,
  Stuijk & de Haan (2017), IEEE TBME 64(7). Unterdrückt Helligkeitsänderungen
  (Bewegung, Lichtflackern) besser, mischt dafür aber das Rauschen aller drei
  Kanäle hinein. Details zur hier verwendeten Variante stehen in `pos()`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

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


def power_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Leistungsspektrum im Pulsband (Frequenzen in Hz), normiert auf Summe 1."""
    x = x - x.mean()
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(max(n * 8, 2048))))
    spec = np.abs(np.fft.rfft(x * np.hanning(n), nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (freqs >= MIN_BPM / 60.0) & (freqs <= MAX_BPM / 60.0)
    f, s = freqs[band], spec[band]
    total = s.sum()
    return f, (s / total if total > 0 else s)


# Bild-zu-Bild-Änderung der Hautfarbe, ab der ein Sprung angenommen wird. In der
# Whoop-Vergleichsmessung lag normales Rauschen bei max. 2,1 %, die Belichtungs-
# automatik der MacBook-Kamera sprang um 3–8 % pro Bild.
MAX_JUMP = 0.03
MAX_GAP_SEC = 1.0  # längere Datenlücken unterbrechen das Signal

TRACK_RATIO = 0.5  # Nebenpeaks ab diesem Anteil des Maximums kommen fürs Tracking in Frage


def pick_peak(
    freqs: np.ndarray, spec: np.ndarray, n: int, fs: float, prev_bpm: float | None = None
) -> tuple[float, float]:
    """Herzfrequenz-Peak als (bpm, SNR in dB).

    Ohne prev_bpm gewinnt der stärkste Peak. Mit prev_bpm wird unter allen
    Peaks, die mindestens TRACK_RATIO des stärksten erreichen, der zum
    bisherigen Wert nächstgelegene genommen. Das verhindert Springen zwischen
    zwei ähnlich starken Peaks (z. B. Puls vs. Bewegungs-/Atemartefakt),
    lässt aber einen deutlich dominanten neuen Peak durch.

    Der SNR folgt de Haan & Jeanne (2013): Leistung um Grundfrequenz und erste
    Oberwelle im Verhältnis zur restlichen Leistung im Pulsband.
    """
    if len(spec) == 0 or spec.sum() <= 0:
        return float("nan"), float("-inf")
    k_max = int(np.argmax(spec))
    k = k_max
    if prev_bpm is not None and np.isfinite(prev_bpm):
        peaks, _ = find_peaks(spec)
        cands = [int(p) for p in peaks if spec[p] >= TRACK_RATIO * spec[k_max]] + [k_max]
        k = min(cands, key=lambda p: abs(freqs[p] * 60.0 - prev_bpm))

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
    sig = spec[signal_mask].sum()
    noise = spec[~signal_mask].sum()
    snr = 10 * np.log10(sig / noise) if noise > 0 else float("inf")
    return float(f_peak * 60.0), float(snr)


def spectrum_peak(x: np.ndarray, fs: float) -> tuple[float, float]:
    """Stärkster Peak eines einzelnen Signals als (bpm, SNR in dB)."""
    f, s = power_spectrum(x, fs)
    return pick_peak(f, s, len(x), fs)


@dataclass
class PulseResult:
    bpm: float  # geglätteter Wert für die Anzeige
    raw_bpm: float  # Schätzung nur aus dem aktuellen Fenster
    snr_db: float
    signal: np.ndarray  # gefiltertes Pulssignal der besten Region (für die Kurve)
    fs: float


class PulseEstimator:
    """Sammelt RGB-Samples und schätzt fortlaufend die Herzfrequenz.

    Pro Sample können mehrere Hautregionen übergeben werden (Array (k, 3)).
    Jede Region liefert ein eigenes Pulssignal; die Spektren werden nach ihrer
    Schärfe gewichtet kombiniert. So verdirbt eine gestörte Region (Haare,
    Schatten, Bewegung am Rand) nicht das Gesamtergebnis.

    window_sec: Länge des Analysefensters. Länger = stabiler, aber träger.
    min_sec:    Mindestdauer, bevor die erste Schätzung ausgegeben wird.
    min_snr_db: Schätzungen darunter fließen nicht in den geglätteten Wert ein.
    """

    def __init__(
        self,
        window_sec: float = 20.0,
        min_sec: float = 10.0,
        fs: float = 30.0,
        min_snr_db: float = -3.0,
        smoothing: int = 9,
        method: str = "green",
        track: bool = False,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"Unbekannte Methode {method!r}, erlaubt: {', '.join(METHODS)}")
        self._extract = METHODS[method]
        self.window_sec = window_sec
        self.min_sec = min(min_sec, 0.9 * window_sec)  # sonst wird bei min_sec >= window nie geschätzt
        self.fs = fs
        self.min_snr_db = min_snr_db
        self.track = track
        self._t: deque[float] = deque()
        self._rgb: deque[np.ndarray] = deque()
        self._history: deque[float] = deque(maxlen=smoothing)
        self.restarts = 0  # wie oft der Puffer wegen Sprung/Lücke neu begonnen wurde

    def reset(self) -> None:
        self._t.clear()
        self._rgb.clear()
        self._history.clear()

    @property
    def buffered_sec(self) -> float:
        return self._t[-1] - self._t[0] if len(self._t) > 1 else 0.0

    def push(self, t: float, rgb) -> None:
        """Fügt ein Sample hinzu: Zeitstempel in Sekunden, (R, G, B) oder (k, 3)."""
        c = np.atleast_2d(np.asarray(rgb, dtype=float))
        if self._rgb and self._rgb[-1].shape != c.shape:
            self.reset()  # Anzahl Regionen hat sich geändert
        if self._t and t <= self._t[-1]:
            return  # doppelte/rückwärtslaufende Zeitstempel ignorieren
        if self._t:
            prev = self._rgb[-1][0].sum()
            jump = abs(c[0].sum() - prev) / prev if prev > 0 else 0.0
            if t - self._t[-1] > MAX_GAP_SEC or jump > MAX_JUMP:
                # Lücke oder Sprung (Belichtungsautomatik, Kopfbewegung): Puffer neu
                # aufbauen. Der geglättete Wert (history) bleibt erhalten.
                self._t.clear()
                self._rgb.clear()
                self.restarts += 1
        self._t.append(float(t))
        self._rgb.append(c)
        while self._t and self._t[-1] - self._t[0] > self.window_sec:
            self._t.popleft()
            self._rgb.popleft()

    def estimate(self) -> PulseResult | None:
        if self.buffered_sec < self.min_sec:
            return None
        stack = np.stack(self._rgb)  # (N, k, 3)
        n_raw, k, _ = stack.shape
        _, flat = resample_uniform(np.array(self._t), stack.reshape(n_raw, k * 3), self.fs)
        n = len(flat)
        if n < int(2 * self.fs):
            return None
        regions = flat.reshape(n, k, 3)

        combined = None
        best_q, best_sig = -1.0, None
        freqs = None
        for i in range(k):
            sig = bandpass(self._extract(regions[:, i], self.fs), self.fs)
            freqs, spec = power_spectrum(sig, self.fs)
            q = float(spec.max()) if len(spec) else 0.0  # Schärfe des Spektrums
            combined = q**2 * spec if combined is None else combined + q**2 * spec
            if q > best_q:
                best_q, best_sig = q, sig
        if combined is None or combined.sum() <= 0:
            return None
        combined = combined / combined.sum()

        prev = float(np.median(self._history)) if (self.track and self._history) else None
        raw_bpm, snr = pick_peak(freqs, combined, n, self.fs, prev)
        if np.isnan(raw_bpm):
            return None
        if snr >= self.min_snr_db:
            self._history.append(raw_bpm)
        smoothed = float(np.median(self._history)) if self._history else float("nan")
        return PulseResult(bpm=smoothed, raw_bpm=raw_bpm, snr_db=snr, signal=best_sig, fs=self.fs)
