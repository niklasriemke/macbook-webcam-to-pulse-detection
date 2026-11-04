#!/usr/bin/env python3
"""Aufgezeichnete Rohdaten (pulse.py --record) offline auswerten.

Spielt die Hautfarben einer Aufnahme mit beliebigen Einstellungen durch die
Pulsschätzung und vergleicht das Ergebnis optional mit einem Referenzgerät.

    python auswertung.py messungen/roh.csv
    python auswertung.py messungen/roh.csv --referenz messungen/whoop_referenz.csv
    python auswertung.py messungen/roh.csv --referenz ... --method pos --window 10

Referenzdatei: CSV mit Spalten `uhrzeit` (HH:MM:SS nach der Uhr des
Referenzgeräts) und `bpm`; jeder Wert gilt bis zum nächsten Eintrag.
`--versatz` gibt an, wie viele Sekunden die Uhr des Referenzgeräts vorgeht.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from rppg import METHODS, PulseEstimator

REGIONS = ("alle", "stirn", "wange_l", "wange_r")


@dataclass
class Recording:
    unix: np.ndarray  # (N,) Unix-Zeit je Bild
    t: np.ndarray  # (N,) Sekunden seit Start
    colors: np.ndarray  # (N, k, 3) RGB je Region, NaN wenn nicht gemessen
    clock0: str  # Uhrzeit des ersten Bilds (HH:MM:SS.mmm)


def load_recording(path: str) -> Recording:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    regions = [r for r in REGIONS if f"{r}_g" in rows[0]]
    val = lambda r, k: float(r[k]) if r.get(k) else np.nan  # noqa: E731
    colors = np.array([[[val(r, f"{reg}_{c}") for c in "rgb"] for reg in regions] for r in rows])
    return Recording(
        unix=np.array([val(r, "unix_s") for r in rows]),
        t=np.array([val(r, "zeit_s") for r in rows]),
        colors=colors,
        clock0=rows[0]["uhrzeit"],
    )


def load_reference(path: str, rec: Recording, offset_sec: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Referenzwerte als (Unix-Zeit nach Mac-Uhr, bpm)."""

    def secs(clock: str) -> float:
        d = datetime.strptime(clock, "%H:%M:%S.%f" if "." in clock else "%H:%M:%S")
        return d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6

    base = rec.unix[0] - secs(rec.clock0)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    times = np.array([base + secs(r["uhrzeit"]) - offset_sec for r in rows])
    return times, np.array([float(r["bpm"]) for r in rows])


def replay(rec: Recording, method: str = "green", window: float = 20.0, track: bool = False, every: float = 0.5):
    """Liefert [(unix, angezeigter bpm, roher bpm, snr)] wie live im Programm."""
    est = PulseEstimator(window_sec=window, min_sec=min(10.0, window), method=method, track=track)
    out = []
    last = -np.inf
    for i in range(len(rec.t)):
        c = rec.colors[i]
        if np.isnan(c).any():
            continue
        est.push(rec.t[i], c)
        if rec.t[i] - last >= every:
            last = rec.t[i]
            r = est.estimate()
            if r is not None and np.isfinite(r.bpm):
                out.append((rec.unix[i], r.bpm, r.raw_bpm, r.snr_db))
    return out


def compare(estimates, ref_times: np.ndarray, ref_bpm: np.ndarray) -> dict:
    """Mittlerer absoluter Fehler und Trefferquote (< 5 BPM) gegen die Referenz."""
    errs = []
    for unix, bpm, *_ in estimates:
        k = np.searchsorted(ref_times, unix, side="right") - 1
        if 0 <= k < len(ref_times) - 1 or (k == len(ref_times) - 1 and unix - ref_times[k] < 5):
            errs.append(abs(bpm - ref_bpm[k]))
    e = np.array(errs)
    if len(e) == 0:
        return {"n": 0, "mae": float("nan"), "treffer_5": float("nan")}
    return {"n": len(e), "mae": float(e.mean()), "treffer_5": float((e < 5).mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("aufnahme")
    p.add_argument("--referenz")
    p.add_argument("--versatz", type=float, default=0.0, help="Sekunden, die die Referenzuhr vorgeht")
    p.add_argument("--method", choices=sorted(METHODS), default="green")
    p.add_argument("--window", type=float, default=20.0)
    p.add_argument("--track", action="store_true")
    args = p.parse_args()

    rec = load_recording(args.aufnahme)
    est = replay(rec, args.method, args.window, args.track)
    ref = load_reference(args.referenz, rec, args.versatz) if args.referenz else None
    print(f"{'Uhrzeit':>12}  {'Skript':>6}  {'roh':>6}  {'SNR':>6}" + ("  Referenz" if ref else ""))
    for unix, bpm, raw, snr in est[::2]:
        line = f"{datetime.fromtimestamp(unix).strftime('%H:%M:%S'):>12}  {bpm:6.1f}  {raw:6.1f}  {snr:+6.1f}"
        if ref:
            k = np.searchsorted(ref[0], unix, side="right") - 1
            line += f"  {ref[1][k]:6.0f}" if 0 <= k < len(ref[0]) else ""
        print(line)
    if ref:
        res = compare(est, *ref)
        print(
            f"\nVergleich mit Referenz ({res['n']} Werte): mittlerer Fehler {res['mae']:.1f} BPM, "
            f"{res['treffer_5'] * 100:.0f} % innerhalb von 5 BPM"
        )


if __name__ == "__main__":
    main()
