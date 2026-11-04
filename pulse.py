#!/usr/bin/env python3
"""Puls über die MacBook-Webcam messen (kontaktlos, rPPG).

Aufruf:
    python pulse.py                  # Live über die eingebaute Kamera
    python pulse.py --camera 1       # andere Kamera
    python pulse.py --video clip.mp4 # aufgenommenes Video auswerten
    python pulse.py --headless       # ohne Fenster, nur Ausgabe im Terminal
    python pulse.py --csv puls.csv   # Messwerte zusätzlich als CSV speichern

Tasten im Fenster: q / Esc = beenden, r = Messung neu starten.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rppg import METHODS, PulseEstimator, PulseResult

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
DETECT_WIDTH = 320  # Gesichtserkennung auf verkleinertem Bild (schneller)
FACE_LOST_RESET_SEC = 1.0  # so lange darf das Gesicht fehlen, bevor neu gemessen wird
ESTIMATE_EVERY_SEC = 0.5
LANDMARK_SMOOTHING = 0.3  # EMA-Faktor gegen ROI-Zittern (kleiner = ruhiger)

Rect = tuple[int, int, int, int]  # x, y, w, h


# --------------------------------------------------------------------------- #
# Gesicht & Hautregionen
# --------------------------------------------------------------------------- #


@dataclass
class Face:
    box: np.ndarray  # x, y, w, h
    eye_a: np.ndarray  # x, y
    eye_b: np.ndarray
    nose: np.ndarray
    mouth_a: np.ndarray
    mouth_b: np.ndarray

    def as_array(self) -> np.ndarray:
        return np.concatenate([self.box, self.eye_a, self.eye_b, self.nose, self.mouth_a, self.mouth_b])

    @classmethod
    def from_array(cls, a: np.ndarray) -> "Face":
        return cls(a[0:4], a[4:6], a[6:8], a[8:10], a[10:12], a[12:14])


class FaceTracker:
    """YuNet-Gesichtserkennung mit Glättung der Landmarken."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            sys.exit(f"Modell nicht gefunden: {model_path}")
        self._detector = cv2.FaceDetectorYN.create(str(model_path), "", (DETECT_WIDTH, DETECT_WIDTH), 0.7, 0.3, 50)
        self._smoothed: np.ndarray | None = None

    def reset(self) -> None:
        self._smoothed = None

    def detect(self, frame: np.ndarray) -> Face | None:
        h, w = frame.shape[:2]
        scale = DETECT_WIDTH / w
        small = cv2.resize(frame, (DETECT_WIDTH, int(round(h * scale))))
        self._detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._detector.detect(small)
        if faces is None or len(faces) == 0:
            return None
        # größtes Gesicht nehmen (= vermutlich die Person vor dem Laptop)
        best = max(faces, key=lambda f: f[2] * f[3])
        raw = best[:14].astype(float) / scale
        if self._smoothed is None:
            self._smoothed = raw
        else:
            self._smoothed = LANDMARK_SMOOTHING * raw + (1 - LANDMARK_SMOOTHING) * self._smoothed
        return Face.from_array(self._smoothed)


def skin_rois(face: Face, frame_shape: tuple[int, ...]) -> list[Rect]:
    """Stirn und beide Wangen, aus den Landmarken abgeleitet.

    Diese Bereiche sind gut durchblutet, meist frei von Haaren/Bart und
    bewegen sich beim Sprechen kaum.
    """
    left_eye, right_eye = sorted([face.eye_a, face.eye_b], key=lambda p: p[0])
    eye_mid = (left_eye + right_eye) / 2
    d = float(np.linalg.norm(right_eye - left_eye))  # Augenabstand als Maßstab
    if d < 8:
        return []
    mouth_y = (face.mouth_a[1] + face.mouth_b[1]) / 2
    cheek_bottom = (face.nose[1] + mouth_y) / 2

    rois = [
        # Stirn
        (eye_mid[0] - 0.65 * d, eye_mid[1] - 1.0 * d, 1.3 * d, 0.55 * d),
    ]
    for eye, outward in ((left_eye, -1), (right_eye, 1)):
        top = eye[1] + 0.35 * d
        rois.append((eye[0] - 0.28 * d + outward * 0.05 * d, top, 0.56 * d, max(cheek_bottom - top, 0.2 * d)))

    H, W = frame_shape[:2]
    out: list[Rect] = []
    for x, y, w, h in rois:
        x0, y0 = int(max(0, x)), int(max(0, y))
        x1, y1 = int(min(W, x + w)), int(min(H, y + h))
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def mean_rgb(frame: np.ndarray, rois: list[Rect]) -> np.ndarray | None:
    """Mittlere RGB-Farbe aller Pixel in den Regionen.

    Sehr dunkle und überbelichtete Pixel werden ignoriert, da sie keine
    Farbinformation tragen (Schatten, Glanzlichter).
    """
    pixels = [frame[y : y + h, x : x + w].reshape(-1, 3) for x, y, w, h in rois]
    if not pixels:
        return None
    px = np.concatenate(pixels).astype(float)
    ok = (px.max(axis=1) < 250) & (px.min(axis=1) > 15)
    if ok.sum() < 50:
        return None
    b, g, r = px[ok].mean(axis=0)  # OpenCV liefert BGR
    return np.array([r, g, b])


# --------------------------------------------------------------------------- #
# Anzeige
# --------------------------------------------------------------------------- #

GREEN, YELLOW, RED, WHITE, GREY = (80, 200, 80), (60, 200, 230), (70, 70, 230), (255, 255, 255), (150, 150, 150)


def quality_color(snr_db: float) -> tuple[int, int, int]:
    if snr_db >= 3:
        return GREEN
    if snr_db >= 0:
        return YELLOW
    return RED


def draw_overlay(
    frame: np.ndarray,
    rois: list[Rect],
    result: PulseResult | None,
    status: str,
    progress: float,
) -> None:
    for x, y, w, h in rois:
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)

    H, W = frame.shape[:2]
    panel = frame[0:110, 0:330]
    cv2.addWeighted(panel, 0.4, np.zeros_like(panel), 0.6, 0, panel)

    if result is not None and not np.isnan(result.bpm):
        color = quality_color(result.snr_db)
        cv2.putText(frame, f"{result.bpm:5.1f}", (12, 62), cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "BPM", (205, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Signal: {result.snr_db:+.1f} dB", (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, status, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1, cv2.LINE_AA)
        bar_w = int(300 * min(max(progress, 0.0), 1.0))
        cv2.rectangle(frame, (12, 65), (312, 80), GREY, 1)
        cv2.rectangle(frame, (12, 65), (12 + bar_w, 80), GREEN, -1)

    cv2.putText(frame, "q: Beenden   r: Neustart", (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)

    # Pulskurve der letzten ~5 Sekunden
    if result is not None:
        sig = result.signal[-int(5 * result.fs) :]
        if len(sig) > 2 and np.ptp(sig) > 0:
            pw, ph = min(400, W - 24), 70
            x0, y0 = W - pw - 12, H - ph - 30
            box = frame[y0 : y0 + ph, x0 : x0 + pw]
            cv2.addWeighted(box, 0.4, np.zeros_like(box), 0.6, 0, box)
            norm = (sig - sig.min()) / np.ptp(sig)
            xs = np.linspace(x0, x0 + pw - 1, len(sig))
            ys = y0 + ph - 5 - norm * (ph - 10)
            pts = np.column_stack([xs, ys]).astype(np.int32)
            cv2.polylines(frame, [pts], False, quality_color(result.snr_db), 2, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Hauptschleife
# --------------------------------------------------------------------------- #


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"Video konnte nicht geöffnet werden: {args.video}")
        return cap

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        msg = f"Kamera {args.camera} konnte nicht geöffnet werden."
        if sys.platform == "darwin":
            msg += (
                "\nPrüfe unter Systemeinstellungen → Datenschutz & Sicherheit → Kamera,"
                " ob dein Terminal (bzw. VS Code/iTerm) Zugriff hat, und starte es danach neu."
            )
        sys.exit(msg)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def parse_roi(s: str) -> Rect:
    try:
        x, y, w, h = (int(v) for v in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("Format: x,y,breite,höhe (z. B. 600,200,120,60)")
    return x, y, w, h


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Puls über die Webcam messen (rPPG, POS-Algorithmus).")
    p.add_argument("--camera", type=int, default=0, help="Kamera-Index (Standard: 0 = eingebaute FaceTime-Kamera)")
    p.add_argument("--video", help="Videodatei statt Kamera auswerten")
    p.add_argument("--window", type=float, default=10.0, help="Analysefenster in Sekunden (Standard: 10)")
    p.add_argument("--method", choices=sorted(METHODS), default="pos", help="Signalverfahren (Standard: pos)")
    p.add_argument("--roi", type=parse_roi, help="Feste Hautregion x,y,w,h statt Gesichtserkennung")
    p.add_argument("--headless", action="store_true", help="Kein Fenster, nur Textausgabe")
    p.add_argument("--csv", help="Messwerte in diese CSV-Datei schreiben")
    p.add_argument("--no-mirror", action="store_true", help="Live-Bild nicht spiegeln")
    args = p.parse_args(argv)

    cap = open_capture(args)
    tracker = None if args.roi else FaceTracker()
    estimator = PulseEstimator(window_sec=args.window, min_sec=min(8.0, args.window), method=args.method)
    mirror = not args.video and not args.no_mirror
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    csv_file = open(args.csv, "w", newline="") if args.csv else None
    writer = csv.writer(csv_file) if csv_file else None
    if writer:
        writer.writerow(["zeit_s", "bpm", "bpm_roh", "snr_db"])

    t_start = time.monotonic()
    last_face_t: float | None = None
    last_estimate_t = -1e9
    last_print_t = -1e9
    result: PulseResult | None = None
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.video:
                t = frame_idx / video_fps
            else:
                t = time.monotonic() - t_start
            if mirror:
                frame = cv2.flip(frame, 1)

            # 1) Hautregionen bestimmen
            if args.roi:
                rois = [args.roi]
            else:
                face = tracker.detect(frame)
                rois = skin_rois(face, frame.shape) if face else []

            # 2) Farbe messen
            rgb = mean_rgb(frame, rois) if rois else None
            if rgb is not None:
                estimator.push(t, rgb)
                last_face_t = t
                status = "Messe Puls, bitte stillhalten ..."
            elif last_face_t is None or t - last_face_t > FACE_LOST_RESET_SEC:
                if last_face_t is not None:
                    estimator.reset()
                    if tracker:
                        tracker.reset()
                    last_face_t = None
                result = None
                status = "Kein Gesicht erkannt"
            else:
                status = "Gesicht kurz verloren ..."

            # 3) Puls schätzen
            if t - last_estimate_t >= ESTIMATE_EVERY_SEC:
                last_estimate_t = t
                new = estimator.estimate()
                if new is not None:
                    result = new
                    if writer:
                        writer.writerow([f"{t:.2f}", f"{new.bpm:.1f}", f"{new.raw_bpm:.1f}", f"{new.snr_db:.2f}"])

            if args.headless and result is not None and t - last_print_t >= 1.0:
                last_print_t = t
                print(f"t={t:6.1f}s  Puls {result.bpm:5.1f} BPM  (Fenster: {result.raw_bpm:5.1f}, SNR {result.snr_db:+5.1f} dB)", flush=True)

            # 4) Anzeigen
            if not args.headless:
                progress = estimator.buffered_sec / estimator.min_sec
                draw_overlay(frame, rois, result, status, progress)
                cv2.imshow("Puls-Messung", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    estimator.reset()
                    result = None
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()

    if result is not None and not np.isnan(result.bpm):
        print(f"Letzter Messwert: {result.bpm:.1f} BPM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
