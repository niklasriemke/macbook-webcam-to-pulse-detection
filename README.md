# MacBook-Webcam → Puls

Misst die Herzfrequenz kontaktlos über die eingebaute Kamera eines MacBooks.

Bei jedem Herzschlag ändert sich die Durchblutung der Haut, und damit ihre Farbe
um Bruchteile eines Prozents. Mit bloßem Auge ist das unsichtbar, eine Kamera
kann es aber messen. Das Verfahren heißt **remote Photoplethysmographie (rPPG)**.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Benutzung

```bash
python pulse.py
```

Beim ersten Start fragt macOS nach dem Kamerazugriff für dein Terminal. Falls
die Kamera nicht aufgeht: **Systemeinstellungen → Datenschutz & Sicherheit →
Kamera**, dort Terminal / iTerm / VS Code erlauben und die App neu starten.

Nach etwa 10 Sekunden erscheint der erste Wert. Die Farbe zeigt die Signalqualität:
**grün** = gut, **gelb** = brauchbar, **rot** = unzuverlässig (fließt nicht in die Anzeige ein).

| Option | Bedeutung |
|---|---|
| `--camera N` | andere Kamera (z. B. Continuity Camera / iPhone) |
| `--video datei.mp4` | aufgenommenes Video statt Live-Kamera auswerten |
| `--window 20` | Analysefenster in Sekunden (länger = ruhiger, aber träger) |
| `--method green` / `pos` | Signalverfahren, siehe unten |
| `--roi x,y,w,h` | feste Hautregion statt Gesichtserkennung |
| `--headless` | kein Fenster, Werte im Terminal |
| `--csv puls.csv` | Messwerte mit Uhrzeit mitschreiben |
| `--record roh.csv` | Rohdaten pro Bild mit Uhrzeit speichern (zum Nachanalysieren) |
| `--track` | Peak-Tracking: bei ähnlich starken Peaks den nächstgelegenen nehmen |

Tasten: `q` / `Esc` beenden, `r` Messung neu starten.

## Tipps für gute Messwerte

- **Stillhalten.** Kopfbewegungen und Sprechen sind die größte Störquelle.
- **Gleichmäßiges, helles Licht von vorn**, am besten Tageslicht. Kein Gegenlicht.
- Stirn frei (Haare/Pony verdecken die beste Messregion).
- Bildschirmhelligkeit konstant lassen, wechselnde Bildschirminhalte strahlen aufs Gesicht.
- **macOS-Videoeffekte aus:** Kontrollzentrum → Videoeffekte → Studiolicht und Porträt
  deaktivieren. Sie verändern Helligkeit und Farbe des Gesichts laufend.
- Zur Kontrolle: Puls am Handgelenk 30 s zählen × 2, oder eine Smartwatch daneben.

## Vergleichsmessung mit Referenzgerät (z. B. Whoop)

Beide CSV-Dateien enthalten die Uhrzeit auf die Millisekunde. Mac und iPhone
stellen ihre Uhren übers Netz, die Zeitstempel passen also zur Bildschirmaufnahme.

1. Auf dem iPhone die Live-Herzfrequenz der Whoop öffnen und eine
   Bildschirmaufnahme starten. Die Statusleiste mit der Uhr muss sichtbar sein.
2. Auf dem Mac starten:
   ```bash
   python pulse.py --record roh.csv --csv puls.csv
   ```
3. Zwei Minuten ruhig sitzen, dann `q` drücken und die Bildschirmaufnahme beenden.
4. Die Referenzwerte als CSV ablegen (Spalten `uhrzeit,bpm`, Uhrzeit des
   Referenzgeräts, siehe `messungen/whoop_referenz.csv`) und auswerten:
   ```bash
   python auswertung.py roh.csv --referenz referenz.csv --versatz 3
   ```
   `--versatz` = Sekunden, die die Uhr des Referenzgeräts vorgeht. Mit
   `--method` und `--window` lassen sich andere Einstellungen auf derselben
   Aufnahme testen, ohne neu zu messen.

### Ergebnis der ersten Vergleichsmessung

MacBook Air, abends bei Raumlicht, ruhig sitzend, Whoop als Referenz (Puls 70–81),
etwa 45 s Überlappung:

| Einstellung | mittlerer Fehler | innerhalb ±5 BPM |
|---|---|---|
| vorher: POS, 10 s, Tracking | 26 BPM | 7 % |
| jetzt: Grünkanal, 20 s, Sprungerkennung | 3–4 BPM | 84–89 % |

Das ist eine einzelne, kurze Messung. Mehr Messungen bei anderem Licht und
anderen Personen wären nötig, um die Einstellungen belastbar zu machen. Die
Aufnahme ist als Regressionstest in `tests/test_real_recording.py` hinterlegt.

## So funktioniert es

1. **Gesicht finden:** [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
   (in OpenCV eingebaut, Modell liegt in `models/`) liefert Gesichtsbox und
   Landmarken (Augen, Nase, Mundwinkel). Die Landmarken werden geglättet, damit
   die Messregion nicht zittert.
2. **Hautregionen:** Stirn (oberhalb der Augenbrauen) und beide Wangen, aus dem
   Augenabstand abgeleitet. Eine adaptive Hautmaske pro Region (Median-Hautfarbe
   in YCrCb) wirft Haare, Augenbrauen, Brillengestelle, Schatten und Glanzlichter
   raus. Fällt eine Region kurz aus, wird ihr letzter Wert bis zu 1 s gehalten.
3. **Gleichmäßig abtasten:** Webcams liefern keine exakt konstante Bildrate,
   deshalb wird auf 30 Hz interpoliert.
4. **Sprünge erkennen:** Ändert sich die Hautfarbe von einem Bild zum nächsten um
   mehr als 3 % (Belichtungsautomatik, Kopfbewegung) oder fehlen über 1 s Daten,
   beginnt die Analyse neu. Der zuletzt angezeigte Wert bleibt so lange stehen.
5. **Pulssignal extrahieren** (`rppg.py`):
   - `green` (Standard): Grünkanal, in dem Hämoglobin am stärksten absorbiert.
     In der Vergleichsmessung klar am genauesten, weil Rot und Blau bei wenig
     Licht stark rauschen.
   - `pos`: *Plane-Orthogonal-to-Skin* nach Wang et al. 2017. Hebt reine
     Helligkeitsänderungen (Bewegung, Lichtflackern) weitgehend auf, mischt dafür
     das Rauschen aller drei Kanäle hinein. Sinnvoll bei viel Licht oder
     flackernden Lampen.
6. **Frequenz bestimmen:** Bandpass 42–210 BPM, FFT über die letzten 20 s pro
   Region. Die Spektren werden nach ihrer Schärfe gewichtet kombiniert.
   Angezeigt wird der Median der letzten Schätzungen mit ausreichender
   Signalqualität.

## Tests

```bash
pip install pytest
python -m pytest
```

Die Tests prüfen die Signalverarbeitung mit synthetischen Hautfarb-Signalen
(Puls + Helligkeitsdrift + Lichtflackern + Rauschen + schwankende Bildrate) und
mit der echten Whoop-Vergleichsmessung aus `messungen/`.

## Grenzen

- **Kein Medizinprodukt.** Die Werte sind eine Schätzung, nicht für Diagnosen geeignet.
- Funktioniert nur bei stillhaltender Person und ausreichend Licht.
- Bei sehr dunkler Haut ist das Signal schwächer, dann helfen mehr Licht und ein längeres `--window`.
- Bisher erst mit einer echten Vergleichsmessung abgestimmt, siehe oben.

## Lizenz

Das YuNet-Modell (`models/face_detection_yunet_2023mar.onnx`) steht unter der
MIT-Lizenz, siehe `models/LICENSE-yunet`.
