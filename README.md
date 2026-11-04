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

Nach etwa 8 Sekunden erscheint der erste Wert. Die Farbe zeigt die Signalqualität:
**grün** = gut, **gelb** = brauchbar, **rot** = unzuverlässig.

| Option | Bedeutung |
|---|---|
| `--camera N` | andere Kamera (z. B. Continuity Camera / iPhone) |
| `--video datei.mp4` | aufgenommenes Video statt Live-Kamera auswerten |
| `--window 10` | Analysefenster in Sekunden (länger = ruhiger, aber träger) |
| `--method pos` / `green` | Signalverfahren, siehe unten |
| `--roi x,y,w,h` | feste Hautregion statt Gesichtserkennung |
| `--headless` | kein Fenster, Werte im Terminal |
| `--csv puls.csv` | Messwerte mit Uhrzeit mitschreiben |
| `--record roh.csv` | Rohdaten pro Bild mit Uhrzeit speichern (zum Nachanalysieren) |
| `--no-track` | jedes Analysefenster unabhängig schätzen (kein Peak-Tracking) |

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
4. `roh.csv`, `puls.csv` und die Bildschirmaufnahme zur Auswertung weitergeben.
   Mit den Rohdaten lassen sich Verfahren und Einstellungen nachträglich gegen
   die Referenz abgleichen, ohne neu zu messen.

## So funktioniert es

1. **Gesicht finden:** [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
   (in OpenCV eingebaut, Modell liegt in `models/`) liefert Gesichtsbox und
   Landmarken (Augen, Nase, Mundwinkel). Die Landmarken werden geglättet, damit
   die Messregion nicht zittert.
2. **Hautregionen:** Stirn (oberhalb der Augenbrauen) und beide Wangen, aus dem
   Augenabstand abgeleitet. Eine adaptive Hautmaske (Median-Hautfarbe in YCrCb)
   wirft Haare, Augenbrauen, Brillengestelle, Schatten und Glanzlichter raus.
   Gemessen wird die mittlere Farbe pro Region und über alle Regionen.
3. **Gleichmäßig abtasten:** Webcams liefern keine exakt konstante Bildrate,
   deshalb wird auf 30 Hz interpoliert.
4. **Pulssignal extrahieren** (`rppg.py`):
   - `pos` (Standard): *Plane-Orthogonal-to-Skin* nach Wang et al. 2017.
     Projiziert die Farbschwankungen so, dass reine Helligkeitsänderungen
     (Bewegung, Lichtflackern) sich weitgehend aufheben. Variante hier: Die
     Projektionen werden vor der Gewichtung bandpassgefiltert, weil echte
     Kameras nicht exakt linear in allen Kanälen reagieren.
   - `green`: einfaches Grünkanal-Signal. Bei sehr ruhigem, gleichmäßigem Licht
     manchmal sauberer, bei Flackern oder Bewegung deutlich schlechter.
5. **Frequenz bestimmen:** Bandpass 42–210 BPM, FFT über die letzten 10 s pro
   Region. Die Spektren werden nach ihrer Schärfe gewichtet kombiniert, damit
   eine gestörte Region nicht alles verdirbt. Gibt es mehrere ähnlich starke
   Peaks, gewinnt der, der am nächsten am bisherigen Wert liegt (Tracking).
   Angezeigt wird der Median der letzten Schätzungen mit ausreichender
   Signalqualität.

## Tests

```bash
pip install pytest
python -m pytest
```

Die Tests prüfen die Signalverarbeitung mit synthetischen Hautfarb-Signalen
(Puls + Helligkeitsdrift + Lichtflackern + Rauschen + schwankende Bildrate).

## Grenzen

- **Kein Medizinprodukt.** Die Werte sind eine Schätzung, nicht für Diagnosen geeignet.
- Funktioniert nur bei stillhaltender Person und ausreichend Licht.
- Bei sehr dunkler Haut ist das Signal schwächer, dann helfen mehr Licht und ein längeres `--window`.
- Bisher nur mit synthetischen Daten validiert. Die Abstimmung auf echte Aufnahmen
  steht noch aus, siehe Vergleichsmessung oben.

## Lizenz

Das YuNet-Modell (`models/face_detection_yunet_2023mar.onnx`) steht unter der
MIT-Lizenz, siehe `models/LICENSE-yunet`.
