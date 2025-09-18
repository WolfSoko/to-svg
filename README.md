# color_vectorize

Tool zur Farb-Vektorisierung (PNG/JPG → SVG) mit:
- Farbreduktion (KMeans)
- Glättung (Chaikin / optional Bézier)
- Polygonvereinfachung (Douglas-Peucker)
- Löcher (RETR_CCOMP + evenodd)
- Optionale Strokes je Fläche
- Überlappung gegen Spalten
- Alpha-Verarbeitung (flatten / binary)

## Installation
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Minimal
```bash
python color_vectorize.py eprivacy_logo.png out.svg
```

## Wichtige Parameter
- --colors N               Anzahl Farben (Cluster)
- --bg #RRGGBB             Hintergrund
- --min-area PX            Mindestfläche äußere Konturen
- --min-hole-area PX       Mindestfläche Löcher
- --smooth N               Chaikin-Iterationen (0–3)
- --epsilon F              Vereinfachung (0 = aus)
- --bezier                 Polylinien → kubische Segmente
- --outline                Strokes aktivieren
- --outline-color C|auto   Stroke-Farbe oder auto (abgedunkelt)
- --outline-width F        Strichbreite
- --overlap PX             Dilation der Maske (Überdeckung gegen Spalten)
- --precision N            Dezimalstellen Koordinaten
- --order area-desc|...    Zeichenreihenfolge
- --alpha-mode ignore|flatten|binary
- --alpha-threshold T      Schwellwert für Alpha

## Typische Aufrufe
Saubere Flächen, leichte Glättung, Überdeckung:
```bash
python color_vectorize.py eprivacy_logo.png out.svg --colors 8 --bg "#ffffff" --smooth 2 --epsilon 1.0 --bezier --overlap 1 --precision 4
```
Mit Strokes:
```bash
python color_vectorize.py eprivacy_logo.png out.svg --outline --outline-width 2 --overlap 1
```
Alpha weich einbetten (Halos weg):
```bash
python color_vectorize.py logo.png out.svg --alpha-mode flatten --alpha-threshold 12 --overlap 1
```
Harte Kanten aus RGBA (transparente Pixel → Hintergrund):
```bash
python color_vectorize.py logo.png out.svg --alpha-mode binary --alpha-threshold 0
```
Superkontur hinzufügen:
```bash
python color_vectorize.py eprivacy_logo.png out.svg --supercontour eprivacy_logo.svg --contour-width 2
```

## Gegen sichtbare Spalten
1. --overlap 1 (ggf. 2)  
2. --precision 4 oder 5  
3. Reihenfolge: große Flächen zuerst (Default area-desc)  
4. Alpha flatten nutzen bevor quantisiert wird.

## Feinere Konturen
- Weniger Zacken: --smooth 1..2 + moderates --epsilon (0.8–1.5)
- Möglichst original: --smooth 0 --epsilon 0
- Bézier weicher, aber mehr Punkte: --bezier

## Performance Hinweise
- Große Bilder vorher skalieren
- Weniger Farben = schneller
- Hohe Overlap + viele Farben → mehr Pfade

## Nachbearbeitung
Optional Optimierung mit svgo / scour:
```bash
npx svgo out.svg -o out.min.svg
```

## Fehlerbehebung
- NameError / ModuleNotFound: Abhängigkeiten installieren
- Leeres SVG: Bildpfad prüfen
- Augen fehlen: --min-hole-area senken (z.B. 2)
- Farb-Ausfransungen: Alpha flatten + overlap

## Lizenz
Eigener Gebrauch / Wolfram Sokollek fragen

