# ScratchHologram Generator

Generatore Python di archi tipo HoloZens/HoloCraft a partire da file STL.

Il progetto include:
- pipeline CLI per `STL -> SVG` (opzionale HTML/JSON),
- app desktop lanciabile con preview archi single-view + export SVG/G-code.

## File principali

- `scratch_pipeline.py`: pipeline e rendering/simulazione.
- `scratch_desktop_app.py`: applicazione desktop interattiva.
- `requirements.txt`: dipendenze Python.

## Requisiti

- Python 3.10+ (testato con Python 3.14).
- `tkinter` (incluso di default su Windows Python standard).

## Setup (PowerShell)

```powershell
cd "c:\Users\filip\Documents\thesis\ScratchHologram Generator"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Avvio App Desktop

```powershell
.\.venv\Scripts\python.exe .\scratch_desktop_app.py
```

## Validazione CNC (punto 1)

Per generare un pacchetto di test ripetibile (SVG + G-code + report):

```powershell
.\.venv\Scripts\python.exe .\cnc_validation_suite.py
```

Output in:
- `.\out\cnc_validation\`
  - file `.svg` e `.nc` per ogni modello in modalita` `semi` e `elliptic`,
  - `validation_report.md` (checklist CAM + metriche),
  - `validation_report.json` (dati strutturati).

Note:
- per mantenere tempi ragionevoli, il suite applica un cap con `--max-arcs-per-case`
  (downsampling uniforme solo a scopo validazione; il valore compare come `DS` nel report).
- nel report, `G1-XY` indica i movimenti lineari in XY:
  in modalita` `semi` deve essere molto inferiore a `G2/G3`.

Puoi passare STL specifici:

```powershell
.\.venv\Scripts\python.exe .\cnc_validation_suite.py `
  --models "..\knots\basic_cube_-_10mm.stl" "..\knots\trefoil.stl"
```

Nell'app puoi:
1. caricare STL (`Apri STL`);
2. ruotare la camera direttamente sulla preview archi (drag) e zoomare con rotellina;
3. regolare i parametri (line resolution, min arc radius, preview quality, view angle, arc mode, ellipse ratio, cull, ecc.);
4. esportare lo SVG della vista corrente (`Esporta SVG`);
5. esportare il G-code (`Esporta G-code`) con dialog parametri:
   - larghezza finale in mm,
   - quota Z sicurezza/incisione,
   - feed XY / Z,
   - segmentazione massima,
   - RPM mandrino opzionale (0 disabilita `M3/M5`),
   - inversione asse Y opzionale.
6. opzionale `Advanced cull (preview + export)` per rimuovere archi occlusi sia in preview sia in export.
   - usa `Cull strength` per regolare quanto e` aggressivo il filtro (`20-50%` consigliato su modelli complessi);
   - `Advanced cull samples` aumenta precisione del filtro (piu` lento).
7. scegliere la geometria degli archi:
   - `Semicircle (CNC)`: semicerchi perfetti, piu` adatti a strategie G2/G3.
   - `Elliptic`: archi schiacciati in verticale; usa `Ellipse ratio` (0.20..1.00).

Nota coerenza preview/export:
- l'export usa lo stesso dataset geometrico della preview come base;
- se `Advanced cull (preview + export)` e` attivo, preview ed export applicano lo stesso filtro di visibilita`.

Note preview:
- `Preview quality` influenza davvero dettaglio e velocita` (campionamento edge + line resolution effettiva).
- durante il drag la preview passa in modalita` `FAST` per ridurre il lag.
- `View angle` + `Show simulated profile` simulano il movimento osservato nello scratch hologram.

## Uso CLI (pipeline)

```powershell
.\.venv\Scripts\python.exe .\scratch_pipeline.py `
  --stl "..\knots\basic_cube_-_10mm.stl" `
  --svg ".\out\basic_cube_holozens_arcs.svg" `
  --simulate-html ".\out\basic_cube_simulation_cam.html" `
  --json ".\out\basic_cube_holozens_arcs.json" `
  --line-resolution 3.28 `
  --min-arc-radius 6 `
  --stroke-width 0.15
```

## Parametri chiave

- `line-resolution`: densita` di campionamento lungo gli spigoli.
- `min-arc-radius`: filtro anti micro-archi.
- `stroke-width`: spessore visuale arco in SVG.
- `arc-mode`: `semi` oppure `elliptic`.
- `ellipse-ratio`: rapporto altezza/larghezza dell'arco quando `arc-mode=elliptic`.
- camera (`po`, `pr`, `look-up`, `zf`, `current-scale`) per la proiezione.

## Output

- SVG con soli path ad arco (`M ... A ...`).
- G-code (`.nc/.gcode`) con:
  - `G21`, `G90`, `G17`, `G94`,
  - movimenti rapidi `G0`,
  - incisione:
    - `Semicircle (CNC)`: archi con `G2/G3`,
    - `Elliptic`: segmentazione lineare con `G1`,
  - opzionale `M3/M5`.
- HTML opzionale con simulazione interattiva e controlli luce/profondita`.
- JSON opzionale con dati di debug/statistiche.
