# ScratchHologram Generator

Generatore Python di archi tipo HoloZens/HoloCraft a partire da file STL.

Il progetto include:
- pipeline CLI per `STL -> SVG` (opzionale HTML/JSON),
- app desktop lanciabile con preview archi single-view + export SVG.

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

Nell'app puoi:
1. caricare STL (`Apri STL`);
2. ruotare la camera direttamente sulla preview archi (drag) e zoomare con rotellina;
3. regolare i parametri (line resolution, min arc radius, quality, scale, zf, view angle/view gain, ecc.);
4. esportare lo SVG della vista corrente (`Esporta SVG`).

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
- camera (`po`, `pr`, `look-up`, `zf`, `current-scale`) per la proiezione.

## Output

- SVG con soli path ad arco (`M ... A ...`).
- HTML opzionale con simulazione interattiva e controlli luce/profondita`.
- JSON opzionale con dati di debug/statistiche.
