# ScratchHologram Generator

Generatore Python di archi tipo HoloZens a partire da file STL.

Questa versione produce **solo archi** (niente linee/profili), con una pipeline edge-based:

1. Carica STL.
2. Estrae vertici e spigoli unici.
3. Applica camera/proiezione in stile HoloZens.
4. Campiona punti lungo ogni spigolo (line resolution).
5. Converte ogni punto in arco 180°.
6. Esporta SVG (opzionale JSON debug).

## File inclusi

- `scratch_pipeline.py`: script principale arc-based.
- `requirements.txt`: dipendenze Python.

## Requisiti

- Python 3.10+ (testato con Python 3.14).

## Setup (PowerShell)

```powershell
cd "c:\Users\filip\Documents\thesis\ScratchHologram Generator"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Test richiesto (basic cube)

```powershell
.\.venv\Scripts\python.exe .\scratch_pipeline.py `
  --stl "..\knots\basic_cube_-_10mm.stl" `
  --svg ".\out\basic_cube_holozens_arcs.svg" `
  --simulate-html ".\out\basic_cube_simulation.html" `
  --json ".\out\basic_cube_holozens_arcs.json" `
  --line-resolution 3.28 `
  --min-arc-radius 6
```

## Parametri principali

- `--line-resolution`: punti per unità di lunghezza spigolo.
- `--min-arc-radius`: filtra gli archi troppo piccoli (utile per pulire il risultato).
- `--canvas-width`, `--canvas-height`: dimensione canvas camera.
- `--po`, `--pr`, `--look-up`, `--zf`, `--current-scale`: parametri camera.
- `--stroke-width`: spessore arco nello SVG (solo visualizzazione, non incisione fisica).
- `--no-auto-center`: disabilita l'allineamento Z automatico del modello.

## Output

- SVG con soli path ad arco (`M ... A ...`).
- HTML interattivo opzionale con slider `View angle` per simulare il movimento percepito.
- JSON opzionale con:
  - configurazione camera/pipeline;
  - numero archi/spigoli;
  - parametri geometrici di ogni arco.

## Note

- Obiettivo: comportamento vicino a HoloZens lato geometria archi.
- Non include ancora GUI, linee/profili o export G-code.
