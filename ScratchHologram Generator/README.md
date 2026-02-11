# ScratchHologram Generator

Pipeline Python per generare traiettorie di scratch hologram su piano 2D a partire da un file STL.

## Cosa fa

1. Carica una mesh STL triangolare.
2. Campiona punti sulla superficie della mesh.
3. Calcola una intensita speculare semplificata (modello geometrico tipo Phong).
4. Proietta i punti sul piano XY.
5. Genera segmenti di incisione orientati secondo la riflessione speculare.
6. Esporta in SVG.
7. Opzionale: esporta G-code e JSON diagnostico.

## File inclusi

- `scratch_pipeline.py`: script principale.
- `requirements.txt`: dipendenze Python.

## Requisiti

- Python 3.10+ (testato con Python 3.14)

## Setup rapido (Windows PowerShell)

```powershell
cd "c:\Users\filip\Documents\thesis\ScratchHologram Generator"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Primo test consigliato (cubo semplice)

Usa lo STL gia presente nel workspace:

```powershell
python .\scratch_pipeline.py `
  --stl "..\knots\basic_cube_-_10mm.stl" `
  --svg ".\out\cube_scratch.svg" `
  --gcode ".\out\cube_scratch.gcode" `
  --json ".\out\cube_scratch_debug.json" `
  --samples 5000 `
  --width-mm 80 --height-mm 80 `
  --shininess 30 `
  --spec-threshold 0.08 `
  --seed 42
```

Output attesi:

- `out/cube_scratch.svg`
- `out/cube_scratch.gcode` (se richiesto)
- `out/cube_scratch_debug.json` (se richiesto)

## Parametri principali

- `--samples`: numero di punti campionati sulla superficie.
- `--shininess`: selettivita speculare (piu alto = highlight piu stretti).
- `--spec-threshold`: soglia minima per tenere uno scratch.
- `--min-len-mm`, `--max-len-mm`: lunghezze minima/massima dei segmenti.
- `--light LX LY LZ`: direzione luce globale.
- `--view VX VY VZ`: direzione osservatore globale.
- `--max-segments`: limite superiore segmenti esportati.

## Esempi utili

Solo SVG:

```powershell
python .\scratch_pipeline.py --stl "..\knots\trefoil.stl" --svg ".\out\trefoil.svg"
```

Direzione luce diversa:

```powershell
python .\scratch_pipeline.py `
  --stl "..\knots\TrefoilKnot.stl" `
  --svg ".\out\trefoil_luce_laterale.svg" `
  --light 1.0 0.0 0.5 `
  --view 0.0 0.0 1.0
```

## Note sul modello fisico

Il modello implementato e volutamente semplificato:

- non simula ottica ondulatoria;
- usa riflessione speculare geometrica locale su normali di faccia;
- e adatto come base riproducibile per una tesi triennale e per iterazioni successive.

## Troubleshooting

Se vedi errore su pacchetti mancanti:

```powershell
pip install -r requirements.txt
```

Se ottieni pochi segmenti:

- abbassa `--spec-threshold` (es. `0.05`)
- riduci `--shininess` (es. `20`)
- aumenta `--samples` (es. `15000`)
