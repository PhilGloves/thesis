### ABSTRACT
Gli scratch hologram rappresentano una tecnica di olografia analogica che consente di ottenere l’illusione di profondità tridimensionale attraverso l’incisione di micro-traiettorie su superfici riflettenti. La progettazione di tali strutture richiede la trasformazione di informazioni geometriche tridimensionali in percorsi di incisione compatibili con sistemi di fabbricazione digitale.

Questa tesi presenta una pipeline software per la generazione di scratch hologram a partire da modelli tridimensionali in formato STL, finalizzata alla generazione di traiettorie di incisione su superfici riflettenti. L’approccio adottato, implementato in Python, integra il caricamento della mesh, l’estrazione e il campionamento degli spigoli, la proiezione sul piano 2D e la generazione di archi secondo un modello geometrico semplificato di riflessione speculare.

Il sistema produce output in formato SVG e G-code, in funzione della geometria delle traiettorie generate. È stata inoltre sviluppata un’applicazione desktop per la preview interattiva e la regolazione dei parametri principali, garantendo coerenza tra simulazione ed esportazione.

Il contributo del lavoro consiste nello sviluppo di una soluzione riproducibile e indipendente da strumenti proprietari, concepita come base estendibile per l’ottimizzazione dei percorsi CNC, la validazione sperimentale e future applicazioni nella fabbricazione di scratch hologram.




# TODO

### Cos’è uno scratch / specular hologram
Uno scratch hologram è una tecnica di rappresentazione tridimensionale analogica che utilizza micro-incisioni su una superficie riflettente per simulare la presenza di punti nello spazio. A differenza dell’olografia classica basata su interferenza laser, gli scratch hologram sfruttano la riflessione speculare della luce.
Ogni graffio inciso sulla superficie agisce come un micro-specchio. Se orientato correttamente, esso riflette la luce verso l’osservatore in modo tale da simulare la provenienza della luce da un punto virtuale nello spazio tridimensionale. L’insieme coordinato di molti graffi consente quindi di ricostruire una forma tridimensionale percepita.
L’effetto risultante dipende fortemente da orientazione locale del graffio, curvatura del graffio, posizione della sorgente luminosa e posizione dell’osservatore.
Questo rende gli scratch hologram un interessante punto di incontro tra ottica geometrica, computer graphics e fabbricazione digitale.
________________________________________
Perché è interessante (informatica / computer graphics)
L’interesse informatico degli scratch hologram deriva dal fatto che il problema può essere formulato come una trasformazione geometrica computazionale.
In particolare, il problema può essere visto come segue.
Dato:
un modello 3D (mesh STL o altra rappresentazione)
una configurazione luce-osservatore
Calcolare:
l’orientazione locale della superficie riflettente necessaria a generare il contributo luminoso corretto.
Questo porta a problemi tipici della computer graphics e del CAD/CAM come il calcolo delle normali di superficie, trasformazioni geometriche, generazione di traiettorie utensile, simulazione ottica semplificata e conversione da modello geometrico a istruzioni macchina (G-code).
Inoltre, l’automatizzazione del processo consente di passare da tecniche artistiche manuali a pipeline digitali riproducibili.
________________________________________
Perché è difficile automatizzarlo
L’automatizzazione della generazione di scratch hologram presenta diverse difficoltà tecniche.
Prima difficoltà: dipendenza da molte variabili fisiche. Il risultato visivo dipende simultaneamente da posizione della luce, posizione dell’osservatore, materiale utilizzato, larghezza dell’incisione e risoluzione della macchina di incisione.
Seconda difficoltà: problema inverso geometrico. Bisogna risolvere il problema inverso: non “che riflessione produce questa superficie”, ma “che superficie serve per ottenere questa riflessione”.
Terza difficoltà: vincoli di fabbricazione reale. Anche se geometricamente perfetto, il risultato deve essere realizzabile con utensili reali, compatibile con G-code e con i limiti di risoluzione delle macchine.
Quarta difficoltà: mancanza di pipeline standard. Non esiste un workflow industriale standard: ogni implementazione è tipicamente sperimentale o personalizzata.
________________________________________
Obiettivo della tesi (versione pratica)
L’obiettivo di questa tesi è progettare e implementare una pipeline software capace di generare traiettorie di incisione per la realizzazione di scratch hologram a partire da modelli tridimensionali.
In particolare, il lavoro si concentra sulla parte implementativa del problema, sviluppando un flusso di lavoro che consenta di passare da un modello 3D (formato STL o equivalente) alla generazione di percorsi utensile esportabili in G-code.
La pipeline proposta prevede analisi e pre-elaborazione del modello 3D, generazione algoritmica delle traiettorie dei graffi, integrazione con slicer esistenti tramite modifica o estensione e simulazione del risultato tramite software CAM o rendering.
L’obiettivo non è sviluppare un sistema di fabbricazione completo da zero, ma esplorare la fattibilità di una soluzione implementativa basata sull’estensione di strumenti software esistenti.


Capitolo: Analisi degli strumenti esistenti
Sottoparagrafo: HoloZens

### Test preliminare HoloZens — Setup ambiente

Durante la fase iniziale del progetto è stato individuato il software HoloZens come possibile strumento di riferimento per la generazione e simulazione di strutture basate su scratch hologram. Il software è stato individuato tramite ricerca su repository pubblici, in particolare sulla piattaforma GitHub, dove è disponibile sotto forma di codice sorgente.

Dalla documentazione disponibile, HoloZens appare come un software sperimentale orientato alla generazione e visualizzazione di pattern ottici basati su incisioni speculari. Il progetto sembra essere pensato principalmente come strumento di ricerca o prototipo, piuttosto che come applicazione commerciale pronta all’uso.

Il software è sviluppato in linguaggio C# e non è distribuito come applicazione già compilata. L’autore indica esplicitamente la necessità di clonare il repository, compilare il progetto utilizzando Visual Studio e successivamente eseguire il programma su sistema operativo Windows.

Per questo motivo, l’utilizzo del software richiede la configurazione di un ambiente di sviluppo compatibile con applicazioni .NET, in particolare Microsoft Visual Studio con i componenti necessari per lo sviluppo desktop. Questo aspetto evidenzia come molti strumenti sperimentali in ambito computer graphics e fabbricazione digitale siano disponibili principalmente come codice sorgente, rendendo necessaria la conoscenza delle procedure di compilazione e build del software.


### Analisi sperimentale preliminare della robustezza rispetto alla complessità della mesh

Durante la fase iniziale di sperimentazione è stato analizzato il comportamento del software HoloZens rispetto a modelli tridimensionali con diversa complessità geometrica.

In una prima fase sono stati utilizzati modelli estremamente semplici generati proceduralmente tramite Blender, come ad esempio un cubo esportato in formato STL. In questo caso il software ha mostrato un comportamento stabile, consentendo la visualizzazione corretta del modello e l’interazione tramite operazioni di zoom e pan senza evidenziare anomalie o crash.

Successivamente sono stati testati modelli geometricamente più complessi, come ad esempio un nodo toroidale (torus knot). In questo caso il software ha mostrato alcune anomalie nella visualizzazione del modello e, dopo ripetute operazioni di zoom e navigazione, ha generato un crash dell’applicazione.

Questo comportamento suggerisce che la stabilità del software sia fortemente influenzata dalla complessità topologica della mesh e dalla qualità della discretizzazione triangolare del modello STL. In particolare, mesh con geometrie più complesse possono contenere triangoli molto piccoli, normali numericamente instabili o configurazioni degeneri che possono portare a instabilità numeriche durante il calcolo delle proprietà geometriche utilizzate dal software.

L’output del software è stato esportato in formato SVG, contenente le traiettorie vettoriali corrispondenti ai percorsi di incisione. Tali file sono stati analizzati tramite software di visualizzazione vettoriale e strumenti di modellazione 3D per verificare la struttura e la distribuzione delle traiettorie generate.

Durante le fasi di test il software ha mostrato la capacità di generare famiglie di traiettorie continue e coerenti con la geometria della superficie del modello. Variando i parametri di risoluzione e angolo di vista è stato possibile osservare variazioni prevedibili nella densità e nella disposizione delle curve generate, indicando un comportamento stabile del sistema di campionamento geometrico.

I tentativi di riprodurre risultati precedentemente dimostrati utilizzando fonti di progetto disponibili al pubblico non hanno prodotto risultati identici, il che suggerisce potenziali discrepanze tra le versioni del codice sorgente distribuito e le build dimostrative, oppure la sensibilità della pipeline a piccole variazioni nella struttura della mesh di input.



NOTES
Tried copying and pasting a knot image on chatgpt asking to give me a .stl file but it produced an horrendous big file which couldn't be opened from the program itself becasue maybe too large (it crashed every time I tried to open it, and my pc is not that bad lol)

On here:
https://www.printables.com/model/444295-trefoil-knot/files
I found this cool knot file which is very small in terms of dimensions


Now I've downloaded a thorus and tried with a cube but it seems like there are some problems:
Hard to manipulate the object and to learn how to use the program


-------------------------PARTE NUOVA IN PYTHON --------------

Obiettivo implementato:

Pipeline riproducibile da STL a traiettorie scratch in SVG, con simulazione interattiva del risultato.
Fase 1, import e preprocessing mesh:

Caricamento STL con trimesh.
Pulizia mesh (facce degeneri/duplicate, vertici non referenziati).
Rounding coordinate per stabilità numerica.
Auto-centering su asse Z (stile HoloZens).
Costruzione vertici unici, edge unici e facce rimappate.
Fase 2, estrazione geometria utile agli scratch:

Edge extraction con adiacenze di faccia.
Filtro coplanare (equivalente a “merge faces”) per rimuovere diagonali da triangolazione.
Calcolo normal per-edge (media normali facce adiacenti), utile per simulazione ottica.
Fase 3, modello di proiezione e generazione archi:

Implementazione matrice model -> view -> perspective -> window.
Campionamento punti lungo gli edge in base a line-resolution.
Costruzione archi semicircolari in 2D tramite profondità rispetto al piano vista.
Dedupe geometrico e filtro su raggio minimo (min-arc-radius) per eliminare micro-archi.
Export SVG con path ad archi (traiettorie incisione).
Fase 4, debug ed export dati:

Export JSON con parametri camera/pipeline e statistiche (edges_count, arcs_count).
Fase 5, simulazione interattiva HTML:

Slider View angle per simulazione visuale dell’effetto.
Camera orbit 3D (yaw, pitch, zoom) con proiezione corretta.
Controlli pattern (arc stride, arc limit, arc alpha, arc min r, view gain).
Fase 6, miglioramenti stile HoloCraft:

Depth sort reale degli archi (ordinamento per profondità).
Slider Depth e Thresh per selezione contributi visivi utili.
Nuovo Light slider e scoring luce (componente diffuse/specular semplificata).
Modalità render: Combined, Pattern, Wireframe, Opaque.
Pass Opaque con shading facce depth-sorted e alpha controllabile (Model alpha).
Test eseguiti:

basic_cube_-_10mm.stl: edge e archi coerenti con attese HoloZens-like.
trefoil.stl: pipeline stabile anche su mesh complesse (migliaia di edge e archi).
Risultato pratico:

Hai ora un generatore robusto STL -> SVG e una preview parametrica molto più vicina al comportamento dei tool competitor, mantenendo stack semplice e controllabile.




È stata sviluppata una GUI desktop in Python che integra visualizzazione interattiva e generazione degli scratch hologram, eliminando la dipendenza dalla preview HTML separata.
L’interfaccia è stata semplificata in modalità single-view, mostrando direttamente la preview degli archi su un’unica canvas.
La camera è controllabile con interazione diretta (drag per orbit, rotellina per zoom), così da impostare la vista di lavoro prima dell’esportazione.
Per ridurre il carico computazionale su mesh complesse, è stato introdotto un campionamento adattivo degli spigoli in preview.
L’algoritmo di alleggerimento agisce sia sul sottoinsieme di spigoli elaborati sia sulla line resolution effettiva, mantenendo la leggibilità visiva.
Durante il movimento della camera viene usata una modalità veloce, mentre a interazione terminata viene aggiornata una preview più accurata.
L’esportazione SVG viene invece eseguita in modalità completa, garantendo coerenza geometrica e qualità del risultato finale.
È stato aggiunto un controllo esplicito della qualità di preview, utile per bilanciare fluidità e dettaglio in funzione dell’hardware disponibile.
I test su modelli semplici (cubo) e ad alta complessità topologica (trefoil) mostrano un miglioramento significativo della reattività dell’interfaccia.
Questa scelta progettuale separa efficacemente rendering interattivo ed export finale, mantenendo una pipeline stabile e riproducibile.






---------------ROBA SCRITTA DA ZIO CHAT PER LA PARTE DI VALIDAZIONE DEL GCODE---------

## Capitolo X - Validazione del G-code generato

### X.1 Obiettivo della validazione

Dopo l’implementazione della pipeline di generazione degli scratch hologram (da modello STL a traiettorie 2D e relativi export), è stata eseguita una fase specifica di validazione del **G-code** prodotto dal programma, con l’obiettivo di verificare:

- correttezza sintattica del file G-code;
- coerenza geometrica delle traiettorie rispetto alla preview del software;
- corretto utilizzo delle primitive di movimento CNC in base alla modalità selezionata;
- assenza di anomalie evidenti nei movimenti rapidi e nelle quote Z.

Questa fase è fondamentale perché rappresenta il passaggio tra una simulazione grafica (preview/SVG) e un output effettivamente utilizzabile in un contesto di lavorazione CNC.

### X.2 Strumento di test scelto: CutViewer

Per la validazione preliminare è stato scelto **CutViewer** (`https://cutviewer.com/app`), un visualizzatore/simulatore G-code via browser.

La scelta è stata motivata da ragioni pratiche:

- interfaccia semplice e immediata;
- caricamento rapido dei file `.nc`;
- visualizzazione simultanea di codice e traiettorie;
- sufficiente per verificare il comportamento del G-code generato in questa fase del progetto.

Era stata considerata anche l’alternativa **CAMotics**, più completa ma meno immediata per l’obiettivo corrente (verifica rapida e focalizzata delle traiettorie generate).

### X.3 Configurazione dei test

I test sono stati eseguiti a partire dal modello di riferimento:

- `basic_cube_-_10mm.stl`

Sono stati generati due file G-code, corrispondenti alle due modalità di generazione degli archi supportate dal programma:

- modalità **Semicircle (CNC)**;
- modalità **Elliptic**.

L’idea è confrontare due strategie geometriche equivalenti sul piano concettuale (stesso insieme di scratch), ma differenti dal punto di vista della rappresentazione CNC.

### X.4 Criteri di verifica adottati

La validazione è stata svolta con una combinazione di controlli statici (lettura del G-code) e dinamici (ispezione della traiettoria in CutViewer).

#### X.4.1 Controlli statici sul file G-code

Sono stati verificati i seguenti elementi nel file esportato:

- presenza dell’header CNC corretto;
- unità di misura in millimetri (`G21`);
- coordinate assolute (`G90`);
- piano di lavoro XY (`G17`);
- feed in unità/minuto (`G94`);
- quota di sicurezza (`G0 Z...`) prima degli spostamenti rapidi;
- quota di incisione (`G1 Z...`) coerente con il preset impostato.

Nel caso testato, il file mostrava correttamente comandi del tipo:

- `G21`, `G90`, `G17`, `G94`
- `G0 Z3.0000` (quota sicura)
- `M3 S12000` (avvio mandrino)
- `G1 Z-0.0800 F220.00` (discesa a quota incisione)

#### X.4.2 Controlli dinamici in simulazione

In CutViewer sono stati controllati:

- forma globale della traiettoria;
- coerenza con il pattern atteso del cubo scratch;
- assenza di archi “impazziti” o segmenti fuori figura;
- corretto alternarsi di:
  - rapido (`G0`) a quota sicura,
  - plunge (`G1 Z...`),
  - passata di incisione (`G2/G3` oppure `G1` segmentati);
- continuità e regolarità del tracciato.

### X.5 Risultati del test - Modalità Semicircle (CNC)

Nella modalità **Semicircle (CNC)**, il G-code generato utilizza archi circolari nativi CNC tramite comandi `G2/G3`.

#### Osservazioni principali

- Le traiettorie risultano coerenti con la geometria prevista.
- Il pattern del cubo è riconoscibile e stabile.
- I movimenti rapidi risultano eseguiti a quota sicura.
- I movimenti di taglio usano correttamente primitive ad arco (`G2/G3`).
- Il file risulta relativamente compatto (numero di righe contenuto rispetto alla modalità elliptic).

Questo comportamento è coerente con l’obiettivo della modalità “CNC”, che privilegia una rappresentazione efficiente e direttamente compatibile con controller che supportano bene gli archi circolari.

### X.6 Risultati del test - Modalità Elliptic

Nella modalità **Elliptic**, le traiettorie non possono essere rappresentate direttamente con `G2/G3`, poiché il G-code standard gestisce archi circolari ma non ellissi generiche.

Per questo motivo, il programma esporta le ellissi come **polilinee** composte da molti segmenti `G1`.

#### Osservazioni principali

- Il comportamento in simulazione risulta corretto e coerente con la forma attesa.
- Le traiettorie appaiono più “dense” e con un numero di segmenti molto maggiore.
- Il numero totale di righe del file cresce sensibilmente rispetto alla modalità semicircle.
- La forma finale è visivamente più vicina a un arco ellittico schiacciato.

In pratica:

- gli **archi logici** (scratch) restano gli stessi;
- aumenta il numero di **movimenti macchina** necessari per approssimarli.

Questo era atteso ed è un risultato corretto.

### X.7 Confronto tra le due modalità di export CNC

Dal punto di vista della validazione, entrambe le modalità risultano funzionanti, ma con trade-off diversi.

#### Modalità Semicircle (CNC)

- Primitive principali: `G2/G3`
- Vantaggi:
  - file più corto;
  - meno movimenti;
  - maggiore efficienza esecutiva;
  - ottima compatibilità CNC.
- Svantaggi:
  - geometria limitata ad archi circolari.

#### Modalità Elliptic

- Primitive principali: `G1` (segmentazione)
- Vantaggi:
  - maggiore libertà geometrica;
  - possibilità di ottenere scratch più “schiacciati”/morbidi.
- Svantaggi:
  - file più lungo;
  - più segmenti;
  - tempi macchina potenzialmente maggiori.

### X.8 Interpretazione dei risultati

La validazione in CutViewer ha mostrato che il G-code prodotto dal programma è:

- **sintatticamente corretto**;
- **geometricamente coerente** con il pattern generato;
- **consistente** con la modalità di esportazione selezionata (`semicircle` vs `elliptic`).

In particolare, il test conferma che:

- la modalità `Semicircle (CNC)` esporta effettivamente archi circolari tramite `G2/G3`;
- la modalità `Elliptic` esporta effettivamente una discretizzazione in segmenti `G1`;
- i movimenti in Z (sicurezza/taglio) sono gestiti correttamente nella sequenza di lavorazione.

### X.9 Limiti della validazione eseguita

La validazione svolta è da considerarsi **preliminare ma significativa**.

Limiti principali:

- CutViewer è stato usato come strumento di verifica delle traiettorie, non come simulatore CAM completo di processo;
- non è stata ancora eseguita una validazione su macchina reale;
- non sono stati ancora ottimizzati i tempi di percorso (ordinamento delle traiettorie, riduzione dei rapidi inutili).

Questi aspetti rappresentano una naturale estensione del lavoro, ma non compromettono la validità della dimostrazione attuale della pipeline.

### X.10 Conclusione del capitolo

La fase di test del G-code ha confermato che il sistema sviluppato produce output CNC coerente e utilizzabile per una simulazione preliminare di lavorazione.

Il risultato è particolarmente rilevante perché dimostra la correttezza del passaggio:

**modello STL -> generazione scratch -> export geometrico -> G-code CNC simulabile**

In altre parole, la pipeline non si limita a una visualizzazione grafica (preview/SVG), ma arriva a un formato operativo compatibile con un flusso di lavorazione CNC reale.


------------------------------


----------------------------

Fine tesi → Appendice A → Glossario

Glossario dei termini principali
Scratch hologram

Tecnica di rappresentazione tridimensionale che utilizza micro-incisioni su una superficie riflettente per simulare la presenza di punti nello spazio tramite riflessione speculare della luce.

Specular reflection

Fenomeno ottico in cui la luce viene riflessa da una superficie liscia mantenendo un angolo di riflessione uguale all’angolo di incidenza rispetto alla normale della superficie.

STL (Stereolithography file format)

Formato file standard utilizzato per rappresentare modelli tridimensionali tramite mesh di triangoli. È ampiamente utilizzato nei processi di stampa 3D e fabbricazione digitale.

Slicer

Software che converte un modello 3D in istruzioni macchina (tipicamente G-code). Nella stampa 3D tradizionale divide il modello in strati; in altri contesti può generare traiettorie utensile.

G-code

Linguaggio standard utilizzato per controllare macchine CNC, stampanti 3D e dispositivi simili. Contiene istruzioni di movimento, velocità e attivazione degli strumenti.

Pipeline

Sequenza organizzata di passaggi software che trasformano progressivamente un input iniziale (es. modello 3D) in un output finale (es. istruzioni macchina o simulazione).

CNC (Computer Numerical Control)

Tecnologia di controllo automatico di macchine utensili tramite istruzioni digitali, generalmente espresse in G-code.