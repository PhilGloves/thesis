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

Test preliminare HoloZens — Setup ambiente

Durante la fase iniziale del progetto è stato individuato il software HoloZens come possibile strumento di riferimento per la generazione e simulazione di strutture basate su scratch hologram. Il software è stato individuato tramite ricerca su repository pubblici, in particolare sulla piattaforma GitHub, dove è disponibile sotto forma di codice sorgente.

Dalla documentazione disponibile, HoloZens appare come un software sperimentale orientato alla generazione e visualizzazione di pattern ottici basati su incisioni speculari. Il progetto sembra essere pensato principalmente come strumento di ricerca o prototipo, piuttosto che come applicazione commerciale pronta all’uso.

Il software è sviluppato in linguaggio C# e non è distribuito come applicazione già compilata. L’autore indica esplicitamente la necessità di clonare il repository, compilare il progetto utilizzando Visual Studio e successivamente eseguire il programma su sistema operativo Windows.

Per questo motivo, l’utilizzo del software richiede la configurazione di un ambiente di sviluppo compatibile con applicazioni .NET, in particolare Microsoft Visual Studio con i componenti necessari per lo sviluppo desktop. Questo aspetto evidenzia come molti strumenti sperimentali in ambito computer graphics e fabbricazione digitale siano disponibili principalmente come codice sorgente, rendendo necessaria la conoscenza delle procedure di compilazione e build del software.






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