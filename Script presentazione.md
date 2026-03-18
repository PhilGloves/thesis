Slide 1 — Titolo

Tempo: circa 50 secondi

“Buongiorno a tutti, sono [NOME COGNOME] e oggi vi presento il mio lavoro di tesi, intitolato Pipeline computazionale per la fabbricazione digitale di scratch hologram: dal modello 3D al G-code.

Questo lavoro nasce dall’interesse per gli scratch hologram, cioè immagini incise su superfici riflettenti che possono produrre un’illusione di profondità.

Dal punto di vista informatico, la domanda che mi interessava affrontare era come passare da un modello tridimensionale, descritto come mesh triangolare, a un pattern bidimensionale controllabile, visualizzabile ed esportabile.

Quindi la tesi si colloca tra geometria computazionale, rappresentazione grafica e preparazione alla fabbricazione digitale.”

Slide 2 — Contesto e obiettivo

Tempo: circa 1 minuto e 30

“In questa slide introduco il contesto generale del lavoro.

Gli scratch hologram sono immagini incise su superfici riflettenti che possono produrre un’illusione di profondità. In altre parole, la forma tridimensionale non viene ricostruita come volume reale, ma suggerita attraverso un pattern di incisioni che modifica il modo in cui la luce viene riflessa.

Dal punto di vista informatico, questo porta a un problema interessante: come si può trasformare una mesh STL, quindi una geometria tridimensionale composta da triangoli, in un pattern 2D che sia abbastanza controllabile da poter essere analizzato, visualizzato e potenzialmente fabbricato?

L’obiettivo della tesi è stato quindi progettare una pipeline riproducibile, con preview interattiva e validazione qualitativa.

È importante chiarire che il focus del lavoro non è stato costruire una simulazione ottica completa del fenomeno, che sarebbe un problema molto più ampio, ma definire una pipeline geometrica coerente, in cui fosse possibile controllare i parametri principali e osservare in modo chiaro il loro effetto sul pattern finale.

Questa impostazione mi ha permesso di concentrarmi sulla struttura del processo: input 3D, trasformazioni geometriche, generazione del pattern, preview, esportazione e verifica dei risultati.”

Slide 3 — Dallo stato dell’arte alla proposta

Tempo: circa 1 minuto e 45

“Prima di sviluppare la pipeline, ho analizzato alcune soluzioni esistenti.

In particolare, ho preso come riferimento HoloZens, che è una fork di un progetto originariamente sviluppato da Mike Miller. A sua volta, quel lavoro riprendeva idee e principi già presenti anche in risorse amatoriali storiche dedicate agli scratch hologram, come il sito di Amasci.

Questo passaggio è stato utile perché mi ha permesso di capire come altri autori avevano affrontato il problema della trasformazione di una geometria 3D in uno scratch pattern.

All’inizio avevo considerato la possibilità di usare direttamente quel software, ma cercando di lavorarci ho riscontrato diversi bug e alcune limitazioni pratiche che rendevano difficile adottarlo così com’era come base di tesi.

A quel punto ho preferito reimpostare la pipeline in un ambiente più adatto alla prototipazione rapida e alla sperimentazione iterativa, scegliendo Python.

La scelta non è nata tanto da una motivazione teorica forte legata al linguaggio in sé, quanto dall’esigenza di lavorare in modo più flessibile, modificare rapidamente i parametri, testare più versioni e integrare con maggiore facilità le varie fasi del processo.

Quindi la proposta della tesi nasce proprio da questo: partire dallo stato dell’arte esistente, in particolare da software open source già disponibile, e trasformarlo in una pipeline più chiara, controllabile e più adatta a essere analizzata e validata in un contesto accademico.”

Slide 4 — Pipeline proposta

Tempo: circa 1 minuto e 40

“Questa slide mostra la struttura generale della pipeline.

Si parte da un input in formato STL. Da lì vengono estratti gli spigoli rilevanti della mesh, che poi vengono campionati lungo la loro lunghezza. I punti campionati vengono proiettati secondo una configurazione di vista scelta dall’utente e successivamente sottoposti a un filtraggio di visibilità, cioè a una fase di culling.

A partire da questi dati viene poi generato il pattern finale sotto forma di archi nel piano bidimensionale. Questo pattern può essere visualizzato in preview e poi esportato.

Qui il punto importante è che il risultato della pipeline non è prima di tutto un render o una semplice immagine, ma una struttura geometrica vera e propria, fatta di archi ed elementi geometrici espliciti, che può poi essere visualizzata ed esportata.

Questo rende il processo più leggibile e più facilmente controllabile.

Un altro aspetto importante è che ogni passaggio dipende dai precedenti. Per esempio, una diversa densità di campionamento cambia il numero di archi; una diversa vista cambia il pattern proiettato; un diverso filtraggio di visibilità modifica quali parti contribuiscono al risultato finale.

Quindi la pipeline non è solo una catena di conversioni, ma un processo in cui le scelte fatte a monte influenzano direttamente la leggibilità e la qualità del pattern.”

Slide 5 — Software sviluppato

Tempo: circa 1 minuto e 35

“In questa slide descrivo il software sviluppato.

L’architettura è stata organizzata separando il nucleo geometrico dall’interfaccia desktop. Questo permette di distinguere chiaramente la parte che carica il modello, genera il pattern e gestisce gli output dalla parte usata per la preview interattiva e per la regolazione dei parametri.

Dal punto di vista pratico, l’applicazione consente di caricare modelli STL, modificare la configurazione di vista, regolare densità di campionamento, modalità degli archi e filtraggio di visibilità, e osservare immediatamente l’effetto di queste scelte sul pattern.

Questo aspetto è stato utile soprattutto nella fase di test, perché ha permesso di confrontare rapidamente configurazioni diverse e individuare i parametri più sensibili.

Più che un software pensato fin da subito come strumento finale di produzione, lo definirei uno strumento sperimentale ben strutturato, utile per capire il comportamento della pipeline e verificare in modo abbastanza sistematico la relazione tra parametri e risultati.”

Slide 6 — Test e validazione

Tempo: circa 2 minuti

“Questa slide riguarda la parte di test e validazione.

Una parte importante del lavoro è stata capire come i parametri influenzassero concretamente il pattern generato. In particolare, qui riporto un test sul cubo in cui ho variato il parametro line_resolution, cioè la densità di campionamento.

Il cubo è stato scelto come caso di studio principale perché è una geometria semplice, leggibile e utile per osservare con chiarezza come cambia il numero di archi al variare dei parametri.

Quello che si vede è che aumentando la densità di campionamento cresce il numero di traiettorie generate. Questo rende il pattern più ricco, ma non automaticamente migliore: oltre una certa soglia aumenta anche il rischio di avere un pattern più denso e quindi meno leggibile.

Durante lo sviluppo, però, non tutti i test producevano risultati coerenti. In alcune configurazioni, modelli molto semplici come il cubo perdevano completamente la propria forma attesa, e il pattern risultante assumeva geometrie deformate, a volte simili a parallelepipedi irregolari o a strutture difficili da interpretare.

Questi casi sono stati utili perché hanno mostrato che la pipeline era molto sensibile a errori di proiezione, parametri di campionamento e gestione della visibilità. In altre parole, non bastava ottenere un output: bisognava anche verificarne la coerenza geometrica.

La validazione, in questa fase, è stata soprattutto qualitativa. L’obiettivo non era ottenere una misura numerica assoluta di qualità, ma verificare se il sistema produceva risultati coerenti e interpretabili al variare dei parametri principali.”

Slide 7 — Risultati qualitativi

Tempo: circa 2 minuti e 10

“In questa slide mostro alcuni risultati qualitativi su casi diversi, in particolare il cubo e il d20.

Qui entra in gioco anche la parte di validazione tramite strumenti esterni. In particolare ho usato Blender e CutViewer, che hanno ruoli diversi.

Blender è stato utile come ambiente di simulazione visiva. Mi ha permesso di importare gli SVG esportati dalla pipeline e osservare il pattern in un contesto di rendering, quindi di verificare meglio la distribuzione delle traiettorie, la leggibilità complessiva e la percezione della struttura geometrica.

CutViewer invece è stato usato per simulare il toolpath CNC a partire dal G-code. In questo caso la verifica non riguardava tanto l’effetto visivo finale, quanto la plausibilità della traiettoria macchina generata dalla pipeline.

Nei casi mostrati qui si vede che il sistema funziona meglio su modelli semplici o moderatamente complessi, mentre su geometrie più dense il pattern tende a diventare più difficile da leggere.

Questa parte, secondo me, è stata importante perché mi ha permesso non solo di vedere se la pipeline produceva un risultato, ma anche di confrontare diverse rappresentazioni dello stesso pattern e capire in quali casi il metodo risultava più convincente.”

Slide 8 — Effetto dei parametri sul pattern

Tempo: circa 2 minuti

“Questa slide mostra in modo abbastanza chiaro quanto il pattern dipenda dai parametri scelti.

Qui il primo parametro importante è l’angolo di vista. Cambiando la vista, cambia la proiezione della mesh e quindi cambia anche il pattern risultante. Questo è naturale, ma è importante vederlo in modo esplicito, perché significa che la vista non è un dettaglio secondario: fa parte del comportamento del sistema.

Il secondo aspetto importante è la modalità di generazione degli archi. Nella pipeline ho considerato sia una modalità semicircle, più semplice e regolare, sia una modalità elliptic, che introduce traiettorie più morbide e continue.

La modalità ellittica è interessante perché in alcuni casi produce pattern visivamente più fluidi e continui. Tuttavia, in ottica di fabbricazione digitale, la modalità semicircolare è in genere più vantaggiosa, perché si adatta meglio alle capacità native delle macchine CNC di eseguire archi tramite movimenti G2 e G3, invece di approssimare tutto con segmenti lineari G1.

Questo, almeno in linea teorica e implementativa, porta a file più compatti, programmi più brevi e tempi di esecuzione potenzialmente inferiori. Non ho ancora effettuato test sistematici su questo aspetto, ma è sicuramente uno sviluppo che intendo approfondire.

Questi test mi hanno fatto capire che non esiste una configurazione ottimale universale. Piuttosto, esiste un equilibrio tra vista, densità, modalità geometrica e filtraggio di visibilità.

Secondo me questo è un risultato importante della tesi: non solo generare un pattern, ma mostrare in modo abbastanza chiaro come i parametri ne influenzino la forma e la leggibilità.”

Slide 9 — Limiti attuali e sviluppi futuri

Tempo: circa 2 minuti e 30

“Questa slide raccoglie i limiti principali del lavoro e anche i possibili sviluppi futuri.

Il primo limite è che, pur essendo coerente e modulare, la pipeline resta una semplificazione geometrica. Non descrive in modo completo il fenomeno ottico reale dello scratch hologram.

Il secondo limite è che, su mesh più complesse, aumentano sia la densità del pattern sia il costo computazionale, e questo può ridurre la leggibilità locale della preview.

Per quanto riguarda invece la parte fisica, qui devo fare una precisazione rispetto a quanto scritto nella tesi. Dopo la consegna sono riuscito a fare anche alcune prove reali di incisione.

Ho testato un piano in acrilico nero riflettente, ma i risultati non sono stati molto precisi perché il piano presentava micro-imperfezioni. Impostando un graffio molto leggero, circa a meno 0,02 millimetri, la CNC ha prodotto incisioni più o meno evidenti a seconda delle deformazioni locali della superficie, e questo rendeva il pattern poco leggibile nel suo insieme.

Quando invece il graffio diventava troppo profondo, emergeva un secondo problema: l’acrilico inciso tendeva a diventare bianco e opaco, perdendo riflettenza. Di conseguenza la luce non riusciva più a riflettersi nel modo desiderato, e l’effetto visivo peggiorava ulteriormente.

In questo caso, probabilmente la soluzione migliore sarebbe stata usare una punta di incisione flottante, con una molla abbastanza morbida da compensare le piccole difformità del piano e mantenere una profondità di graffio più costante anche su superfici non perfettamente uniformi.

Ho fatto però anche prove su pannelli di alluminio, e lì i risultati sono stati più soddisfacenti, perché il metallo mantiene meglio le proprietà riflettenti anche con graffi più profondi. Quindi si ottiene un compromesso migliore tra visibilità dell’incisione e mantenimento dell’effetto riflettente.

Per questo motivo, oggi direi che i risultati sono promettenti ma non conclusivi: le simulazioni, in particolare quelle in Blender, restano più leggibili e controllabili, mentre la fabbricazione reale apre problemi pratici molto interessanti, soprattutto legati al materiale e al controllo della profondità di incisione.

Ed è proprio qui che vedo gli sviluppi futuri più interessanti: migliorare il culling, rendere il campionamento più adattivo, ottimizzare il toolpath e soprattutto fare più test fisici con setup meccanici più adatti.”

Slide 10 — Grazie

Tempo: circa 45–50 secondi

“Per concludere, quello che considero più importante di questo lavoro è aver reso esplicito un processo che spesso rimane implicito: il passaggio da una geometria tridimensionale a un pattern inciso controllabile, osservabile e verificabile.

La tesi non pretende di risolvere in modo definitivo il problema degli scratch hologram, ma propone una base software chiara e analizzabile, su cui è possibile continuare a lavorare.

Vi ringrazio per l’attenzione e resto a disposizione per eventuali domande.”