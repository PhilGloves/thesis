Se ti chiedono perché hai usato python

All’inizio avevo considerato la possibilità di usare direttamente HoloZens (hz-scratch-hologram-master) come base di partenza. Tuttavia, durante le prove pratiche, sono emersi diversi bug e alcune limitazioni operative che rendevano difficile adottarlo così com’era all’interno del lavoro di tesi. In particolare, il progetto risultava meno adatto a un flusso di prototipazione rapida, sperimentazione iterativa e integrazione diretta tra le diverse fasi della pipeline.

Per questo motivo ho preferito reimpostare il lavoro in Python. La scelta non è nata tanto da una motivazione teorica legata al linguaggio in sé, quanto dall’esigenza di lavorare in modo più flessibile: modificare rapidamente i parametri, testare versioni alternative degli algoritmi, automatizzare le verifiche e integrare più facilmente caricamento della mesh, proiezione, generazione degli archi ed esportazione del G-code.

Inoltre, rispetto a un’applicazione desktop C# già strutturata, Python offre un ambiente più leggero per la sperimentazione algoritmica. Librerie come NumPy permettono di implementare e testare operazioni geometriche e numeriche in modo diretto, senza dover costruire da zero tutta l’infrastruttura di supporto.

Questa scelta si è rivelata particolarmente utile perché mi ha permesso di intervenire direttamente sugli aspetti più critici della pipeline, come la gestione della visibilità, l’ordinamento delle traiettorie e l’esportazione verso il G-code.

------------
LIBRERIE

Ho usato librerie dove aveva senso farlo, soprattutto per la gestione di base della mesh e del calcolo numerico. In particolare trimesh mi serve per caricare e pulire il file STL, mentre numpy mi serve per gestire in modo efficiente vettori, matrici e trasformazioni geometriche.
La logica specifica della pipeline, però, cioè come passare dal modello 3D al pattern di graffi, è stata costruita nel progetto: estrazione degli spigoli rilevanti, campionamento dei punti, proiezione, filtro di visibilità e generazione finale degli archi.


-----------
Come funziona la pipeline?
La pipeline non incide direttamente tutta la superficie del modello 3D. Parte invece dalla mesh STL e ne estrae gli elementi più informativi, in particolare gli spigoli. Lungo questi spigoli vengono presi molti punti campionati. I punti vengono poi proiettati sul piano della vista corrente, così il modello 3D viene trasformato in una rappresentazione bidimensionale coerente con l’osservazione scelta. A questo punto il software conserva i campioni utili alla vista e, per ciascuno di essi, genera un piccolo arco/graffio 2D. Il risultato finale è quindi un pattern di archi che può essere mostrato in preview ed esportato come SVG o G-code.

----------------



Di cosa parla la tua tesi in una frase?

La tesi propone una pipeline che parte da un modello 3D in formato STL e lo trasforma in un pattern 2D di graffi/arci, utilizzabile per preview ed esportazione verso SVG e G-code per scratch hologram.
Che cos’è, in parole semplici, uno scratch hologram?

È un’immagine ottenuta non con colori o voxel, ma con piccoli graffi orientati, che riflettono la luce in modo controllato e fanno percepire una forma o un effetto tridimensionale.
Qual è il problema che hai affrontato?

Il problema era passare in modo coerente da una mesh 3D a un pattern 2D di incisione, mantenendo un legame tra geometria del modello, visibilità nella vista scelta e traiettorie finali da esportare.
Qual è il contributo principale del tuo lavoro?

Il contributo principale è una pipeline unificata che integra caricamento della mesh, estrazione degli elementi rilevanti, proiezione, generazione degli archi, preview e export, invece di lasciare questi passaggi separati o dipendenti da software esterni.
Domande Sulla Pipeline

Come avviene il passaggio dal modello 3D al pattern 2D?

Si carica la mesh STL, si estraggono soprattutto gli spigoli, si campionano punti lungo questi spigoli, si proiettano nella vista corrente e da quei punti si generano archi 2D che formano il pattern finale.
Perché hai usato gli spigoli e non tutta la superficie?

Perché gli spigoli sono elementi geometrici molto informativi: descrivono bene la forma locale e permettono di costruire un pattern discreto più controllabile, evitando di trattare l’intera superficie come qualcosa da incidere uniformemente.
Perché hai bisogno di campionare punti sugli spigoli?

Perché gli spigoli da soli sono segmenti continui, mentre la generazione del pattern richiede elementi discreti. Il campionamento crea punti di riferimento da cui costruire gli archi e consente di controllare densità e dettaglio.
Che ruolo ha la proiezione?

La proiezione serve a trasformare il modello tridimensionale in coordinate coerenti con la vista scelta. In questo modo il pattern finale non rappresenta il modello in astratto, ma il modo in cui quel modello appare da una determinata osservazione.
Perché il pattern finale è bidimensionale se il modello è tridimensionale?

Perché il supporto fisico da incidere è piano. Quindi l’informazione 3D deve essere convertita in una struttura 2D che mantenga comunque una relazione visiva con il modello originale.
Domande Tecniche Ma Abbordabili

Hai implementato tutto da zero?

No. Ho usato librerie dove aveva senso farlo, soprattutto per lettura della mesh e calcolo numerico. Però la logica specifica della pipeline, cioè come passare dal modello al pattern di graffi, è stata costruita e adattata nel progetto.
Quali librerie hai usato?

Principalmente trimesh per caricare e ripulire le mesh STL e numpy per i calcoli numerici e geometrici. La pipeline sopra questi strumenti è però definita nel codice del progetto.
Quindi cosa fa trimesh e cosa fai tu?

trimesh mi fornisce una mesh triangolare pulita, con vertici, facce e normali. Il progetto poi costruisce gli spigoli rilevanti, campiona i punti, li proietta, applica il filtro di visibilità e genera il pattern finale di archi.
Perché hai scelto Python?

Perché mi serviva un ambiente adatto alla prototipazione rapida e alla sperimentazione iterativa. Python, con librerie come numpy, mi ha permesso di modificare più facilmente i parametri e verificare rapidamente varianti della pipeline.
Perché non hai usato direttamente HoloZens?

L’avevo considerato, ma durante le prove ho trovato bug e limiti pratici che rendevano difficile usarlo come base diretta della tesi. Ho quindi preferito ricostruire una pipeline più controllabile e meglio integrabile con il workflow che mi serviva.
Domande Su Preview, Export e Risultati

Perché hai sia preview che export?

Perché la preview serve a controllare visivamente il risultato, mentre l’export serve a trasformarlo in un output realmente utilizzabile, come SVG o G-code. Un obiettivo importante era mantenerli coerenti tra loro.
Perché esporti sia SVG che G-code?

SVG è utile come formato vettoriale intermedio e leggibile, mentre G-code è il formato direttamente eseguibile dalla macchina CNC. Averli entrambi rende la pipeline più flessibile.
Come hai verificato che il risultato fosse corretto?

Ho usato sia modelli semplici, come il cubo, per verificare il comportamento della pipeline in modo controllato, sia modelli più complessi, come il D20, per osservare la robustezza del metodo su geometrie meno banali.
Perché nelle slide mostri il cubo e non solo modelli più complessi?

Perché il cubo è il caso più leggibile per spiegare i passaggi della pipeline. Essendo semplice, permette di capire bene campionamento, proiezione e pattern finale senza essere confusi dalla complessità della mesh.
Domande Sui Limiti

Quali sono i limiti del tuo lavoro?

I limiti principali sono la dipendenza dalla qualità della mesh, la sensibilità ai parametri di campionamento e i vincoli fisici del processo di incisione reale, per esempio planarità del supporto, profondità effettiva e comportamento dell’utensile.
Il sistema funziona su qualsiasi mesh STL?

Funziona bene su molte mesh triangolari, ma non tutte le mesh sono equivalenti: modelli molto rumorosi, estremamente densi o con topologia problematica possono richiedere più pulizia o parametri diversi.
Quanto conta la parte fisica rispetto alla parte software?

Conta molto. Il software genera il pattern corretto dal punto di vista geometrico, ma il risultato finale dipende anche da materiale, profondità di incisione, planarità del pezzo e comportamento meccanico dell’utensile.
Hai trovato problemi nella prova pratica su CNC?

Sì, soprattutto nella gestione della profondità reale di incisione. Anche differenze molto piccole nella planarità del materiale possono influenzare la presenza o l’assenza di certi graffi, oppure renderli troppo opachi.
Perché alcuni graffi non riflettevano bene?

Perché se l’incisione è troppo profonda o troppo aggressiva, la superficie diventa più diffusa e opaca invece che speculare. In quel caso il graffio non produce più un glint pulito ma una traccia biancastra o lattiginosa.
Domande Di Confronto / Scelte Progettuali

In cosa il tuo progetto migliora rispetto ai repository esistenti che hai studiato?

Il miglioramento principale è l’integrazione in una pipeline più coerente e sperimentabile: input STL diretto, preview, export e maggiore controllo sui passaggi intermedi, senza dipendere da strumenti esterni come SketchUp.
Il tuo lavoro è una reimplementazione o una riprogettazione?

Direi una riprogettazione ispirata a lavori precedenti. Alcune idee geometriche di base sono simili, ma la pipeline è stata riorganizzata e adattata agli obiettivi della tesi e al nuovo flusso di lavoro.
Domande Sul Futuro

Come miglioreresti il progetto in futuro?

Lavorerei su tre fronti: maggiore robustezza del filtro di visibilità, ottimizzazione delle traiettorie per ridurre i tempi macchina e integrazione con una gestione più adattiva della profondità, per compensare meglio le imperfezioni del supporto fisico.
Qual è il passo successivo più naturale?

Rendere il collegamento tra modello geometrico e fabbricazione più robusto, cioè passare da una pipeline che funziona bene in simulazione e in molti casi pratici a una pipeline più affidabile anche rispetto alle variabili meccaniche reali.
Domande Un Po’ Più “Scomode”

Quanto di questo lavoro è tuo e quanto deriva da librerie o strumenti esterni?

Le librerie mi hanno dato strumenti di base, come caricamento mesh e algebra numerica. Il lavoro di tesi sta nella progettazione, integrazione e adattamento della pipeline: scelta della rappresentazione, passaggi di trasformazione, parametri, debugging, verifica e collegamento con l’output finale.
Hai usato strumenti di AI o generazione assistita nello sviluppo?

Ho usato strumenti di supporto allo sviluppo come aiuto alla programmazione e alla prototipazione, ma le scelte progettuali, la verifica dei risultati e l’integrazione finale nella pipeline sono state parte del lavoro di tesi.


"CHE VUOL DIRE FABBRICAZIONE DIGITALE?"

La parola fabbricazione digitale (o digital fabrication, o fabbing) fa riferimento al processo attraverso cui è possibile creare oggetti solidi e tridimensionali partendo da disegni digitali. Questo processo, utilizzato ampiamente in manifattura per la creazione rapida di modelli e prototipi, può sfruttare diverse tecniche di fabbricazione sia additive (come la stampa 3D), sia sottrattive come il taglio laser e la fresatura.