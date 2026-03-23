Slide 1 — Titolo

Buongiorno a tutti, sono Filippo Guanti e oggi vi presento il mio lavoro di tesi, intitolato Pipeline computazionale per la fabbricazione digitale di scratch hologram: dal modello 3D al G-code.
Questo lavoro nasce dall’interesse per gli scratch hologram, cioè immagini incise su superfici riflettenti che possono produrre un’illusione di profondità.
Dal punto di vista informatico, la domanda che mi interessava affrontare era come passare da un modello tridimensionale, descritto come mesh triangolare, cioè una geometria 3D approssimata tramite un insieme di triangoli, a un pattern bidimensionale controllabile, visualizzabile ed esportabile.
Quindi il punto centrale della tesi è stato capire come trasformare una forma 3D in un insieme di traiettorie 2D che mantenessero una relazione leggibile con il modello di partenza.

Slide 2 — Contesto e obiettivo

Gli scratch hologram sono immagini incise su superfici riflettenti che possono produrre un’illusione di profondità. In altre parole, la forma tridimensionale non viene ricostruita come volume reale, ma suggerita attraverso un pattern di incisioni che modifica il modo in cui la luce viene riflessa.

Dal punto di vista informatico, questo porta a un problema interessante: come si può trasformare un modello 3d (sotto forma di file STL) in un pattern 2D che sia abbastanza controllabile da poter essere analizzato, visualizzato e potenzialmente fabbricato?
L’obiettivo della tesi è stato quindi progettare una pipeline riproducibile, con preview interattiva e validazione qualitativa.
È importante chiarire che il focus del lavoro non è stato costruire una simulazione ottica completa del fenomeno, ma definire una pipeline geometrica coerente, in cui fosse possibile controllare i parametri principali e osservare in modo chiaro il loro effetto sul pattern finale.

Questa impostazione mi ha permesso di concentrarmi soprattutto sulla struttura del processo: input 3D, trasformazioni geometriche, generazione del pattern, preview, esportazione e verifica dei risultati.”

Slide 3 — Dallo stato dell’arte alla proposta

Innanzitutto ho iniziato il mio lavoro di tesi ricercando e analizzando alcune soluzioni esistenti.

In particolare, ho preso come riferimento HoloZens, che è una fork di un progetto originariamente sviluppato da Mike Miller. A sua volta, quel lavoro riprendeva idee e principi già presenti anche in risorse amatoriali storiche dedicate agli scratch hologram, come il sito di Amasci, del ricercatore William J. Beaty, un Ingegnere ricercatore americano che aveva investigato sul fenomeno già a partire dal 1994.

Questo passaggio è stato utile perché mi ha permesso di capire come altri autori avevano affrontato il problema della trasformazione di una geometria 3D in uno scratch pattern.

All’inizio avevo considerato la possibilità di usare direttamente quel software, ma cercando di lavorarci ho riscontrato diversi bug e alcune limitazioni pratiche che rendevano difficile adottarlo così com’era come base di tesi.

A quel punto ho preferito reimpostare la pipeline in un ambiente più adatto alla prototipazione rapida e alla sperimentazione iterativa, scegliendo Python.

Rispetto al C#, usato nel progetto originale, Python mi permetteva di testare più rapidamente le modifiche e di sfruttare librerie numeriche molto potenti, come NumPy, utili per lavorare su trasformazioni geometriche e algoritmi visivi senza dover costruire tutto da zero.

Quindi la proposta della tesi nasce proprio da questo: partire dallo stato dell’arte esistente, capirne i punti forti e i limiti, e costruire una pipeline più chiara, controllabile e più adatta a essere analizzata e validata in un contesto accademico.”

Slide 4 — Pipeline proposta

Si parte da un input in formato STL. Da lì vengono estratti gli spigoli rilevanti della mesh, che poi vengono campionati lungo la loro lunghezza. I punti campionati vengono proiettati secondo una configurazione di vista scelta dall’utente e successivamente sottoposti a una fase di culling. Qui il culling è semplicemente un filtraggio che serve a scartare i punti che, rispetto alla vista scelta, risultano nascosti o comunque non utili alla costruzione del pattern finale.

A partire da questi dati viene poi generato il pattern sotto forma di archi nel piano bidimensionale. Questo pattern può essere visualizzato in preview e poi esportato.
Qui il punto importante è che il risultato della pipeline non è prima di tutto un render o una semplice immagine, ma una struttura geometrica vera e propria, fatta di archi ed elementi geometrici espliciti, che può poi essere visualizzata ed esportata.
Questo rende il processo più leggibile e più facilmente controllabile.
Inoltre, ogni passaggio dipende dai precedenti: se cambia la vista, cambia la proiezione; se cambia il campionamento, cambia il numero di archi; se cambia il filtraggio di visibilità, cambia il contenuto stesso del pattern.

Slide 5 — Dal modello 3D al pattern di graffi
Questa slide riassume in modo molto sintetico la parte centrale del lavoro sviluppato nella tesi.

A sinistra si vede il modello 3D di partenza; al centro una rappresentazione semplificata degli spigoli con i punti campionati lungo di essi; a destra il pattern finale di graffi, cioè l’insieme di archi bidimensionali generati dalla pipeline.

Il software non si limita a visualizzare il modello, ma lo trasforma in una struttura 2D utilizzabile per la preview e per l’esportazione.

In altre parole, questa è la funzione principale della pipeline che ho sviluppato: prendere una mesh 3D, selezionare gli elementi geometrici più rilevanti, campionarli e convertirli in traiettorie bidimensionali leggibili.

Qui non entro nei dettagli tecnici, ma il messaggio che volevo far passare è proprio questo: dal modello tridimensionale si arriva a un pattern di graffi controllabile, che costituisce la base dello scratch hologram.

Slide 6 — Software sviluppato

L’architettura è stata organizzata separando il nucleo geometrico dall’interfaccia desktop. Questo permette di distinguere chiaramente la parte che carica il modello, genera il pattern e gestisce gli output dalla parte usata per la preview interattiva e per la regolazione dei parametri.

Dal punto di vista pratico, l’applicazione consente di caricare modelli STL, modificare la configurazione di vista, regolare densità di campionamento, modalità degli archi e filtraggio di visibilità, e osservare immediatamente l’effetto di queste scelte sul pattern.

Durante lo sviluppo è emerso abbastanza presto un limite importante: nelle prime versioni il culling non era ancora contemplato, e questo portava alla generazione di un numero enorme di archi, spesso eccessivo rispetto alla leggibilità del pattern. In pratica il sistema tendeva a sovraccaricare l’immagine con troppe traiettorie.

Per questo motivo il software non è stato pensato solo come generatore di archi, ma come strumento sperimentale in cui fosse possibile controllare meglio la visibilità, ridurre il rumore geometrico e osservare in modo più chiaro l’effetto dei parametri.

Più che un software pensato fin da subito come strumento finale di produzione, lo definirei quindi uno strumento sperimentale ben strutturato, utile per capire il comportamento della pipeline e verificare in modo abbastanza sistematico la relazione tra parametri e risultati.

todo: non parli di output considerati, così come nella slide

Slide 7 — Test e validazione

Una parte importante del lavoro è stata capire come i parametri influenzassero concretamente il pattern generato. In particolare, qui riporto un test sul cubo in cui ho variato il parametro line_resolution, cioè la densità di campionamento.

Il cubo è stato scelto come caso di studio principale perché è una geometria semplice, leggibile e utile per osservare con chiarezza come cambia il numero di archi al variare dei parametri.

Questo grafico aiuta a vedere in modo più immediato come, al crescere della line resolution, aumenti il numero di traiettorie generate. Questo rende quantitativo un effetto che altrimenti si vedrebbe solo in modo visivo.

Quello che emerge è che aumentando la densità di campionamento il pattern diventa più ricco, ma non automaticamente migliore: oltre una certa soglia aumenta anche il rischio di avere un pattern troppo denso e quindi meno leggibile.

Durante lo sviluppo ho riscontrato un problema nella preview: la proiezione risultava inizialmente troppo prospettica e, in alcune configurazioni di vista, anche un modello semplice come il cubo poteva apparire deformato in modo eccessivo. Il problema non dipendeva tanto dalla mesh in sé, quanto dal modo in cui camera e proiezione trasformavano il modello sullo schermo.

Per migliorare la situazione, la proiezione è stata resa più stabile, avvicinandola a una vista quasi ortografica, e lo zoom è stato separato dal movimento fisico della camera. In questo modo, cambiando la vista, la forma del modello resta più coerente e leggibile. Questo passaggio è stato importante perché ha mostrato che non bastava ottenere un output: bisognava anche controllare la stabilità visiva della rappresentazione.

La validazione, in questa fase, è stata soprattutto qualitativa: l’obiettivo non era ottenere una misura numerica assoluta di qualità, ma verificare se il sistema produceva risultati coerenti e interpretabili al variare dei parametri principali.

Slide 8 — Effetto dei parametri sul pattern

Questa slide mostra in modo abbastanza chiaro quanto il pattern dipenda dai parametri scelti.

Il primo parametro importante è l’angolo di vista. Cambiando la vista, cambia la proiezione della mesh e quindi cambia anche il pattern risultante. Questo è naturale, ma è importante vederlo in modo esplicito, perché significa che la vista non è un dettaglio secondario: fa parte del comportamento stesso del sistema.

Il secondo aspetto importante è la modalità di generazione degli archi. Nella pipeline ho considerato sia una modalità semicircle, più semplice e regolare, sia una modalità elliptic, che introduce traiettorie più morbide e continue.

La modalità ellittica è interessante perché in alcuni casi produce pattern visivamente più fluidi.
In ottica di fabbricazione digitale, la modalità semicircolare è in genere più vantaggiosa, perché si adatta meglio alle capacità native delle macchine a controllo numerico (CNC) di eseguire archi tramite movimenti G2 e G3, cioè comandi di interpolazione circolare, invece di approssimare tutto con segmenti lineari G1.

Questo, almeno dal punto di vista implementativo, dovrebbe portare a file più compatti, programmi più brevi e tempi di esecuzione potenzialmente inferiori.

Ho effettuato prove preliminari sul cubo, mettendo in confronto le due modalità. In termini di numero di istruzioni macchina, la modalità semicircle produce un file molto più compatto, con circa 790 istruzioni, contro circa 2700 nella modalità elliptic con ellipse ratio impostato a 0.65 (l'eccentricità di un'ellisse è un termine che misura quanto l'ellisse è schiacciata rispetto ai suoi assi).

Slide 9 — Risultati qualitativi

In questa slide mostro alcuni risultati qualitativi su casi diversi, in particolare il cubo e il d20.

Qui entra in gioco anche la parte di validazione tramite strumenti esterni. In particolare ho usato Blender e CutViewer, che hanno ruoli diversi.

Blender è stato utile come ambiente di simulazione visiva. Mi ha permesso di importare i pattern esportati come file di grafica vettoriale bidimensionale e osservarli in un contesto di rendering, quindi di verificare meglio la distribuzione delle traiettorie, la leggibilità complessiva e la percezione della struttura geometrica.

CutViewer invece è stato usato per simulare il toolpath CNC, cioè il percorso utensile che la macchina seguirebbe durante l’incisione. In questo caso la verifica non riguardava tanto l’effetto visivo finale, quanto la plausibilità della traiettoria macchina generata dalla pipeline.

Nei casi mostrati qui si vede che il sistema funziona meglio su modelli semplici o moderatamente complessi, mentre su geometrie molto più dense o ricche di dettagli il pattern tende a diventare più difficile da leggere.

Per esempio, su un oggetto architettonico molto complesso come la Mole, o comunque su modelli di quel tipo, la pipeline tenderebbe probabilmente a generare un pattern molto più affollato e difficile da interpretare localmente.

Questa parte, secondo me, è stata importante perché mi ha permesso non solo di vedere se la pipeline produceva un risultato, ma anche di confrontare diverse rappresentazioni dello stesso pattern e capire in quali casi il metodo risultava più convincente.”

Slide 10 — Setup sperimentale e prove di incisione

Dopo la parte di simulazione e validazione qualitativa, ho svolto anche alcune prove preliminari di incisione reale, e in questa slide mostro il setup sperimentale utilizzato.

La macchina impiegata è una VEVOR CNC Router a 3 assi, con mandrino da 300 watt che alla fine non ho utilizzato e controllo GRBL, equipaggiata con una punta diamantata da 60 gradi.
Punte da 90 o 120 gradi avrebbero prodotto graffi troppo larghi.

I materiali testati sono stati soprattutto acrilico nero riflettente e alluminio. Nella foto a destra si vede la macchina durante una prova di incisione.

Questi test mi sono serviti per verificare quanto il passaggio dal pattern digitale alla lavorazione reale fosse sensibile non solo alla geometria generata dal software, ma anche a fattori pratici come il materiale, la planarità della superficie e la profondità del graffio.

Le profondità usate sono state circa meno 0,02 millimetri su acrilico per cercare di essere il più precisi possibili e circa meno 0,1 su alluminio, dove la profondità del graffio incide meno.


Tornando al confronto tra la modalità a semicerchio e quella ellittica, durante le prove di incisione il tempo complessivo è risultato in realtà molto simile nei due casi.

È un risultato che richiede ulteriori verifiche, perché mostra che la dimensione del file e il numero di istruzioni non si traducono necessariamente in modo diretto nel tempo totale di lavorazione.

Secondo me questo è interessante proprio perché fa capire che non esiste una configurazione universalmente migliore: bisogna sempre valutare insieme forma del pattern, modalità geometrica e comportamento della macchina.

Slide 11 — Limiti attuali e sviluppi futuri

Questa slide raccoglie i limiti principali del lavoro e anche i possibili sviluppi futuri.

Il primo limite è che la pipeline modella bene il problema dal punto di vista geometrico, ma non descrive in modo fisicamente completo il comportamento reale della luce sul materiale inciso.

In altre parole, il sistema riesce a generare un pattern plausibile a partire dalla geometria del modello, ma non simula in modo completo tutto ciò che succede nella realtà, per esempio il comportamento ottico preciso del graffio, la riflettenza del materiale, la rugosità della superficie o le variazioni dovute alla profondità reale dell’incisione.

Il secondo limite è che, su mesh più complesse, aumentano sia la densità del pattern sia il costo computazionale, e questo può ridurre la leggibilità locale della preview.

Le prove reali di incisione che ho appena mostrato mi hanno permesso di evidenziare alcuni limiti pratici che nella tesi erano rimasti solo accennati.

Ho testato un piano in acrilico nero riflettente, ma i risultati non sono stati molto precisi perché il piano presentava micro-imperfezioni. Impostando un graffio molto leggero, circa a meno 0,02 millimetri, la CNC ha prodotto incisioni più o meno evidenti a seconda delle deformazioni locali della superficie, e questo rendeva il pattern poco leggibile nel suo insieme.

Quando invece il graffio diventava troppo profondo, emergeva un secondo problema: l’acrilico inciso tendeva a diventare bianco e opaco, perdendo riflettenza. Di conseguenza la luce non riusciva più a riflettersi nel modo desiderato, e l’effetto visivo peggiorava ulteriormente.

In questo caso, probabilmente la soluzione migliore sarebbe stata usare una punta di incisione flottante, con una molla abbastanza morbida da compensare le piccole difformità del piano e mantenere una profondità di graffio più costante anche su superfici non perfettamente uniformi.

Ho fatto però anche prove su pannelli di alluminio, e lì i risultati sono stati più soddisfacenti, perché il metallo mantiene meglio le proprietà riflettenti anche con graffi più profondi. Quindi si ottiene un compromesso migliore tra visibilità dell’incisione e mantenimento dell’effetto riflettente.

Per questo motivo, oggi direi che i risultati sono promettenti ma non conclusivi: le simulazioni, in particolare quelle in Blender, restano più leggibili e controllabili, mentre la fabbricazione reale apre problemi pratici molto interessanti, soprattutto legati al materiale e al controllo della profondità di incisione.

Ed è proprio qui che vedo gli sviluppi futuri più interessanti: migliorare il culling, rendere il campionamento più adattivo, ottimizzare il toolpath e soprattutto fare più test fisici con setup meccanici più adatti.

Slide 12 — Grazie
Per concludere, quello che considero più importante di questo lavoro è aver reso esplicito un processo che spesso rimane implicito: il passaggio da una geometria tridimensionale a un pattern inciso controllabile, osservabile e verificabile.

La tesi non pretende di risolvere in modo definitivo il problema degli scratch hologram, ma propone una base software chiara e analizzabile, su cui è possibile continuare a lavorare.

Vi ringrazio per l’attenzione e resto a disposizione per eventuali domande.