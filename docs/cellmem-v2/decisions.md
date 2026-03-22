# CellMem v2 — Design Decisions

## Decision 1: Memory-Transformer Interaction

**Date:** 2026-03-22
**Question:** Come i vettori di memoria interagiscono con il forward pass del transformer?

### Opzioni considerate

**A. Residual Injection** — somma dei vettori di memoria nel residual stream con gate scalare.
- Pro: semplicissimo, zero-init preserva il modello, compatibile torch.compile
- Contro: il modello non è addestrato a usare info iniettata lì, iniezione "brutale"

**B. Attention-based Read** — memorie come key/value extra nella self-attention.
- Pro: meccanismo naturale (il transformer sa già leggere via attention), alta capacità informativa, supportato in letteratura (Memorizing Transformers, inf-former)
- Contro: modifica Flash Attention (K/V extra), costo O(T*K) aggiuntivo, complessità KV cache

**C. Gate Modulation** — memorie modulano i gate del transformer (resid_lambdas, x0_lambdas).
- Pro: non modifica il data flow, overhead minimo
- Contro: bandwidth informativa bassissima (pochi scalari), non fedele al paper, più "mood setter" che memoria

### Scelta: B — Attention-based Read

**Motivazione:**
1. Meccanismo più fedele al paper NIMH/Scripps 2025 — la memoria modifica *cosa il modello vede*, non solo *come processa*
2. Bandwidth informativa più alta — K vettori da 768 dim
3. Il transformer sa già usare l'attention per leggere informazione
4. Le memorie sono pre-pended come extra K/V, Flash Attention le gestisce nativamente

---

## Decision 2: Memory Write Strategy

**Date:** 2026-03-22
**Question:** Cosa decide quando un vettore viene scritto/aggiornato nella memoria durante l'inferenza?

### Opzioni considerate

**A. Surprise-gated (per-token)** — scrivi una memoria solo quando il modello è sorpreso (alta loss/entropia). Anti-Hebbian puro: memorizzi ciò che non sai predire.
- Pro: traduzione diretta del principio anti-Hebbian del paper
- Contro: costoso calcolare surprise per singolo token durante l'inferenza

**B. Every-token with decay** — ogni token produce un candidato, le vecchie memorie decadono. Solo le rinforzate sopravvivono.
- Pro: semplice, nessuna soglia da tuning
- Contro: più Hebbian che anti-Hebbian (sopravvive ciò che è frequente, non ciò che è sorprendente). Non fedele al paper.

**C. Chunk-based surprise** — alla fine di ogni chunk (es. 64 token) calcoli un vettore riassuntivo e scrivi in memoria se l'errore medio del chunk è alto.
- Pro: approssimazione pratica di A, molto più efficiente, cattura lo stesso segnale
- Contro: granularità più grossa, potrebbe perdere eventi sorprendenti puntuali

### Scelta: Implementare sia A che C, valutare con eval

**Motivazione:**
- Entrambe sono fedeli al principio anti-Hebbian del paper (scrivi quando sei sorpreso)
- A è la versione "pura", C è l'approssimazione efficiente
- Approccio scientifico: implementare entrambe e confrontare empiricamente con test/eval
- Se C performa comparabilmente a A, preferire C per efficienza

---

## Decision 3: Memory Capacity

**Date:** 2026-03-22
**Question:** Quanti vettori di memoria (K)?

### Analisi

Nanochat ha 286M parametri (0.286B), scala GPT-2. Con d_model=768, seq_len=2048, 6 heads:

| K     | % modello | Costo attn | Note                                    |
|-------|-----------|------------|-----------------------------------------|
| 64    | 0.017%    | +3.1%      | Conservativo, buon punto di partenza    |
| 128   | 0.034%    | +6.2%      | Sweet spot per 286M                     |
| 256   | 0.069%    | +12.5%     | Aggressivo ma fattibile                 |
| 512   | 0.137%    | +25.0%     | Il modello potrebbe non saper usarle    |
| 1024  | 0.275%    | +50.0%     | Troppo per questa scala                 |

Il bottleneck è la capacità del modello di *leggere* le memorie (6 heads, 128 dim), non la quantità.

### Scelta: K=64, scalabile a 128/256

**Motivazione:**
- Trascurabile in parametri (0.017%) e costo computazionale (+3.1%)
- Con 6 attention heads, 64 memorie sono un ratio ragionevole
- Possiamo aumentare se il modello dimostra di saperle usare

---

## Decision 4: Memory Layer Placement

**Date:** 2026-03-22
**Question:** A quali layer del transformer attacchiamo la lettura delle memorie?

### Opzioni considerate

**A. Solo ultimi layer (9-11)** — layer profondi con rappresentazioni più semantiche.
- Pro: le rappresentazioni sono più astratte/concettuali, adatte a "ricordare"
- Contro: potrebbe essere troppo tardi nel processing per influenzare il risultato

**B. Solo layer mediano (6)** — singolo punto di iniezione.
- Pro: semplice da debuggare, un solo overhead
- Contro: potrebbe non essere il layer giusto, arbitrario

**C. Tutti i layer (0-11)** — ogni layer può leggere le memorie.
- Pro: massima flessibilità, il modello decide dove usare le memorie
- Contro: 12x il costo attention delle memorie

### Scelta: Implementare tutte e tre come configurazione, valutare con eval

**Motivazione:**
- Approccio scientifico: il layer placement ottimale non è ovvio a priori
- Il costo di implementazione è minimo (un parametro di configurazione `cellmem_layers`)
- Eval comparativo rivelerà dove il modello beneficia di più delle memorie

---

## Decision 5: Memory Persistence

**Date:** 2026-03-22
**Question:** Come persistiamo le memorie tra invocazioni?

### Opzioni considerate

**A. Save/load su disco** — salvi i K vettori + metadata alla fine, ricarichi al prossimo avvio con decay.
- Pro: semplice
- Contro: nessun recovery se le memorie si corrompono

**B. Append-only log** — ogni invocazione appende nuove memorie. Al load ricarichi le ultime K.
- Pro: hai uno storico completo
- Contro: il file cresce indefinitamente

**C. Snapshot + decay** — save/load come A, più snapshot periodici per rollback.
- Pro: recovery possibile, puoi tornare a uno stato precedente se le memorie degradano
- Contro: leggermente più complesso

### Scelta: C — Snapshot + decay

**Motivazione:**
- Le memorie sono state che evolvono durante l'inferenza — se qualcosa va storto servono rollback
- Lo snapshot periodico è cheap (64 vettori * 768 dim * 2 bytes bf16 = 96KB per snapshot)
- Il decay al load simula il "dimenticare" naturale tra sessioni

---

## Decision 6: Learnable Parameters

**Date:** 2026-03-22
**Question:** CellMem v2 ha parametri learnable o è puramente algoritmico?

### Opzioni considerate

**A. Puramente algoritmico** — zero parametri extra, riusa W_k/W_v del transformer. Il modello non sa che le memorie esistono.
- Pro: zero training, massima interpretabilità
- Contro: il modello non ha mai visto memorie durante training, nessun modo di spegnerle

**B. Proiezioni learnable dedicate** — W_k_mem, W_v_mem separate (~196K params/layer, ~2.4M totali).
- Pro: proiezioni ottimali per memorie
- Contro: serve ri-training, complicazioni optimizer (lezione da v1), instabilità

**C. Ibrido** — riusa W_k/W_v esistenti + gate scalare α per layer (init=0).
- Pro: 12 params totali, α=0 = modello identico al baseline, nessun rischio degradazione, segnale diagnostico (se α resta ~0, le memorie non servono a quel layer)
- Contro: gate globale per layer (non per-head), proiezioni K/V devono funzionare anche per memorie

### Scelta: C — Ibrido (W_k/W_v riusati + gate α)

**Motivazione:**
- Combina il meglio di A (riuso pesi) e B (gate learnable)
- Rischio zero di degradazione all'init (α=0)
- Pochissimi parametri → nessun problema optimizer, nessuna instabilità
- Se α resta ~0 dopo training → diagnostica gratis su utilità delle memorie per layer

---

## Decision 7: Surprise Score Calculation

**Date:** 2026-03-22
**Question:** Come calcoliamo la "sorpresa" per decidere se scrivere in memoria?

### Opzioni considerate

**A. Cross-entropy loss** — -log(P(token_reale)). Misura prediction error.
- Pro: traduzione diretta del principio anti-Hebbian (errore di predizione), cattura il caso "sicuro ma sbagliato"
- Contro: richiede il token reale (disponibile in inference autogressiva)

**B. Entropia della distribuzione** — misura incertezza del modello.
- Pro: non serve il token reale
- Contro: NON cattura il caso cruciale "sicuro ma sbagliato" (bassa entropia, alto errore)

### Scelta: A — Cross-entropy loss

**Motivazione:**
- Il paper parla di prediction error, non di incertezza
- L'entropia misura "quanto sei confuso", la loss misura "quanto ti sei sbagliato" — il paper vuole il secondo
- Il caso più informativo (alta confidenza, predizione sbagliata) è catturato solo da A
- Durante l'inferenza autogressiva il token reale è sempre disponibile
- Serve una soglia o media mobile per filtrare rumore (token intrinsecamente imprevedibili come nomi, numeri)
