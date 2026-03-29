# CellMem v4 — Vision Document

> **Status:** Pre-design. Blocked by v3 architectural injection failure.
> **Date:** 2026-03-29
> **Prerequisite:** Risolvere il problema fondamentale: memoria dentro l'architettura con backbone frozen/low-cost fine-tuning.

## Goal

Memoria episodica persistente **dentro l'architettura** del modello (non text prefix/RAG), con:
1. **Albero semantico dei ricordi** — B-tree semantico per organizzazione gerarchica
2. **Decay biologico** — ispirato da paper di neuroscienze sulla memoria umana
3. **Memorizzazione automatica** — niente /mem, tutto implicito dalla conversazione
4. **Persistenza cross-sessione** — la memoria sopravvive ai restart

## Perche v3 non basta

v3 ha dimostrato che:
- **Text prefix injection funziona** (85% recall) ma e RAG, non memoria architetturale
- **KV injection fallisce** (0% recall) — backbone frozen ignora K/V che non ha generato
- **HS injection e instabile** (80% eval ma loop in chat) — LoRA rank 16 su 200 samples non generalizza
- **MemoryLLM richiede 8xA100 x 3 giorni** — troppo costoso per il nostro setup

### Lezione chiave da v3

> Un backbone frozen **non puo leggere** informazioni iniettate nell'architettura senza fine-tuning significativo. Il modello non ha mai visto K/V o hidden states "alieni" durante il pre-training.

## Architettura proposta

### 1. Albero semantico (B-tree dei ricordi)

```
Root
  |-- Persone
  |     |-- Giuseppe (nome, ruolo, preferenze)
  |     |-- Mario (colleghi, progetti)
  |-- Progetti
  |     |-- CellMem (stato, decisioni, blocchi)
  |     |-- Fairmind (architettura, deploy)
  |-- Fatti
        |-- Tecnici (librerie, config, fix)
        |-- Personali (compleanni, hobby)
```

- Ogni nodo ha un **router key** (embedding) per retrieval semantico
- I figli specializzano il padre (come un B-tree ma semantico)
- Inserimento: router decide dove nell'albero inserire un nuovo ricordo
- Retrieval: tree traversal top-down con early stopping

### 2. Decay biologico

Ispirato dalla memoria umana (paper di neuroscienze da studiare):
- **Working memory** (secondi): buffer corrente della conversazione
- **Short-term memory** (minuti-ore): ricordi recenti, alta risoluzione
- **Long-term memory** (giorni+): ricordi consolidati, compressi
- **Consolidamento**: working -> short-term -> long-term durante "sleep" (fine sessione)
- **Forgetting curve**: Ebbinghaus decay, rinforzato da accesso ripetuto

### 3. Injection architetturale

Approcci da investigare per v4:

| Approccio | Costo | Rischio | Note |
|---|---|---|---|
| **LoRA fine-tuning massivo** | Medio (1xA100, giorni) | Overfit | Piu dati e diversi, rank piu alto |
| **Adapter layers** | Basso | Capacita limitata | Layer addizionali tra decoder layers |
| **Prefix tuning** | Basso | Limitato | Learned soft prompts, piu architetturale del testo |
| **Full fine-tuning su C4+memory** | Alto (8xA100, settimana) | MemoryLLM approach | Provato e funziona, ma costoso |
| **Distillation** | Medio | Complessita | Modello teacher con memoria -> student |

### Decisione aperta

Prima di procedere con v4, serve rispondere a:

1. **Budget compute**: quanto possiamo spendere per il fine-tuning? (1xL40S spot? 8xA100?)
2. **Approccio injection**: LoRA massivo vs adapter vs prefix tuning vs full fine-tuning?
3. **Paper biologici**: quali paper di neuroscienze sulla memoria/decay studiare?
4. **Albero vs flat**: l'albero semantico e necessario subito o puo venire dopo?

## Paper da studiare per v4

### Memoria biologica e decay
- Ebbinghaus forgetting curve
- Complementary Learning Systems (McClelland et al.)
- Memory consolidation during sleep (paper da identificare)
- Hippocampal replay (paper da identificare)

### Injection architetturale
- **MemoryLLM** (ICML 2024) — full fine-tuning approach che funziona
- **Prefix Tuning** (Li & Liang, 2021) — learned soft prompts
- **Adapter Layers** (Houlsby et al., 2019) — bottleneck adapters
- **LoRA** (Hu et al., 2021) — low-rank, gia usato in v3

## Dipendenze da v3

Componenti v3 da riusare in v4:
- `MemoryRouter` — funziona perfettamente (recall@4=100%)
- `MemoryStore` — struttura di storage, save/load, decay
- `KVInterceptor` — capture hooks
- `SurpriseCalculator` — BTSP surprise-gated writes
- Training pipeline (SkyPilot/Vast.ai)

## File map (preliminare)

| File | Contenuto |
|---|---|
| `nanochat/cellmem_v4.py` | SemanticTree, BiologicalDecay, MemoryConsolidator |
| `nanochat/cellmem_v3.py` | MemoryRouter, MemoryStore (riusati) |
| `scripts/train_cellmem_v4.py` | Fine-tuning per injection architetturale |
| `docs/cellmem-v4/` | Design docs, literature review |
