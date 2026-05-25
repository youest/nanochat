# MemoryBridge — esperimento di injection latente (RecursiveMAS-style)

**Data:** 2026-05-25
**Branch:** cellmem/v2
**Backbone:** `Qwen/Qwen3.5-4B` (ibrido Gated DeltaNet + Gated Attention, hidden=2560, 32 layer, thinking mode default ON)

## Tesi da validare

Un **MemoryBridge** — memoria → backbone frozen → hidden states → proiezione addestrata → soft-prefix a livello di **input embedding** — **pareggia il text prefix sul recall single-fact**, senza i loop generativi dell'HS injection.

Questo è il risultato che nessuno dei 7 tentativi architetturali ha ottenuto: un backbone **frozen** che legge un prefisso latente addestrato e risponde come col testo. È *quello* il finding.

(Stress / compressione / beat-text sono downstream: se il single-fact fallisce, sono rumore su un sistema rotto; se passa, diventano bonus esplorativi non bloccanti.)

Origine: arXiv 2604.25917 (RecursiveMAS) — backbone frozen + bridge leggero (0.31% params) che trasferisce stati latenti a livello di embedding, con loss che li tiene *in-distribution*.

## Perché potrebbe funzionare dove i 7 tentativi sono falliti

| Tentativo | Punto injection | Esito |
|---|---|---|
| KV injection + Parallel RoPE | KV cache | 0% |
| HS injection + LoRA | tra decoder layer | 80% eval → loop |
| **MemoryBridge (questo)** | **input embedding** | **da testare** |

Injection più "gentile" (vicino alla distribuzione degli embedding) + bridge che proietta esplicitamente nello spazio embedding del modello.

## Architettura

`MemoryBridge` (backbone **frozen**, solo bridge trainabile):

1. **Encode memoria:** tokenizza `memory`, appende **K=16 gist tokens** (embedding trainabili). Forward sul backbone frozen con `output_hidden_states=True`. Prende gli hidden dell'ultimo layer alle K posizioni gist → `H_mem ∈ [K, d]`.
2. **Proietta** (forma RecursiveMAS outer): `R(h) = W3·h + W2·σ(W1·h)` (σ=GELU) → prefix latente `L ∈ [K, d]`.
3. **Inietta:** `inputs_embeds = [L ; embed(query)]` → genera.

Trainabili: gist tokens (`[K, d]`) + W1, W2, W3. ~pochi M params. **No zero-init** su nessuna matrice (chain rule).

Agnostico all'architettura: usa solo `get_input_embeddings()` + `inputs_embeds` + `output_hidden_states`, quindi vale anche sui layer DeltaNet.

## Loss

- **Primaria:** cross-entropy sui token della `answer`, dato `[L ; query]` (mascherando query/prefix nel target). Test diretto: il modello sa rispondere dal latente?
- **Fallback** (se la primaria non converge): aggiungere ricostruzione autoencoder ICAE-style (il latente deve poter ricostruire la `memory`), come ancora *in-distribution*.

## Eval

Riuso `eval_generation` (substring match, greedy) + aggiungo `mode='embed_prefix'` (inietta il soft prefix via `inputs_embeds`). Confronto diretto **single-fact**: `embed_prefix` vs `text_only` sullo stesso dataset/modello.

**Esplorativo (solo se il single-fact passa):** K-sweep ∈ {4, 8, 16, 32} sulla stessa eval single-fact. Se K=4 funziona → storia di compressione gratis; se serve K≥16 → niente claim di compressione, ma il mechanism test resta un risultato.

NB: lo **stress eval** (N memorie distrattori) **non è un criterio di successo**: su un modello a 262k di contesto il text_only non degrada a nessun N costruibile dal dataset (48 item ≈ 1k token) → claim non falsificabile. Tenibile come curiosità, non come obiettivo.

Tutte le generazioni con `enable_thinking=False` (altrimenti il thinking mode rompe il match a 30 token).

## Criteri di successo

1. **Plumbing:** unit test del bridge passano (shape, gradiente, no zero-init); eval mode gira.
2. **Single-fact (la tesi):** `embed_prefix` pareggia `text_only` (≥75%, idealmente ≥80%; batte kv_only=0% e hs-loop).

Se (2) fallisce → injection latente non leggibile dal frozen backbone (come i 7 tentativi). Se passa → primo successo di lettura latente su backbone frozen; K-sweep e stress diventano esplorazioni opzionali.

## Esecuzione

Diretto su GPU (Vast, `gpus:` nel YAML). Modello piccolo → GPU 24-48GB. TDD unit test in locale (CPU, config sintetica, no download).

**Prima del training run, ~5 min di verifica di load sul box** (non è un "tier", è caricare il modello giusto): load `Qwen3.5-4B`, un forward, `get_input_embeddings()` → shape `[V, 2560]`, `output_hidden_states=True` funziona, `apply_chat_template(..., enable_thinking=False)` fa la cosa giusta. Solo dopo, training.

Non distruggere il cluster a fine run. Push su remote `fork`.

## Rischi noti

- Qwen3.5 è post-cutoff: serve `transformers` recente, possibile `trust_remote_code`. Verificare classe di load (CausalLM vs VLM) sul box.
- Thinking mode default ON → forzare off.
- Compressione K=16 potrebbe perdere info su memorie lunghe → al limite alzare K.
