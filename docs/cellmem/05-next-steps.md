# 05 — Next Steps

## Problema aperto: loop sequenziale

CellMem processa token uno alla volta (M_t dipende da M_{t-1}).
Su 2048 token per layer, questo e' troppo lento per il training.

### Opzione A: Chunked update (consigliata)

Aggiornare M ogni K token (es. K=32) usando la media del chunk:
- 64 update invece di 2048 -> 32x piu' veloce
- Perde granularita' ma mantiene accumulazione
- Implementazione semplice

### Opzione B: CellMem su pochi layer

Solo 2-3 layer su 12 hanno CellMem. Gli altri restano standard (paralleli).
Riduce l'overhead al 17-25% del modello.

### Opzione C: Parallel scan (avanzato)

Riscrivere l'update di M come ricorrenza lineare parallelizzabile
con prefix-sum (come Mamba per le SSM).
Richiede che l'update sia lineare in M.

## Piano

```
Step 1: Implementare chunked CellMem (opzione A)
        Test su counting per verificare che funzioni ancora

Step 2: Integrare in nanochat/gpt.py
        CellMem su 2 layer (allineati ai KV heads)
        Training corto d12: 500 step su 1 shard
        Misurare: val loss, CORE eval, throughput

Step 3: Solo se Step 2 positivo -> training completo d12
        Dataset: ClimbMix-400B, 8 H100, ~2 ore stimate
```

## Domande aperte

1. Su quali layer mettere CellMem? Layer bassi (pattern locali)
   o alti (semantica)?

2. Il chunked update perde troppa granularita'? K=32 vs K=64 vs K=128?

3. Il counting e' rappresentativo dei sub-task utili nel language modeling?
   Servono altri task sintetici (tracking, statistica, etc.)?

4. Come interagisce CellMem con le sliding window attention di nanochat?
   Le finestre piccole potrebbero beneficiare di piu' dalla memoria.
