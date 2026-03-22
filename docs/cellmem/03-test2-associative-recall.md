# 03 — Test 2: Associative Recall (tiny transformer)

## Ipotesi

CellMem migliora l'in-context learning su un task di associative recall.

## Setup

- Tiny transformer: 2 layer, d_model=64, 2 heads, ~107K parametri
- Task: "A=3 B=7 C=2 | query=B -> ?" (target: 7)
- Ogni batch ha associazioni NUOVE -> impossibile memorizzare
- 3 varianti: baseline, cellmem (+10K params), ttt (+10K params)
- 2000 step, 3 seed

## Risultati

```
baseline: 25.3% +/- 0.1%
cellmem:  25.2% +/- 0.3%
ttt:      25.3% +/- 0.3%
```

NESSUNA differenza. Random chance = 6.25%, tutti a ~25%.

## Diagnosi

1. L'attenzione risolve GIA' questo task perfettamente (induction heads).
2. CellMem non aggiunge nulla quando l'attenzione basta.
3. Con M detached (prima run): tutte le ablazioni identiche (CellMem invisibile).
4. Con M differenziabile (seconda run): ancora nessun miglioramento.

## Lezione

L'associative recall e' il task SBAGLIATO per CellMem.
L'attenzione e' lo strumento nativo per "trova X nel contesto e copia Y".
CellMem serve per task che richiedono ACCUMULAZIONE DI STATO.

## Gate: FAIL (su questo task) -> necessario cambiare task
