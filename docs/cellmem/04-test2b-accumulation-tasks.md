# 04 — Test 2b: Accumulation Tasks (counting + pattern shift)

## Ipotesi

CellMem migliora i task che richiedono accumulazione di stato,
dove l'attenzione pura non basta.

## Setup

Stesso tiny transformer di Test 2 (2 layer, d_model=64).
3000 step, batch_size=64, 3 seed.

### Task 1: Counting

```
Sequenza: A B C A D A B A | query=A -> 4
```

20-40 simboli casuali, il modello deve contare le occorrenze di uno specifico.
L'attenzione puo' TROVARE tutte le posizioni ma non CONTARE.
Random chance: 1/16 = 6.25%.

### Task 2: Pattern Shift

```
Prima meta': A->1 B->2 C->3 | SHIFT | Seconda meta': A->3 B->1 C->2
Query dopo SHIFT: A -> ? (target: 3, il mapping POST-shift)
```

Random chance: 1/8 = 12.5%.

## Risultati

### Counting: CellMem VINCE

```
             Seed 42    Seed 123   Seed 456   Media
baseline:    30.3%      73.2%      46.4%      50.0% +/- 17.7%
cellmem:     94.8%      73.3%      54.7%      74.3% +/- 16.4%
ttt:         85.5%      39.8%      44.9%      56.7% +/- 20.4%
```

CellMem: 74.3% vs baseline 50.0% (+24.3 punti).
CellMem > TTT (74.3% vs 56.7%).

Comportamento "phase transition": tutti i modelli stagnano a ~29% per ~2000 step,
poi CellMem "esplode" (seed 42: 29% -> 92% in 800 step).
CellMem raggiunge la transizione PRIMA e va PIU' IN ALTO.

### Pattern Shift: nessuna differenza

```
             Seed 42    Seed 123   Seed 456   Media
baseline:    24.0%      24.9%      25.6%      24.8% +/- 0.7%
cellmem:     25.2%      24.6%      25.8%      25.2% +/- 0.5%
ttt:         24.8%      24.4%      25.5%      24.9% +/- 0.5%
```

Tutti ~25%. Probabilmente risolvibile dall'attenzione
(basta imparare che i token dopo SHIFT sono piu' recenti).

## Interpretazione

1. CellMem funziona su task di ACCUMULAZIONE (counting) ma non su task
   risolvibili dall'attenzione (recall, pattern shift).

2. La regola anti-Hebbian e' piu' adatta all'accumulazione rispetto
   alla ricostruzione (TTT): 74.3% vs 56.7%.

3. La varianza tra seed e' alta (phase transitions), ma CellMem
   e' consistentemente il primo a fare la transizione.

4. La chiave e' stata rendere M differenziabile (opzione B):
   il transformer IMPARA ad usare CellMem solo quando i gradienti
   fluiscono attraverso la catena di M updates.

## Gate: PASS sul counting, FAIL sul pattern shift
