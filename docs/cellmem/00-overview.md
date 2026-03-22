# CellMem: Cellular Memory for Transformers

## Motivazione

Un paper di neuroscienze (NIMH/Scripps 2025) mostra che la formazione della
memoria nell'ippocampo avviene in modo fondamentalmente diverso da come i
transformer la implementano:

- **Multi-Synaptic Boutons (MSB)**: un assone contatta piu' neuroni (1-a-molti)
- **Anti-Hebbian**: i neuroni della memoria NON rafforzano connessioni reciproche.
  La memoria e' nella topologia (quali connessioni esistono), non nella forza dei pesi
- **Riorganizzazione strutturale**: si creano nuovi tipi di connessione (MSB),
  si riorganizzano mitocondri e strutture energetiche intracellulari
- **Interazione con astrociti**: cellule di supporto non-neuronali che modulano
  energia e comunicazione dei neuroni coinvolti nella memoria

Ref: https://www.nimh.nih.gov/news/science-updates/2025/study-illuminates-the-structural-features-of-memory-formation-at-the-cellular-and-subcellular-levels

## Idea

Aggiungere a ogni layer del transformer un modulo CellMem che:
- Ha il suo stato (M, T) che si aggiorna durante l'inferenza
- Produce 4 output (MSB broadcast) che modulano attenzione, MLP, residui
- Usa una regola anti-Hebbian (impara dalla sorpresa, non dalla ripetizione)
- Ha una maschera topologica T che evolve (crea/chiude connessioni)
- Ha un "astrocita" che modula il learning rate in base alla stabilita' del contesto

## File

- `nanochat/cellmem.py` — modulo standalone
- `nanochat/ttt.py` — TTT layer (baseline di confronto, stessa interfaccia)
- `tests/test_cellmem.py` — 15 test unitari
- `tests/test_ttt.py` — 6 test unitari
- `scripts/test2_associative_recall.py` — Test 2: associative recall (task facile)
- `scripts/test2b_accumulation_tasks.py` — Test 2b: counting + pattern shift

## Stato attuale

Vedi i log numerati (01, 02, ...) per l'evoluzione cronologica.
