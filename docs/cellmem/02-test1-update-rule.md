# 02 — Test 1: Update Rule (isolato, CPU)

## Ipotesi

La regola anti-Hebbian (dM = alpha * outer(error, z)) cattura pattern nuovi
meglio della regola Hebbian (dM = alpha * outer(z, z)).

## Setup

Solo la matrice M, sequenze sintetiche ortogonali, nessun transformer.
Config: d_model=16, d_cell=8, n_cells=2.

## Risultati

15/15 test passano:

| Test                          | Risultato                                   |
|-------------------------------|---------------------------------------------|
| novel_pattern_recall          | M accumula dopo pattern nuovi               |
| familiar_pattern_stability    | Novelty cala su pattern ripetuti             |
| hebbian_comparison            | Anti-Hebbian predice C meglio di Hebbian     |
| topology_opens_for_novelty    | T cambia dopo input diversi                  |
| topology_closes_for_familiar  | T decade con tau su pattern ripetuti         |
| topology_sparsifies           | T sviluppa struttura non-triviale           |
| high_variance_increases_lr    | Astrocita: sigma alto -> alpha alto          |
| low_variance_decreases_lr     | Astrocita: sigma basso -> alpha basso        |
| output_shapes                 | 4 output MSB con shape corretta              |
| four_outputs_are_different    | Le 4 proiezioni W_msb producono output diversi |
| reset_clears_state            | Reset riporta M a 0.01*I, T a ones           |
| state_accumulates             | M non-zero dopo un forward                   |
| gradient_flows                | Gradienti su W_in e W_msb                    |
| gradient_flows_through_M_chain| Gradienti fluiscono attraverso la catena di M |
| deterministic                 | Stessa sequenza + seed -> stesso stato        |

## Gate: PASS

L'update rule funziona come previsto in isolamento.

## Bug trovati e corretti durante il test

1. **M inizializzato a zero**: alpha_eff partiva da ~0 perché sigma=0.
   Fix: M init a 0.01*I, astro_mu e astro_sigma init a 1.0.

2. **W_in e W_msb con init 0.02 std**: troppo piccolo, catena di moltiplicazioni
   produceva output ~0.00002, sigmoid(~0) = 0.5 costante.
   Fix: Xavier init (1/sqrt(d)) per entrambi.

3. **M detached**: il transformer non poteva imparare ad usare CellMem.
   Fix: rimosso .detach() dall'update di M (opzione B).
   T resta detached (maschera strutturale).
