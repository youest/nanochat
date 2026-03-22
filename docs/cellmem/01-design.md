# 01 — Design architetturale

## Principi biologici -> decisioni di design

Fonte: NIMH/Scripps 2025 (formazione della memoria nell'ippocampo del topo).

| Principio dal paper                 | Decisione architetturale                              |
|-------------------------------------|-------------------------------------------------------|
| MSB (1 assone -> N neuroni)         | 1 CellMem produce 4 output (g_attn, g_mlp, r_add, x0_mod) |
| Anti-Hebbian (topologia > pesi)     | Update rule error-driven: dM = alpha * (error x z)    |
| Riorganizzazione strutturale        | Maschera topologica T che evolve a runtime             |
| Interazione neuroni-astrociti       | Running stats (mu, sigma) modulano il learning rate    |

Nota: il paper mostra che gli astrociti aumentano l'interazione con i neuroni
della memoria. Il nostro "astrocita" e' un'interpretazione libera: modula il
learning rate in base alla stabilita' del contesto (alta varianza -> impara di piu').
Non e' un mapping diretto dal paper.

## CellMem per cella

Parametri fissi (trainati con backprop):
- W_in: (d_slice -> d_cell) proiezione input
- W_msb: 4x (d_cell -> d_slice) proiezioni MSB
- alpha_base, gamma, tau: scalari per-cella

Stato runtime (aggiornato durante inferenza, reset a inizio sequenza):
- M: (d_cell x d_cell) matrice di memoria, init 0.01*I
- T: (d_cell x d_cell) maschera topologica, init ones
- astro_mu, astro_sigma: statistiche astrocita, init ones

## Forward pass

```
z = x_slice @ W_in                         # proiezione in cell space
z_pred = (M * T) @ z                       # predizione della memoria
error = z - z_pred                          # segnale di novita'

# Update anti-Hebbian (DIFFERENZIABILE — gradienti passano attraverso M)
dM = alpha_eff * outer(error, z)
M = M + dM

# Update topologia (detached — maschera strutturale)
dT = gamma * (outer(|error|, |z|) - tau * T)
T = clamp(T + dT, 0, 1)

# Astrocita modula alpha
alpha_eff = alpha_base * (sigma / (mu + eps))

# MSB fan-out: 4 output verso target diversi
mem_out = (M * T) @ z
g_attn  = mem_out @ W_msb[0]    # gate sull'attenzione
g_mlp   = mem_out @ W_msb[1]    # gate sul MLP
r_add   = mem_out @ W_msb[2]    # contributo additivo al residuo
x0_mod  = mem_out @ W_msb[3]    # modulazione del blending x0
```

## Integrazione nel blocco transformer

```
x -> CellMem -> (g_attn, g_mlp, r_add, x0_mod)
        |
x -> LN -> Attention -> * sigmoid(g_attn) -> + residual + r_add
        |
x -> LN -> MLP -> * sigmoid(g_mlp) -> + residual -> x_out
```

## Overhead stimato (nanochat d12)

- Parametri: +1.5M su ~85M (+1.8%)
- Stato runtime: ~384KB (trascurabile vs KV cache)
- FLOP per token: +5% circa
- Problema aperto: il loop sequenziale su 2048 token
