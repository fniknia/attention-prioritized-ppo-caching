# Attention-Enhanced Prioritized PPO for Adaptive Edge Caching

This repository contains the paper and a rewritten simulation of the PPO-with-prioritized-replay caching agent.

- `paper/` — the manuscript (Niknia, Wang, et al.), *Attention-Enhanced Prioritized Proximal Policy Optimization for Adaptive Edge Caching*, IEEE Transactions on Vehicular Technology, 2025. DOI: 10.1109/TVT.2025.3552084.
- `code/ppo_per_caching.py` — rewritten simulation (replaces the original notebook `CL_PPO_PER.ipynb`).
- `code/file_types.csv` — per-type lifetime, importance and size, extracted from the notebook.

## Workload

- **Requests:** file IDs 1–50 are drawn from a Zipf distribution with α = 1 (p_k ∝ 1/k).
- **Arrivals:** a Poisson process with mean interarrival time 0.2 (exponential interarrivals, cumulative sum gives arrival times).
- **File attributes:** each file type has a fixed lifetime, importance and size, read from `file_types.csv`.

## Running

```bash
pip install -r requirements.txt
cd code
python ppo_per_caching.py --steps 10000 --seed 0 --out ../results
```

Output is written to `results/ppo_per_seed<seed>.npz` with per-step rewards, hit indicators (after the 5,000-request warm-up), and the final hit ratio.

## Changes from the notebook

- The large per-request vectors for lifetime, importance, size, request stream and interarrivals were removed and replaced with the generators above.
- Interarrivals are converted to arrival times with a cumulative sum. The notebook used the interarrival values directly as absolute times.
- Cache-state presence and utility are indexed by file type. The notebook indexed them by cache slot.
- The critic loss applies the importance-sampling weights per sample. The notebook multiplied them by a scalar sum.
- Returns and values have matching shapes, avoiding a (B, B) broadcast.
- Removed: the HDF5 demo-copy path (no demonstrations are used), the L2 term (no regularizers are defined), and the waiting-time bookkeeping.

## Known discrepancy with the paper

The notebook does **not** implement the attention-based priority described in the paper. It inserts each transition with the current maximum priority and updates priorities from absolute TD errors, as in standard prioritized replay. The paper's text describes computing priorities against the current state with attention over the buffer. The two versions differ, and the code reflects the notebook's stored-at-insertion version. Results from this script should not be attributed to the attention mechanism until that is resolved.
