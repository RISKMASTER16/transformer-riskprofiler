# Caution or Constraint?

Code for **"Caution or Constraint? Testing Explanations of a Transformer Risk Profiler Against Exact Bayesian Ground Truth"** (Anay Gupta, Imperial College London, 2026), submitted to the IAFM'26 workshop at ACM ICAIF '26.

When a household declines a financial risk, is it **cautious** (a preference an adviser should respect) or **constrained** (thin buffers that savings, insurance or credit could relieve)? UK and EU suitability rules require advisers to assess *risk tolerance* and *ability to bear losses* separately. This repo tests whether a transformer can tell the two apart from a household's choices alone, and whether explanations of its profiles can be trusted, using a simulator where the exact Bayesian answer is known.

## How it works

1. **Simulator.** A structural model of household risk-taking ([dlsriskreward](https://github.com/RISKMASTER16/dlsriskreward)) gives each simulated household two hidden traits: caution `u` (scales loss aversion) and constraint `v` (scales all eight buffers), each `~ N(0, 0.4²)` on a log scale.
2. **Data.** Each household faces 24 random gambles and answers yes/no. 30,000 training, 2,000 validation and 1,500 test households.
3. **Bayes-optimal benchmark.** For 300 test households, the exact posterior over `(u, v)` is computed on a 17×17 grid. No estimator can beat it on average.
4. **Set transformer.** Each choice is a token; a demographic token carries the prior and a CLS token is read out as a Gaussian posterior. No positional encoding, so the network is permutation-invariant. 3 layers, d=64, 4 heads, 155k parameters, trained in JAX.
5. **Baseline.** DeepSets (mean pooling, 44k parameters) isolates what attention contributes.

## Results

All numbers come from `results/` (produced by `run_all.sh`); networks are trained with three seeds.

| After 24 choices, 300 households | RMSE caution | RMSE constraint | Distance to Bayes-optimal |
|---|---|---|---|
| Demographics only | 0.412 | 0.419 | |
| DeepSets | 0.350 ± 0.001 | 0.384 ± 0.001 | 0.064, 0.092 |
| **Set transformer** | **0.348 ± 0.005** | **0.385 ± 0.000** | **0.060, 0.078** |
| Bayes-optimal | 0.341 | 0.371 | 0 |

**Explanation faithfulness** (Spearman correlation with the exact information each choice adds to the posterior):

| Method | Set transformer | DeepSets |
|---|---|---|
| Attention, last layer | 0.43 ± 0.06 | |
| Attention rollout | 0.65 ± 0.06 | |
| Integrated gradients | 0.48 ± 0.14 | 0.56 ± 0.07 |
| Occlusion | **0.76 ± 0.01** | **0.79 ± 0.00** |

- Raw attention is a weak explanation in every layer; integrated gradients is unstable across equally accurate retrained models; occlusion is the most faithful and the most stable.
- 24 choices from a low-income household carry about 3× the information in the prior; from a high-income household, about a tenth. Constraint is legible only where it binds.

The data, Bayes benchmark and DeepSets results reproduce exactly. Transformer results can vary in the third decimal between runs, because JAX on CPU is not bit-for-bit deterministic.

## Reproduce

The working paper is in `paper/`.


```bash
pip install -r requirements.txt
./run_all.sh        # ~70 minutes on one CPU core
```

| Script | Output |
|---|---|
| `gen_data.py` | `data/{train,val,test}.npz` (simulated households) |
| `bayes.py` | `data/bayes.npz`, `data/bayes_ll.npy` (exact grid posterior, per-choice likelihoods) |
| `train.py` | `checkpoints/{tx,ds}_{0,1,2}.pkl` (included, so training can be skipped) |
| `evaluate.py`, `evaluate_seeds.py` | `results/results_tx.json`, `results/seeds.json` |
| `attention.py`, `attn_layers.py` | `results/attention.json`, `results/attn_layers.json` |
| `faithfulness.py` | `results/faithfulness.json` |
| `example_household.py` | `results/example.json`, `figures/fig_example.pdf` |
| `figures_iafm.py` | `figures/fig_rmse.pdf`, `figures/fig_legibility_faith.pdf` |

The simulator files (`dls_core.py`, `dls_study.py`, `dls_analysis.py`, `calib.json`) are copied from [dlsriskreward](https://github.com/RISKMASTER16/dlsriskreward) so this repo runs on its own. `sim.py` wraps them with the two hidden traits.

## Limitations

All households are simulated; choice noise is assumed rather than estimated; the Bayes benchmark covers 300 households; integrated gradients was tested with one baseline.
