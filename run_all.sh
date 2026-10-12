#!/bin/sh
# Reproduces every number and figure in the paper (~70 min on one CPU core).
set -e
python3 gen_data.py train 30000 1
python3 gen_data.py val 2000 2
python3 gen_data.py test 1500 3
python3 bayes.py                       # exact grid posterior + per-choice likelihoods (~4 min)
for s in 0 1 2; do
  python3 train.py tx 4000 $s          # set transformer (~11 min each)
  python3 train.py ds 4000 $s          # DeepSets baseline (~1 min each)
done
python3 evaluate.py                    # seed 0, all K -> results/results_tx.json
python3 evaluate_seeds.py              # seeds 0-2     -> results/seeds.json
python3 attention.py                   # Fisher information / legibility -> results/attention.json
python3 attn_layers.py                 # attention by layer and band -> results/attn_layers.json
python3 faithfulness.py                # explanations vs exact information -> results/faithfulness.json
python3 example_household.py           # worked example -> results/example.json, figures/fig_example.pdf
python3 figures_iafm.py                # figures/fig_rmse.pdf, figures/fig_legibility_faith.pdf
