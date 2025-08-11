#!/bin/bash

seeds=(6 11 16 21 26)
acquisitions=("pool_random" "lcmd" "top_k")

for acq in "${acquisitions[@]}"; do
  for seed in "${seeds[@]}"; do
    sbatch \
      --export=ACQ=$acq,SEED=$seed \
      --output=out_logs/burgers_${acq}_seed${seed}_k3.out \
      --error=err_logs/burgers_${acq}_seed${seed}_k3.err \
      run_single.sh
  done
done