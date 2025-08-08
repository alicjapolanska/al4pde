#!/bin/bash

seeds=(5 10 15 20 25)
acquisitions=("pool_random" "lcmd" "top_k" "second_top_k")

for acq in "${acquisitions[@]}"; do
  for seed in "${seeds[@]}"; do
    sbatch \
      --export=ACQ=$acq,SEED=$seed \
      --output=out_logs/burgers_${acq}_seed${seed}.out \
      --error=err_logs/burgers_${acq}_seed${seed}.err \
      run_single.sh
  done
done