#!/bin/bash

set -e

# Seeds and acquisition strategies
seeds=(6 11 16 21 26)
acquisitions=("pool_random" "lcmd" "top_k")

# Output log directories
out_dir="out_logs"
err_dir="err_logs"

# Associative array to collect run IDs
declare -A acq_to_ids

echo "----------------------------------------"
echo "Checking for successful runs and syncing"
echo "----------------------------------------"

for acq in "${acquisitions[@]}"; do
    echo "==> Acquisition: $acq"
    acq_to_ids["$acq"]=""

    for seed in "${seeds[@]}"; do
        out_file="${out_dir}/burgers_${acq}_seed${seed}_k3.out"
        err_file="${err_dir}/burgers_${acq}_seed${seed}_k3.err"

        if [[ ! -f "$out_file" ]]; then
            echo "  ❌ Seed $seed: Output file not found."
            continue
        fi

        if grep -q "Done with active learning experiment." "$out_file"; then
            if [[ ! -f "$err_file" ]]; then
                echo "  ⚠️  Seed $seed: No corresponding .err file found."
                continue
            fi

            # Strip ANSI escape codes before parsing .err
            sync_line=$(sed -r 's/\x1B\[[0-9;]*[mK]//g' "$err_file" | grep "wandb sync")
            sync_path=$(echo "$sync_line" | awk '{print $4}')
            run_id=$(basename "$sync_path" | awk -F'-' '{print $NF}')
            echo "  ✅ Seed $seed: Success (Run ID: $run_id)"

            # Save to map
            acq_to_ids["$acq"]+="$run_id"$'\n'

            # Sync
            echo "     → Syncing: wandb sync $sync_path"
            wandb sync "$sync_path"
        else
            echo "  ❌ Seed $seed: Incomplete or failed."
        fi
    done

    echo
done

echo "----------------------------------------"
echo "W&B Run IDs by acquisition strategy:"
echo "----------------------------------------"
for acq in "${acquisitions[@]}"; do
    echo "## $acq"
    if [[ -n "${acq_to_ids[$acq]}" ]]; then
        echo "${acq_to_ids[$acq]}"
    else
        echo "(no successful runs)"
    fi
    echo
done
