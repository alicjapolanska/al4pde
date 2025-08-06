import os
import re
import shutil
import random
from pathlib import Path

# === CONFIGURATION ===
random_seed = 42

initial_pct = 0.10  # 10% to initial
val_pct     = 0.15  # 15% to validation
test_pct    = 0.15  # 15% to test
# 60% will remain in pool

# === PATHS ===
pool_dir = Path("/pitagora_work/FUPA1_UKAEA_ML/apolansk/al4pde/data/data/jorek/pool")
target_base = Path("/pitagora_work/FUPA1_UKAEA_ML/apolansk/al4pde/data/data/jorek")

# === Setup ===
for subset in ['initial', 'val', 'test']:
    os.makedirs(target_base / subset, exist_ok=True)

# === Get all run IDs from .h5 files ===
pattern = re.compile(r"jorek_run(\d{4})\.h5")
all_h5_files = list(pool_dir.glob("jorek_run*.h5"))
run_ids = sorted([
    int(pattern.search(f.name).group(1))
    for f in all_h5_files
    if pattern.search(f.name)
])

print(f"Total runs found in pool: {len(run_ids)}")

# === Shuffle and split ===
random.seed(random_seed)
random.shuffle(run_ids)

n_total = len(run_ids)
n_initial = int(n_total * initial_pct)
n_val     = int(n_total * val_pct)
n_test    = int(n_total * test_pct)

initial_ids = run_ids[:n_initial]
val_ids     = run_ids[n_initial:n_initial + n_val]
test_ids    = run_ids[n_initial + n_val:n_initial + n_val + n_test]

print(f"Initial:    {len(initial_ids)}")
print(f"Validation: {len(val_ids)}")
print(f"Test:       {len(test_ids)}")
print(f"Remaining in pool: {n_total - (n_initial + n_val + n_test)}")

# === Move .h5 and .txt if present ===
def move_run_files(run_id, dest_dir):
    h5_name = f"jorek_run{run_id:04d}.h5"
    txt_name = f"jorek_input_run{run_id:04d}.txt"

    for filename in [h5_name, txt_name]:
        src = pool_dir / filename
        dst = target_base / dest_dir / filename
        if src.exists():
            shutil.move(str(src), str(dst))
        else:
            print(f"Warning: missing {filename}")

for run_id in initial_ids:
    move_run_files(run_id, "initial")
for run_id in val_ids:
    move_run_files(run_id, "val")
for run_id in test_ids:
    move_run_files(run_id, "test")
