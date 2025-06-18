import os
import shutil
import re
from math import ceil

# Set paths
source_dir = 'data/data/jorek/pool'
target_dir = 'data/data/jorek/initial'
os.makedirs(target_dir, exist_ok=True)

# Collect valid .h5 files with existing matching .txt files
valid_ids = []
for f in os.listdir(source_dir):
    m = re.match(r'jorek_run(\d{4})\.h5$', f)
    if m:
        run_id = m.group(1)
        txt_name = f'jorek_input_run{run_id}.txt'
        if os.path.exists(os.path.join(source_dir, txt_name)):
            valid_ids.append(int(run_id))  # store as int for proper sorting

# Sort and compute 15%
valid_ids.sort()
n_to_move = ceil(0.15 * len(valid_ids))
ids_to_move = valid_ids[:n_to_move]

# Move files
for i, run_id in enumerate(ids_to_move, 1):
    run_str = f"{run_id:04d}"
    print(f"[{i}/{n_to_move}] Moving run {run_str}")
    
    h5_file = f"jorek_run{run_str}.h5"
    txt_file = f"jorek_input_run{run_str}.txt"

    for file in [h5_file, txt_file]:
        src = os.path.join(source_dir, file)
        dst = os.path.join(target_dir, file)
        if os.path.exists(src):
            shutil.move(src, dst)

        else:
            print(f"Warning: {file} missing in source folder")
