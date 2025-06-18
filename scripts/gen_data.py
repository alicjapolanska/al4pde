import os
import re


def generate_data_init(task):
    # Extract run indices from the initial folder
    initial_folder = os.path.join(task.data_path, task.pde_name, "initial")
    run_indices = sorted([int(re.search(r'jorek_run(\d+)\.h5', file).group(1)) for file in os.listdir(initial_folder) if file.endswith(".h5")])

    # Generate and save trajectories using task.save_trajectories for each run index
    for run_index in run_indices:
        print(f"Generating data for run index {run_index}")
        task.save_trajectories(run_index, "initial")