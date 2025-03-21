import numpy as np
import os
import sys 

sys.path.append('/leonardo/home/userexternal/apolansk/codes/pdearena')

import hydra
from omegaconf import DictConfig

from al4pde.prob_models.PRE_model import PREModel
from al4pde.utils import load_checkpoint
from al4pde.prob_models.build_prob_model import build_prob_model

path_to_models = "data/runs/cnr1ahtx/checkpoints"
num_al_iter = 4

@hydra.main(version_base="1.3.2", config_path="../config", config_name="main")
def main(cfg: DictConfig):

    run_id = cfg.checkpoint_id
    run_save_path = os.path.join(cfg.task.run_save_path, run_id)
    print(run_save_path)
    task = hydra.utils.instantiate(cfg.task, run_save_path=run_save_path)
    model = build_prob_model(task, cfg.prob_model)

    for i in range(num_al_iter-1):

        load_checkpoint(path_to_models, str(i), model)

        print(model.current_ground_truths)


if __name__ == "__main__":
    main()
    print("Done")