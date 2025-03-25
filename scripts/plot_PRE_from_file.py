import numpy as np
import os
import sys 
import wandb
wandb.init(mode="offline")
import torch
import jax
import jax.numpy as jnp
jnp.arange(0, 100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from jax.lib import xla_bridge

sys.path.append('/leonardo/home/userexternal/apolansk/codes/pdearena')

import hydra
from omegaconf import OmegaConf, DictConfig

from al4pde.prob_models.PRE_model import PREModel
from al4pde.utils import load_checkpoint, set_current_seed
from al4pde.prob_models.build_prob_model import build_prob_model
from al4pde.acquisition.build_selection import build_strategy

if False:
    path_to_models = "data/runs/cnr1ahtx"
    num_al_iter = 4

    @hydra.main(version_base="1.3.2", config_path="../config", config_name="main")
    def main(cfg: DictConfig):

        run_id = cfg.checkpoint_id
        run_save_path = os.path.join(cfg.task.run_save_path, run_id)
        print(run_save_path)
        task = hydra.utils.instantiate(cfg.task, run_save_path=run_save_path)
        model = build_prob_model(task, cfg.prob_model)

        for i in range(num_al_iter-1):

            load_checkpoint(path_to_models, str(i)+".pt", model)

            print(model.current_ground_truths)


    al_iter = 0
    run = wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg)
    )

    # init components
    set_current_seed(cfg.seed, 0, sampling_finished=True, task=None, use_test=use_test)
    run_save_path = os.path.join(cfg.task.run_save_path, run_id)
    task = hydra.utils.instantiate(cfg.task, run_save_path=run_save_path)
    acq_strat = build_strategy(task, cfg.acquisition)
    print("acq", acq_strat, flush=True)

    if __name__ == "__main__":
        main()
        print("Done")




@hydra.main(version_base="1.3.2", config_path="../config", config_name="main")
def main(cfg: DictConfig):

    path_to_models = "data/runs/cnr1ahtx/checkpoints/"
    num_al_iter = 0
    run_id = cfg.checkpoint_id


    print("run_id", run_id, flush=True)
    print("torch_device", device)
    print("jax_dev", xla_bridge.get_backend().platform, flush=True)

    run_save_path = os.path.join(cfg.task.run_save_path, run_id)
    task = hydra.utils.instantiate(cfg.task, run_save_path=run_save_path)
    print("Prob model cfg", cfg.prob_model)
    cfg.prob_model = OmegaConf.to_container(cfg.prob_model, resolve=True)
    print("Prob model cfg", cfg.prob_model)
    prob_model = build_prob_model(task, cfg.prob_model)

    save_dict = torch.load(os.path.join(run_save_path, "checkpoints", str(num_al_iter)+".pt"))
    print("Dict keys ", save_dict['model'].keys())
    prob_model.init_training(num_al_iter)
    print("Model keys ", prob_model.model.state_dict().keys())


    print(prob_model.current_ground_truths)

    prob_model.choose_idxs_to_plot()
    prob_model.calculate_current_predictions(model_trained= False)
    prob_model.calculate_current_predictions(model_trained= True)

    print(prob_model.current_ground_truths)


if __name__ == "__main__":
    main()
    print("Done.", flush=True)
