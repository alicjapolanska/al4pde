import numpy as np
import os
import sys 
import wandb
wandb.init(mode="offline")
import torch
import jax.numpy as jnp
jnp.arange(0, 100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from jax.lib import xla_bridge

sys.path.append('/leonardo/home/userexternal/apolansk/codes/pdearena')

import hydra
from omegaconf import OmegaConf, DictConfig

from al4pde.prob_models.PRE_model import PREModel
from al4pde.prob_models.build_prob_model import build_prob_model
import matplotlib.pyplot as plt

def plot_PRE_comp(PRE_traj, PRE_before, PRE_after, save_path):
    """Plot PRE as heatmaps for a data instance over all time steps.
    Assumes trajectory has shape (1, Nx, Nt, 1).
    
        add_to_label (str): string that will be appended at the end of the plots' filenames"""
    


    # Define x and t axes
    Nx, Nt = PRE_before.shape
    x_vals = np.arange(Nx)  # Spatial dimension
    t_vals = np.arange(Nt)  # Time dimension
    
    # Set consistent color scale
    vmin = min(PRE_before.min(), PRE_traj.min(), PRE_after.min())
    vmax = max(PRE_before.max(), PRE_traj.max(), PRE_after.max())

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 5), constrained_layout=True)

    # Plot "Before Training" heatmap
    im1 = axes[0].imshow(PRE_before, aspect='auto', origin='lower', cmap='coolwarm',
                        extent=[t_vals.min(), t_vals.max(), x_vals.min(), x_vals.max()],
                        vmin=vmin, vmax=vmax)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("x")
    axes[0].set_yticks([])
    axes[0].set_title("PRE Before AL Iter")

    # Plot "Ground Truth" heatmap
    im2 = axes[1].imshow(PRE_traj, aspect='auto', origin='lower', cmap='coolwarm',
                        extent=[t_vals.min(), t_vals.max(), x_vals.min(), x_vals.max()],
                        vmin=vmin, vmax=vmax)
    axes[1].set_xlabel("Time")
    axes[1].set_yticks([])
    axes[1].set_title("Ground Truth PRE")

    # Plot "After Training" heatmap
    im3 = axes[2].imshow(PRE_after, aspect='auto', origin='lower', cmap='coolwarm',
                        extent=[t_vals.min(), t_vals.max(), x_vals.min(), x_vals.max()],
                        vmin=vmin, vmax=vmax)
    axes[2].set_xlabel("Time")
    axes[2].set_yticks([])
    axes[2].set_title("PRE After AL Iter")

    # Add shared colorbar
    cbar = fig.colorbar(im3, ax=axes, orientation='vertical', fraction=0.02)
    cbar.set_label("PRE Value")

    # Save and show plot
    plt.savefig(save_path)
    plt.show()



@hydra.main(version_base="1.3.2", config_path="../config", config_name="main")
def main(cfg: DictConfig):

    num_al_iter =  cfg.num_al_iter

    for al_iter in range(1,num_al_iter):
        print("AL iter", al_iter)
        run_id = cfg.checkpoint_id


        print("run_id", run_id, flush=True)
        print("torch_device", device)
        print("jax_dev", xla_bridge.get_backend().platform, flush=True)

        run_save_path = os.path.join(cfg.task.run_save_path, run_id)
        task = hydra.utils.instantiate(cfg.task, run_save_path=run_save_path)

        # previous
        if al_iter == 1:
            cfg.prob_model = OmegaConf.to_container(cfg.prob_model, resolve=True)
            prob_model = build_prob_model(task, cfg.prob_model)

            save_dict = torch.load(os.path.join(run_save_path, "checkpoints", str(al_iter-1)+".pt"))
            prob_model.init_training(al_iter-1)
            print("Task norm is ", prob_model.task_norm)
            prob_model.load_state_dict(save_dict['model'])

            # Choose idxs at random and keep them constant
            prob_model.choose_idxs_to_plot()
            idxs_to_plot = prob_model.idxs_to_plot

            prob_model.calculate_current_predictions(model_trained= False)
            prob_model.calculate_current_predictions(model_trained= True)
            # predictions after 0th iter
            pred_after_last_iter = prob_model.current_good_predictions
            ground_truth_pred = prob_model.current_ground_truths

            PRE_traj_dict = {}
            PRE_before_dict = {}
            for data_idx in ground_truth_pred:
                PRE_traj_dict[data_idx] = prob_model.residual(*ground_truth_pred[data_idx]).squeeze()
                PRE_before_dict[data_idx] = prob_model.residual(*pred_after_last_iter[data_idx]).squeeze()

        cfg.prob_model = OmegaConf.to_container(cfg.prob_model, resolve=True)
        prob_model = build_prob_model(task, cfg.prob_model)

        save_dict = torch.load(os.path.join(run_save_path, "checkpoints", str(al_iter)+".pt"))

        prob_model.init_training(al_iter)

        print("Model keys:", prob_model.model.state_dict().keys())
        print("Prob model keys:", prob_model.state_dict().keys())
        print("Task norm is ", prob_model.task_norm)
        prob_model.load_state_dict(save_dict['model'])

        # Choose idxs at random and keep them constant
        prob_model.idxs_to_plot = idxs_to_plot

        prob_model.calculate_current_predictions(model_trained= True)
        pred_after_current_iter = prob_model.current_good_predictions
        
        PRE_after_dict = {}
        for data_idx in ground_truth_pred:
            PRE_after_dict[data_idx] =prob_model.residual(*pred_after_current_iter[data_idx]).squeeze()

        for data_idx in ground_truth_pred:
            save_path = os.path.join(run_save_path, "img", "PRE_comp_it_" + str(al_iter) + "_" + str(data_idx) + ".png")
            plot_PRE_comp(PRE_traj_dict[data_idx], PRE_before_dict[data_idx], PRE_after_dict[data_idx], save_path)

        PRE_after_dict = PRE_before_dict
        pred_after_last_iter = pred_after_current_iter

if __name__ == "__main__":
    main()
    print("Done.", flush=True)
