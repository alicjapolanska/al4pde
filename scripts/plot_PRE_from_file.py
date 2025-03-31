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
import random

sys.path.append('/leonardo/home/userexternal/apolansk/codes/pdearena')

import hydra
from omegaconf import OmegaConf, DictConfig

from al4pde.prob_models.PRE_model import PREModel
from al4pde.prob_models.build_prob_model import build_prob_model
import matplotlib.pyplot as plt


def choose_idxs_to_plot(prob_model, num_to_plot):
    """Choose num_to_plot number if random points from the validation set to plot.
        If num_to_plot is bigger than validation dataset size,
        return all indices in the dataset size. Returns a set of indices."""
    
    dataset_size = len(prob_model.val_loader.dataset)  # Total number of samples
    idxs_to_plot = set(random.sample(range(dataset_size), min(num_to_plot, dataset_size)))

    return idxs_to_plot


def calculate_current_predictions(prob_model, idxs_to_plot):
    """Returns dictionary with keys being indices of datapoints in validation set and 
            items being a list of the trajectory and pde param. Returns tuple containing 
            dictionary for unnormalized model predictions and ground truths respectively.
        """

    chosen_indices = idxs_to_plot.copy()
    current_idx = 0  # Track global index in dataset
    ground_truths = {}
    predictions = {}

    with torch.no_grad():
        for batch_idx, (xx, yy, grid, param, t_idx) in enumerate(prob_model.val_loader):
            batch_size = xx.shape[0]

            for i in range(batch_size):
                if current_idx in chosen_indices:
                    sample_traj = yy[i, :, :, :].unsqueeze(0)  # Keep batch dimension
                    pde_param = param[i, :].item()

                    ground_truths[current_idx] = [sample_traj, pde_param]

                    xx = xx.to(device)
                    grid = grid.to(device)
                    param = param.to(device)
                    t_idx = t_idx.to(device)
                    pred = prob_model.model.roll_out(xx, grid, yy.shape[2], param, t_idx)[i, :, :, :].unsqueeze(0)
                    #pred = prob_model.model.task_norm.denorm_traj(pred)
                    pred = pred.to("cpu")

                    predictions[current_idx] = [pred, pde_param]


                    chosen_indices.remove(current_idx)  # Remove so we stop early if needed
                    if not chosen_indices:  # Stop once we've processed all chosen indices
                        return predictions, ground_truths

                current_idx += 1  # Update global index across batches

def calculate_mean_PRE_v_t(prob_model):
    """Calculate PRE averaged over the validation set and x over time for prob_model."""

    created_summary_arr = False
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (xx, yy, grid, param, t_idx) in enumerate(prob_model.val_loader):
            num_batches += 1

            xx = xx.to(device)
            grid = grid.to(device)
            param = param.to(device)
            t_idx = t_idx.to(device)
            yy = yy.to(device)

            pred = prob_model.model.roll_out(xx, grid, yy.shape[2], param, t_idx)
            #pred = prob_model.model.task_norm.denorm_traj(pred)
            unc = prob_model.residual(pred, param).squeeze() #squeeze out channel dim of size 1
            unc_av = torch.mean(unc, dim=(0,1)) #mean over batch and x

            unc_sim = prob_model.residual(yy, param).squeeze() #squeeze out channel dim of size 1
            unc_av_sim = torch.mean(unc_sim, dim=(0,1)) #mean over batch and x

            err = (unc-unc_sim)**2 
            err_av = torch.mean(err, dim=(0,1)) #mean over batch and x

            if not created_summary_arr:
                unc_av_all = torch.zeros_like(unc_av)
                unc_av_sim_all = torch.zeros_like(unc_av_sim)
                err_av_all = torch.zeros_like(err_av)
                created_summary_arr = True
            
            unc_av_all += unc_av
            unc_av_sim_all += unc_av_sim
            err_av_all += err_av
        
    unc_av_all *= 1/num_batches
    unc_av_sim_all *= 1/num_batches
    err_av_all *= 1/num_batches

    return unc_av_all, unc_av_sim_all, err_av_all

def plot_mean_PRE_v_t(unc_av_all, unc_av_sim_all, al_iter, save_path):
    """Plot the average PRE over time for model and simulation.

        Inputs:
            unc_av_all - PRE averaged over batch dimension and x for model 
                prediction, vector of size Nt

            unc_av_sim_all - PRE averaged over batch dimension and x for 
                simulation (ground truth), vector of size Nt

            al_iter (int) - active learning iteration the model is from

            save_path (str) - path where figure should be saved """

    if not unc_av_all.device == "cpu":
        unc_av_all = unc_av_all.to("cpu")
    if not unc_av_sim_all.device == "cpu":
        unc_av_sim_all = unc_av_sim_all.to("cpu")

    # Define t axis
    Nt = len(unc_av_all)
    t_vals = np.arange(Nt)
    
    fig, ax = plt.subplots()
    plt.plot(t_vals, unc_av_sim_all, "--", label="Simulation")
    plt.plot(t_vals, unc_av_all, "--", label="Model after iteration")
    plt.title("Average PRE over time for iteration " + str(al_iter))
    plt.ylabel("PRE")
    plt.xlabel("t")
    plt.legend()
    plt.savefig(os.path.join(save_path, "PRE_v_t_al_it" + str(al_iter) + ".png"))
    plt.show()

def plot_mean_PRE_MSE_v_t(err_av_all, al_iter, save_path):
    """Plot the average PRE MSE of model wrt simulation over time.

        Inputs:
            unc_av_all - (model PRE - simulation PRE)^2 averaged over 
                batch dimension and x, vector of size Nt

            al_iter (int) - active learning iteration the model is from

            save_path (str) - path where figure should be saved """

    if not err_av_all.device == "cpu":
        err_av_all = err_av_all.to("cpu")

    # Define t axis
    Nt = len(err_av_all)
    t_vals = np.arange(Nt)
    
    fig, ax = plt.subplots()
    plt.plot(t_vals, err_av_all, "--")
    plt.title("Average PRE MSE over time for iteration " + str(al_iter))
    plt.ylabel("(model PRE - simulation PRE)^2")
    plt.xlabel("t")
    plt.legend()
    plt.savefig(os.path.join(save_path, "PRE_MSE_v_t_al_it" + str(al_iter) + ".png"))
    plt.show()


def plot_PRE_comp(PRE_traj, PRE_before, PRE_after, al_iter, data_idx, save_path):
    """Plot PRE as heatmaps for a data instance over all time steps.
        Assumes PRE has shape (Nx, Nt)."""
    

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
    plt.savefig(os.path.join(save_path, "PRE_comp_al_it_" + str(al_iter) + "_" + str(data_idx) + ".png"))
    plt.show()


def plot_PRE_slice(PRE_traj, PRE_before, PRE_after, times_to_plot, al_iter, data_idx, save_path):
    """Plot PRE as heatmaps for a data instance over time steps in times_to_plot.
        Assumes PRE has shape (Nx, Nt)."""
    
    x_vals = np.arange(PRE_traj.shape[0])  # Spatial dimension

    for time in times_to_plot:

        PRE_traj_slice = PRE_traj[:,time].squeeze()
        PRE_before_slice = PRE_before[:,time].squeeze()
        PRE_after_slice = PRE_after[:,time].squeeze()

        fig, ax = plt.subplots()
        plt.plot(x_vals, PRE_before_slice, "--", label="Before iteration")
        plt.plot(x_vals, PRE_traj_slice, "--", label="GT")
        plt.plot(x_vals, PRE_after_slice, "--", label="After iteration")
        plt.title("PRE slice at time "+str(time) + " iteration " + str(al_iter))
        plt.ylabel("PRE")
        plt.xlabel("x")
        plt.legend()
        plt.savefig(os.path.join(save_path, "PRE_slice_t" + str(time) + "_al_it" + str(al_iter) + "_" + str(data_idx) + ".png"))
        plt.show()

@hydra.main(version_base="1.3.2", config_path="../config", config_name="main")
def main(cfg: DictConfig):

    num_al_iter =  cfg.num_al_iter
    times_to_plot = [3,15,35,38] 
    num_to_plot = 3 #how many datapoints to choose from val set

    for al_iter in range(1,num_al_iter):
        print("AL iter", al_iter)
        run_id = cfg.checkpoint_id


        print("run_id", run_id, flush=True)
        print("torch_device", device)
        print("jax_dev", xla_bridge.get_backend().platform, flush=True)

        run_save_path = os.path.join(cfg.task.run_save_path, run_id)
        task = hydra.utils.instantiate(cfg.task, run_save_path=run_save_path)

        # Include zeroth iteration
        if al_iter == 1:
            cfg.prob_model = OmegaConf.to_container(cfg.prob_model, resolve=True)
            prob_model = build_prob_model(task, cfg.prob_model)

            save_dict = torch.load(os.path.join(run_save_path, "checkpoints", str(al_iter-1)+".pt"))
            prob_model.init_training(al_iter-1)
            print("Task norm is ", prob_model.task_norm)
            prob_model.load_state_dict(save_dict['model'])
            print("Prob model dict keys ", save_dict.keys())
            print("Model dict keys ", save_dict['model'].keys())
            print("Prob model keys:", prob_model.state_dict().keys())
            print("Model keys:", prob_model.model.state_dict().keys())

            # Choose idxs at random and keep them constant
            idxs_to_plot = choose_idxs_to_plot(prob_model, num_to_plot)

            # predictions after 0th iter
            pred_after_last_iter, ground_truth_pred = calculate_current_predictions(prob_model, idxs_to_plot)

            PRE_traj_dict = {}
            PRE_before_dict = {}
            for data_idx in ground_truth_pred:
                PRE_traj_dict[data_idx] = prob_model.residual(*ground_truth_pred[data_idx]).squeeze()
                PRE_before_dict[data_idx] = prob_model.residual(*pred_after_last_iter[data_idx]).squeeze()
            
            save_path = os.path.join(run_save_path, "img")
            print("Calculating mean PRE v t")
            PRE_av, PRE_av_sim, PRE_MSE = calculate_mean_PRE_v_t(prob_model)
            plot_mean_PRE_v_t(PRE_av, PRE_av_sim, al_iter-1, save_path)

        cfg.prob_model = OmegaConf.to_container(cfg.prob_model, resolve=True)
        prob_model = build_prob_model(task, cfg.prob_model)

        save_dict = torch.load(os.path.join(run_save_path, "checkpoints", str(al_iter)+".pt"))

        prob_model.init_training(0)

        print("Model keys:", prob_model.model.state_dict().keys())
        print("Prob model keys:", prob_model.state_dict().keys())
        print("Task norm is ", prob_model.task_norm)
        prob_model.load_state_dict(save_dict['model'])

        pred_after_current_iter, temp = calculate_current_predictions(prob_model, idxs_to_plot)
        
        PRE_after_dict = {}
        for data_idx in ground_truth_pred:
            PRE_after_dict[data_idx] = prob_model.residual(*pred_after_current_iter[data_idx]).squeeze()

        for data_idx in ground_truth_pred:
            save_path = os.path.join(run_save_path, "img")
            plot_PRE_comp(PRE_traj_dict[data_idx], PRE_before_dict[data_idx], PRE_after_dict[data_idx], al_iter, data_idx, save_path)
            plot_PRE_slice(PRE_traj_dict[data_idx], PRE_before_dict[data_idx], PRE_after_dict[data_idx], times_to_plot, al_iter, data_idx, save_path)


        print("Calculating mean PRE v t")
        PRE_av, PRE_av_sim, PRE_MSE = calculate_mean_PRE_v_t(prob_model)
        plot_mean_PRE_v_t(PRE_av, PRE_av_sim, al_iter, save_path)
        plot_mean_PRE_MSE_v_t(PRE_MSE, al_iter, save_path)

        PRE_before_dict = PRE_after_dict
        pred_after_last_iter = pred_after_current_iter

if __name__ == "__main__":
    main()
    print("Done.", flush=True)
