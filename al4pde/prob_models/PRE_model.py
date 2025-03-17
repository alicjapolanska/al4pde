import os
import time
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import numpy as np
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import random

from al4pde.evaluation.visualization import plot_worst_traj
from al4pde.prob_models.prob_model import ProbModel
from al4pde.models.build_wrapper import build_wrapper
from al4pde.evaluation.stats import LossUncCorr, UncAvg
from al4pde.prob_models.PREConvOps import ConvOps_1d


class PREModel(ProbModel):

    def __init__(self, task, model_cfg):
        super().__init__(task, model_cfg.training_type, model_cfg.t_train, model_cfg.batch_size, model_cfg.val_period, model_cfg.vis_period, model_cfg.loss)
        self.model = model_cfg
        self.stats.append(UncAvg("unc")) 
        self.stats.append(LossUncCorr("corr_unc_loss", self.loss))
        self.current_bad_predictions = {}
        self.current_good_predictions = {}
        self.current_ground_truths = {}
    
    def residual(self, uu, pde_param: float, boundary: bool = False, dx: float = 0.001, dt: float = 0.05):
        """Compute PRE residual for a rolled out solution u. Hardcoded to Burgers (for now).
            uu  - tensor containing solution value (bs, nx, nt, 1)
            pde_param - Burgers parameter nu (bs, 1)
            boundary - Whether to include boundary in PRE or not
            # TODO Feed in dx and dt from config file
            """
        device = uu.device  # Get the device of uu (GPU or CPU)

        dx = torch.tensor(dx, dtype=torch.float32, device=device)
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        nu = torch.tensor(pde_param, dtype=torch.float32, device=device).unsqueeze(-1)
        #print("Field shape before permuting ", uu.shape)
        #print("Shape of nu ", nu.shape)
        # solutions are [bs, nx, nt, nc] but for PRE code we need [BS, Nt, nx]
        uu = uu.squeeze(-1) #last dimension is just one channel, squeeze out
        uu = uu.permute(0, 2, 1) #permute for correct PRE computation
        #print("Field shape after permuting ", uu.shape)

        #Defining the required Convolutional Operations. 
        D_t = ConvOps_1d.ConvOperator(domain='t', order=1, device=device)
        D_x = ConvOps_1d.ConvOperator(domain='x', order=1, device=device)
        D_xx = ConvOps_1d.ConvOperator(domain='x', order=2, device=device)

        res = dx*D_t(uu) + dt * uu * D_x(uu) - nu / np.pi * D_xx(uu) * (2*dt/dx)
        #print("Residual shape ", res.shape)

        if boundary:
            return res.permute(0, 2, 1).unsqueeze(-1)
        else: 
            res = res[...,1:-1,1:-1].permute(0, 2, 1).unsqueeze(-1)
            #print("Residual shape after permuting ", res.shape)
            return res
        
    @property
    def val_loader(self):
        return self.model.val_loader

    @property
    def train_loader(self):
        return self.model.train_loader

    @property
    def train_loader_full_traj(self):
        return self.model.train_loader_full_traj
    
    def eval_pred(self, xx, yy, grid, param=None, t_idx=None):
        xx = xx.to(device)
        grid = grid.to(device)
        yy = yy.to(device)
        param = param.to(device)
        t_idx = t_idx.to(device)
        pred, unc = self.unc_roll_out(xx, grid, yy.shape[-2], param, t_idx)
        onestep_pred = self.one_step_pred(yy, grid, param, t_idx)
        return {"pred": pred, "onestep_pred": onestep_pred, "yy": yy, "param": param, "unc": unc}
    
    def train_single_epoch(self, current_epoch, total_epoch, num_epoch):
        self.model.train_single_epoch(current_epoch, total_epoch, num_epoch)
    
    def train_n_epoch(self, al_iter: int, num_epoch: int, step_offset: int, vis: bool = True,
                      prefix: str = "", is_last=False) -> float:
        """ Train model for num_epoch epochs.
        @param al_iter: Current AL iteration.
        @param num_epoch: Number of epochs to train for.
        @param step_offset: Total epoch added up over all AL iterations.
        @param vis: Whether to visualize.
        @param prefix: Will be put before the name of all logged metrics.
        @return: Training duration in seconds.
        """
        total_time = 0
        self.model.init_training(al_iter) #Basically the same as the one in model but it's self.model.init_training
        self.task_norm = self.model.task_norm
        
        for i in range(num_epoch):
            t = time.time()
            self.train_single_epoch(i, step_offset + i, num_epoch)
            total_time += time.time() - t
            if i % self.val_period == 0:
                self.validate(step_offset + i, prefix=prefix)
            if vis:
                if (i > 0 and i % self.vis_period == 0) or i == num_epoch - 1:
                    self.visualize(step_offset + i)
            if i==2:
                if vis:
                    # Calculate prediction at second step
                    print("Calculating bad model prediction")
                    self.calculate_current_predictions(model_trained = False)
        
        if vis:
            # Calculate prediction after training
            print("Calculating good model prediction")
            self.calculate_current_predictions(model_trained = False)
            print("Plot PRE comparison over trajectories")
            self.plot_PRE(add_to_label = "_" + str(al_iter))
        
            # Reset prediction to plot PRE for
            self.current_bad_predictions = {}
            self.current_good_predictions = {}
            self.current_ground_truths = {}
        
        return total_time
        
    def unc_roll_out(self, xx, grid, final_step,  pde_param=None, t_idx=None, return_features=False):

        if self.training_type in ['autoregressive', 'teacher_forcing']:
        
            #print("Input field shape is ", xx.shape)
            pred = self.model.roll_out(xx, grid, final_step, pde_param, t_idx, return_features)
            #print("Prediction shape is ", pred.shape)
            unc = torch.abs(self.residual(pred, pde_param))
            #print("Uncertainty shape is ", unc.shape)

            return pred, unc

        else:
            raise ValueError(self.training_type)

    def forward(self, xx, grid, pde_param=None, t_idx=None):
        return self.model(xx, grid, pde_param, t_idx)

    def plot_PRE(self, add_to_label: str = None):
        """Plot PRE as heatmaps for a data instance over all time steps.
        Assumes trajectory has shape (1, Nx, Nt, 1).
        
            add_to_label (str): string that will be appended at the end of the plots' filenames"""

        PRE = self.residual(yy, pde_param)  # Compute residuals
        PRE = PRE.squeeze()  # Remove batch and channel dimensions

        x_vals = np.arange(PRE.shape[0])  # x-axis (spatial dimension)
        t_vals = np.arange(PRE.shape[1])  # y-axis (time dimension)

        fig, ax = plt.subplots()
        
        # Heatmap of PRE
        im1 = ax.imshow(PRE, aspect='auto', origin='lower', cmap='coolwarm', 
                            extent=[t_vals.min(), t_vals.max(), x_vals.min(), x_vals.max()])
        ax.set_xlabel("Time")
        ax.set_yticks([])  # Remove y-axis ticks
        fig.colorbar(im1, label="PRE")   
        plt.tight_layout()
        plt.savefig(os.path.join(self.task.img_save_path, save_label + ".png"))
        plt.show()

        for data_idx in self.current_ground_truths:

            PRE_traj = self.residual(*self.current_ground_truths[data_idx]).squeeze()  # Compute residuals
            PRE_before = self.residual(*self.current_bad_predictions[data_idx]).squeeze()  # Compute residuals
            PRE_after = self.residual(*self.current_good_predictions[data_idx]).squeeze()  # Compute residuals

            # Define x and t axes
            Nx, Nt = PRE_before.shape
            x_vals = np.arange(Nx)  # Spatial dimension
            t_vals = np.arange(Nt)  # Time dimension
            
            # Set consistent color scale
            vmin = min(PRE_before.min(), PRE_traj.min(), PRE_after.min())
            vmax = max(PRE_before.max(), PRE_traj.max(), PRE_after.max())

            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Plot "Before Training" heatmap
            im1 = axes[0].imshow(PRE_before, aspect='auto', origin='lower', cmap='coolwarm',
                                extent=[t_vals.min(), t_vals.max(), x_vals.min(), x_vals.max()],
                                vmin=vmin, vmax=vmax)
            axes[0].set_xlabel("Time")
            axes[0].set_ylabel("x")
            axes[0].set_yticks([])
            axes[0].set_title("PRE Before Training")

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
            axes[2].set_title("PRE After Training")

            # Add shared colorbar
            cbar = fig.colorbar(im3, ax=axes, orientation='vertical', fraction=0.02)
            cbar.set_label("PRE Value")

            plt.tight_layout()

            # Save and show plot
            plt.savefig(os.path.join(self.task.img_save_path, "PRE_comp_" + str(data_idx) + add_to_label + ".png"))
            plt.show()


    def calculate_current_predictions(self, model_trained: bool, num_samples: int = 3):
        """Randomly select num_samples validation instances and plot PRE for their trajectory, as well as rolled out trajectory by the model. 
            The plot is just at time step save_step.

            model_trained (bool): Whether model is trained or not. Determines if predictions are stored in 
                current_bad_predictions (False) or current_good_predictions (True). If False, ground truths 
                are also added to self.current_ground_truths.

            num_samples (int): number of trajectories to plot
            """

        if not model_trained:
            dataset_size = len(self.val_loader.dataset)  # Total number of samples
            chosen_indices = set(random.sample(range(dataset_size), min(num_samples, dataset_size)))
        else:
            chosen_indices = [*self.current_ground_truths] # Keys ie data indices as a list

        current_idx = 0  # Track global index in dataset
        predictions = {}

        with torch.no_grad():
            for batch_idx, (xx, yy, grid, param, t_idx) in enumerate(self.val_loader):
                batch_size = xx.shape[0]

                print("Shapes: batch_idx", batch_idx, " xx ", xx.shape, " yy ", yy.shape, " grid ", grid.shape, " param ", param.shape)


                for i in range(batch_size):
                    if current_idx in chosen_indices:
                        sample_traj = yy[i, :, :, :].unsqueeze(0)  # Keep batch dimension
                        pde_param = param[i, :].item()

                        if not model_trained:
                            self.current_ground_truths[current_idx] = [sample_traj, pde_param]

                        print("Devices xx ", xx.device, " grid ", grid.device, " param ", param.device)
                        #plot rolled out timestep
                        xx = xx.to(device)
                        grid = grid.to(device)
                        param = param.to(device)
                        t_idx = t_idx.to(device)
                        pred = self.model.roll_out(xx, grid, yy.shape[2], param, t_idx)[i, :, :, :].unsqueeze(0)
                        pred = pred.to("cpu")

                        predictions[current_idx] = [pred, pde_param]


                        chosen_indices.remove(current_idx)  # Remove so we stop early if needed
                        if not chosen_indices:  # Stop once we've processed all chosen indices
                            if not model_trained:
                                self.current_bad_predictions = predictions
                            else:
                                self.current_good_predictions = predictions
                            return

                    current_idx += 1  # Update global index across batches

def build_PREModel(task, cfg):
    model = build_wrapper(task, cfg.model_wrapper)
    return PREModel(task,  model)