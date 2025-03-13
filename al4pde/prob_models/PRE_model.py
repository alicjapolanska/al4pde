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
        
        if vis:
            print("Plotting PRE")
            self.visualize_PRE(add_to_label = "good_" + str(al_iter))
        
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

    def plot_PRE(self, yy, pde_param: float, save_label: str, save_step: int = 5):
        """Plot PRE for one data instance at timestep . Trajectory should be of shape (1, Nx, Nt, 1).
            save_step (int): time step to plot"""

        PRE = self.residual(yy, pde_param)

        #select time step save_step
        PRE = PRE[:, :, save_step, :]
        yy = yy[:, :, save_step, :]

        #remove dimensions of size 1 (batch size, time and channels)
        PRE = PRE.squeeze()
        yy = yy.squeeze()[1:-1] #cut off boundary

        fig = plt.figure()
        plt.plot(yy, PRE, ".")
        plt.xlabel("x")
        plt.ylabel("PRE")
        plt.savefig("plots/" + save_label + ".png")
        plt.show()   

    def visualize_PRE(self, num_samples: int = 3, add_to_label: str = None, save_step: int = 5):
        """Randomly select num_samples validation instances and plot PRE for their trajectory, as well as rolled out trajectory by the model. The plot is just at time step save_step.

            num_samples (int): number of trajectories to plot
            add_to_label (str): string that will be appended at the end of the plots' filenames
            """

        dataset_size = len(self.val_loader.dataset)  # Total number of samples
        chosen_indices = set(random.sample(range(dataset_size), min(num_samples, dataset_size)))

        current_idx = 0  # Track global index in dataset

        with torch.no_grad():
            for batch_idx, (xx, yy, grid, param, t_idx) in enumerate(self.val_loader):
                batch_size = xx.shape[0]

                print("Shapes: batch_idx", batch_idx, " xx ", xx.shape, " yy ", yy.shape, " grid ", grid.shape, " param ", param.shape)

                for i in range(batch_size):
                    if current_idx in chosen_indices:
                        data_sample = yy[i, :, :, :].unsqueeze(0)  # Keep batch dimension
                        pde_param = param[i, :].item()
                        save_label = f"PRE_val_sample_{current_idx}"
                        if add_to_label:
                            save_label += "_" + add_to_label

                        self.plot_PRE(data_sample, pde_param, save_label, save_step)

                        chosen_indices.remove(current_idx)  # Remove so we stop early if needed
                        if not chosen_indices:  # Stop once we've processed all chosen indices
                            return

                    current_idx += 1  # Update global index across batches

def build_PREModel(task, cfg):
    model = build_wrapper(task, cfg.model_wrapper)
    return PREModel(task,  model)