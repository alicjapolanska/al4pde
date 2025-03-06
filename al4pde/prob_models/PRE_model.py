import os
import time
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import numpy as np
from hydra.utils import instantiate

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
            uu  - tensor containing solution value 
            pde_param - Burgers parameter nu 
            boundary - Whether to include boundary in PRE or not
            # TODO Feed in dx and dt from config file
            """
        device = uu.device  # Get the device of uu (GPU or CPU)

        dx = torch.tensor(dx, dtype=torch.float32, device=device)
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        nu = torch.tensor(pde_param, dtype=torch.float32, device=device)

        # solutions are [bs, nx, nt, nc] but for PRE code we need [BS, Nt, nx]
        uu = uu.squeeze(-1) #last dimension is just one channel, squeeze out
        uu = uu.permute(0, 2, 1) #permute for correct PRE computation

        #Defining the required Convolutional Operations. 
        D_t = ConvOps_1d.ConvOperator(domain='t', order=1, device=device)
        D_x = ConvOps_1d.ConvOperator(domain='x', order=1, device=device)
        D_xx = ConvOps_1d.ConvOperator(domain='x', order=2, device=device)

        res = dx*D_t(uu) + dt * uu * D_x(uu) - nu / np.pi * D_xx(uu) * (2*dt/dx)
        #print("Residual shape ", res.shape)

        if boundary:
            return res.permute(0, 2, 1).unsqueeze(-1)
        else: 
            return res[...,1:-1,1:-1].permute(0, 2, 1).unsqueeze(-1)
        
    @property
    def val_loader(self):
        return self.model.val_loader

    @property
    def train_loader(self):
        return self.model.train_loader

    @property
    def train_loader_full_traj(self):
        return self.model.train_loader_full_traj
    

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
        for i in range(num_epoch):
            t = time.time()
            self.train_single_epoch(i, step_offset + i, num_epoch)
            total_time += time.time() - t
            if i % self.val_period == 0:
                self.validate(step_offset + i, prefix=prefix)
            if vis:
                if (i > 0 and i % self.vis_period == 0) or i == num_epoch - 1:
                    self.visualize(step_offset + i)
        return total_time

    #def uncertainty(self, xx, grid, final_step, pde_param=None, t_idx=None, return_state=False, return_features=False):
        #print("pde param", pde_param, pde_param.shape)
        #if pde_param.dim() == 1:
        #    pde_param = pde_param.unsqueeze(0)
        #    print("pde param unsqueezed", pde_param, pde_param.shape)
        #pred = self.model.roll_out(xx, grid, final_step, pde_param, t_idx, return_features)
    def uncertainty(self, xx, grid, t_idx=None, pde_param=None, return_state=True):
        pred = self.model(xx, grid, pde_param)
        #print("Pred shape", pred.shape)
        unc = self.residual(pred, pde_param)
        if return_state:
            return pred, unc
        return unc

    def unc_roll_out(self, xx, grid, final_step,  pde_param=None, t_idx=None, return_features=False):
        unc_cell = []
        if self.training_type in ['autoregressive', 'teacher_forcing']:
            pred = xx
            unc_cell = []
            for t in range(self.initial_step, final_step):
                m, unc = self.uncertainty(xx, grid, t_idx, pde_param, True)
                pred = torch.cat((pred, m), -2)
                xx = torch.cat((xx[..., 1:, :], m), dim=-2)
                if not unc_cell:
                    unc_cell.append(torch.zeros_like(unc)) #append zeros for first entry
                unc_cell.append(unc)

                if t_idx is not None:
                    t_idx += 1
            print("Print in loop.")
            for unc in unc_cell:
                print("Uncertainty shape ", unc.shape)
            return pred, torch.concat(unc_cell, dim=-2)

        else:
            raise ValueError(self.training_type)

    def forward(self, xx, grid, pde_param=None, t_idx=None):
        return self.model(xx, grid, pde_param, t_idx)


def build_PREModel(task, cfg):
    model = build_wrapper(task, cfg.model_wrapper)
    return PREModel(task,  model)