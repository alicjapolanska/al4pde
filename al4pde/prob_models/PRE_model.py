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
        self.model = instantiate(model_cfg)
        self.stats.append(UncAvg("unc")) 
        self.stats.append(LossUncCorr("corr_unc_loss", self.loss))
    
    def residual(uu, pde_param: float, boundary: bool = False, dx: float = 0.001, dt: float = 0.05):
        """Compute PRE residual for a rolled out solution u. Hardcoded to Burgers (for now).
            uu  - tensor containing solution value 
            pde_param - Burgers parameter nu 
            boundary - Whether to include boundary in PRE or not
            # TODO Feed in dx and dt from config file
            """

        dx = torch.tensor(dx, dtype=torch.float32)
        dt = torch.tensor(dt, dtype=torch.float32)
        nu = torch.tensor(pde_param, dtype=torch.float32)

        # solutions are [bs, nx, nt, nc] but for PRE code we need [BS, Nt, nx]
        uu = uu.squeeze(-1) #last dimension is just one channel, squeeze out
        uu = uu.permute(0, 2, 1) #permute for correct PRE computation

        #Defining the required Convolutional Operations. 
        D_t = ConvOps_1d.ConvOperator(domain='t', order=1)
        D_x = ConvOps_1d.ConvOperator(domain='x', order=1)
        D_xx = ConvOps_1d.ConvOperator(domain='x', order=2)

        res = dx*D_t(uu) + dt * uu * D_x(uu) - nu / np.pi * D_xx(uu) * (2*dt/dx)
        if boundary:
            return res
        else: 
            return res[...,1:-1,1:-1]
    

    def train_single_epoch(self, current_epoch, total_epoch, num_epoch):
        self.model.train_single_epoch(current_epoch, total_epoch, num_epoch)

    def train_n_epoch(self, al_iter: int, num_epoch: int, step_offset: int, vis: bool = True, prefix: str = "") -> float:
        total_time = 0
        self.model.init_training(al_iter, load_train_data=True)
        for i in range(num_epoch):
            t = time.time()
            self.model.train_single_epoch(i, step_offset + i, num_epoch)
            total_time += time.time() - t
            if i % self.val_period == 0:
                self.model.validate(step_offset + i, prefix=prefix)
            if vis and ((i > 0 and i % self.model.vis_period == 0) or i == num_epoch - 1):
                self.model.visualize(step_offset + i)
        return total_time

    def uncertainty(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False, return_state=False):
        
        pred = self.model.roll_out(xx, grid, final_step, pde_param, t_idx, return_features)
        unc = self.residual(pred)
        if return_state:
            return pred, unc
        return unc


    def forward(self, xx, grid, pde_param=None, t_idx=None):
        return self.model(xx, grid, pde_param, t_idx)


def build_PREModel(task, cfg):
    return PREModel(task,  cfg.model_wrapper)