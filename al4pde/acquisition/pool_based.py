import wandb
import time
import torch
import numpy as np
from al4pde.acquisition.batch_selection import BatchSelection
from al4pde.prob_models.prob_model import ProbModel
import os
import re
import tqdm
import psutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_mem_usage():
    process = psutil.Process(os.getpid())
    mem_MB = process.memory_info().rss / 1024 ** 2
    print(f"Memory used: {mem_MB:.1f} MB")

print_mem_usage()

class PoolBased(BatchSelection):
    """Base class for all pool based approaches."""

    def __init__(self, task, data_schedule, batch_size, pool_size, unc_eval_mode,
                 unc_num_rollout_steps_rel, pred_batch_size=128):
        super().__init__(task, data_schedule, batch_size, unc_eval_mode,
                         unc_num_rollout_steps_rel)

        self.pred_batch_size = pred_batch_size

        ic_params = []

        pattern = re.compile(r"jorek_run(\d+)\.h5")

        print("Loading pool ics from", self.task.pool_path)

        pool_files = [f for f in os.listdir(self.task.pool_path) if pattern.match(f)]
        num_files = len(pool_files)
        print(f"Found {num_files} files to load")

        # Load first file to get IC shape and dtype/device
        first_param = int(pattern.match(pool_files[0]).group(1))
        first_ic = self.task.get_ic(first_param)
        ic_shape = first_ic.shape
        ic_dtype = first_ic.dtype
        ic_device = first_ic.device

        all_ics = torch.empty((num_files, *ic_shape), dtype=ic_dtype, device=ic_device)

        for idx, filename in enumerate(tqdm.tqdm(pool_files)):

            match = pattern.match(filename)
            if match:
                param = int(match.group(1))
                current_ic = self.task.get_ic(param)
                all_ics[idx] = current_ic
                ic_params.append(param)

        print("Loaded", len(ic_params), "initial conditions from pool")

        self.ic = all_ics
        self.ic_params = torch.tensor(ic_params)
        self.pool_mask = torch.ones((len(self.ic),), dtype=torch.bool)
        self.batch_size = batch_size
        self.pde_params = torch.zeros(len(self.ic))
        self.pde_params_normed = torch.zeros(len(self.ic))

    def prepare_data(self, train_loader):
        """ Prepares pool and train inputs."""
        pde_params = []
        ics = []
        for batch_idx, (xx, yy, grid, param, t_idx) in enumerate(train_loader):
            print("Batch", batch_idx, "of", len(train_loader), "with", len(xx), "samples")
            print("Parameters:", param)
            pde_params.append(param)
            ics.append(xx)
        ics_train = torch.concat(ics, 0)
        pde_params_train = torch.concat(pde_params, 0)
        ic_pool = self.ic[self.pool_mask]
        print("pde params ", self.pde_params)
        pde_params_pool = self.pde_params[self.pool_mask]
        print("ic params ", self.ic_params)
        print("pool mask ", self.pool_mask)
        if self.pde_params_normed is not None:
            ic_params_pool = self.ic_params[self.pool_mask]
            pde_params_normed_pool = self.pde_params_normed[self.pool_mask]
        else:
            ic_params_pool = None
            pde_params_normed_pool = None

        if len(ic_pool) == 0:
            raise Exception("Pool is empty")

        return ics_train, pde_params_train, ic_pool, pde_params_pool, pde_params_normed_pool, ic_params_pool

    def generate(self, prob_model: ProbModel, al_iter: int, train_loader: torch.utils.data.DataLoader)-> None:
        with torch.no_grad():

            t = time.time()

            (ics_train, pde_params_train, ic_pool, pde_params_pool, pde_params_normed_pool,
             ic_params_pool) = self.prepare_data(train_loader)

            sel_idx = self.select_next(prob_model, ic_pool, pde_params_pool, ics_train, pde_params_train,
                                       self.task.get_grid()[0], al_iter, train_loader)

            print(len(sel_idx), len(self.pool_mask[self.pool_mask]))
            new_mask = self.pool_mask.clone()[self.pool_mask]
            new_mask[sel_idx] = False

            self.pool_mask[self.pool_mask.clone()] = new_mask

            sel_ic = ic_pool[sel_idx]
            sel_pde_params = pde_params_pool[sel_idx]
            sel_ic_params = ic_params_pool[sel_idx]
            sel_pde_params_normed = pde_params_normed_pool[sel_idx]

            print("selection time", time.time() - t)
            wandb.log({"al/al_iter": al_iter, "al/sel_time": time.time() - t})

            self.simulate_all(prob_model, sel_ic, sel_pde_params, al_iter, ic_params=sel_ic_params,
                              pde_params_normed=sel_pde_params_normed)

    def select_next(self, prob_model: ProbModel, ic_pool: torch.Tensor, pde_param_pool: torch.Tensor,
                    ic_train: torch.Tensor, pde_param_train: torch.Tensor, grid: torch.Tensor, al_iter: int,
                    train_loader: torch.utils.data.DataLoader=None) -> torch.Tensor:
        """Returns the index of the selected pool samples."""
        raise NotImplementedError
