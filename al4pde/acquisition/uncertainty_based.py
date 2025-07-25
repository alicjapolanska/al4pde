import torch
import wandb
from torch.utils.data import TensorDataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from al4pde.acquisition.pool_based import PoolBased
from al4pde.prob_models.prob_model import ProbModel
from al4pde.prob_models.PRE_model import PREModel
import os


def top_k(unc: torch.Tensor, k: int) -> torch.Tensor:
    return torch.argsort(unc, descending=True)[:k]

def second_top_k(unc: torch.Tensor, k: int) -> torch.Tensor:
    return torch.argsort(unc, descending=True)[k:2*k]

def power_sampling(unc: torch.Tensor, k: int, beta=1) -> torch.Tensor:
    weights = torch.pow(unc, beta)
    prob = weights / weights.sum()
    return torch.multinomial(prob, k, replacement=False)


def random(n: int, k: int) -> torch.Tensor:
    weights = torch.ones(n)
    prob = weights / weights.sum()
    return torch.multinomial(prob, k, replacement=False)


class UncertaintyBased(PoolBased):

    def __init__(self, task, data_schedule, batch_size, pool_size, unc_eval_mode,
                 unc_num_rollout_steps_rel, selection_mode, power_beta=1, pred_batch_size=128):
        print("Instantiating UncertaintyBased")
        super().__init__(task, data_schedule, batch_size, pool_size, unc_eval_mode, unc_num_rollout_steps_rel,
                         pred_batch_size=pred_batch_size)
        print("Init done")
        self.selection_mode = selection_mode
        assert selection_mode in ["random", "top_k", "power", "second_top_k"]
        self.power_beta = power_beta

    def select_next(self, prob_model: ProbModel, ic_pool: torch.Tensor, pde_param_pool: torch.Tensor,
                    ic_train: torch.Tensor, pde_param_train: torch.Tensor, grid: torch.Tensor, al_iter: int,
                    train_loader=None, k=1) -> torch.Tensor:

        save_path = os.path.join(self.task.traj_save_path, "unc_pool" + str(al_iter) + ".pt")

        dataset = TensorDataset(ic_pool, pde_param_pool)
        unc = []
        #if self.selection_mode != "random":
        loader = DataLoader(dataset, batch_size=self.pred_batch_size)
        grid = grid.to(device)
        for i, batch in enumerate(loader):
            ic = batch[0].to(device)
            pde_param = batch[1].to(device)
            grid_batch = grid.expand([len(ic), ] + list(grid.shape))
            unc_batch = self.model_uncertainty(prob_model, ic, grid_batch, pde_param).detach().cpu()
            unc.append(unc_batch.reshape((len(unc_batch), -1)).mean(1))
        unc = torch.concat(unc, dim=0)

        if isinstance(prob_model, PREModel):

            pde_param_pool = pde_param_pool.to(device)
            pde_param_train = train_loader.dataset.pde_params.to(device)
            distances = torch.cdist(pde_param_pool, pde_param_train)
            knn_indices = torch.topk(distances, k, dim=1, largest=False).indices

            train_trajectories = train_loader.dataset.data
            
            knn_unc_means = []
            for indices in knn_indices:
                yy_knn = train_trajectories[indices.cpu()].to(device)
                param_knn = pde_param_train[indices.cpu()].to(device)
                res = prob_model.residual(yy_knn, param_knn)
                res_mean = res.reshape(res.shape[0], -1).abs().mean(1).mean()  # mean over batch, space, time
                knn_unc_means.append(res_mean)
            knn_unc_means = torch.stack(knn_unc_means).cpu()  # shape: [num_pool]

            del train_trajectories

            print("knn_unc_means device", knn_unc_means.device)
            print("unc device", unc.device)
            unc = unc / knn_unc_means

            del knn_unc_means

        torch.save(unc, save_path)


        n_samples = self.num_batches(al_iter) * self.batch_size
        if self.selection_mode == "top_k":
            sel_idx = top_k(unc, n_samples)
        elif self.selection_mode == "second_top_k":
            sel_idx = second_top_k(unc, n_samples)
        elif self.selection_mode == "power":
            sel_idx = power_sampling(unc, n_samples, self.power_beta)
        elif self.selection_mode == "random":
            sel_idx = random(len(ic_pool), n_samples)
        else:
            raise ValueError(self.selection_mode)
        if self.selection_mode != "random":
            wandb.log({"al/acq_avg_unc": unc[sel_idx].mean(), "al/al_iter": al_iter})
        return sel_idx

    @property
    def name(self):
        return self.selection_mode
