import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import re
from al4pde.utils import subsample_grid, subsample_trajectory
import al4pde.tasks.sim.jorek as jorek


def load_grid(folders, reduced_resolution):
    grid = None
    for folder in folders:
        # Define path to files
        root_path = os.path.abspath(folder)

        x_coord_fname = os.path.join(root_path, 'x_coordinate.npy')
        y_coord_fname = os.path.join(root_path, 'y_coordinate.npy')
        z_coord_fname = os.path.join(root_path, 'z_coordinate.npy')
        xy_coord_fname = os.path.join(root_path, 'xy_coordinate.npy')

        # 3D
        if os.path.exists(z_coord_fname) and os.path.exists(y_coord_fname) and  os.path.exists(x_coord_fname):
            _gridx = np.load(os.path.join(root_path, x_coord_fname))[:, np.newaxis]
            _gridy = np.load(os.path.join(root_path, y_coord_fname))[:, np.newaxis]
            _gridz = np.load(os.path.join(root_path, z_coord_fname))[:, np.newaxis]
            _gridx = torch.from_numpy(_gridx)
            _gridy = torch.from_numpy(_gridy)
            _gridz = torch.from_numpy(_gridz)
            X, Y, Z = torch.meshgrid(_gridx, _gridy, _gridz, indexing='ij')
            grid = torch.stack((X, Y, Z), dim=-1)

        # 2d
        elif os.path.exists(xy_coord_fname):
            _gridxy = np.load(os.path.join(root_path, xy_coord_fname))
            grid = torch.from_numpy(_gridxy)

        # 1d
        elif os.path.exists(x_coord_fname):
            _gridx = np.load(os.path.join(root_path, x_coord_fname))
            if len(_gridx.shape) == 1:
                _gridx = _gridx[:, np.newaxis]
            grid = torch.from_numpy(_gridx)
    if grid is None:
        raise IOError("no grid found in folders" , folders)

    grid = subsample_grid(grid.unsqueeze(0), reduced_resolution).float()[0]
    spatial_dim = len(grid.shape) - 1

    return grid, spatial_dim


def load_fdata(root_path, fname, pde_name, spatial_dim):
    fdata = np.load(os.path.join(root_path, fname))
    if len(fdata.shape) == spatial_dim + 2:
        fdata = fdata[..., None]  # assume squeezed channel dim

    fdata = fdata.transpose([0,] + list(range(2, len(fdata.shape) - 1)) + [1, -1])

    return fdata


class TrajDataset(Dataset):
    def __init__(self,
                 data,
                 pde_params,
                 grid,
                 initial_step=1,
                 num_steps=None,
                 ):

        self.initial_step = initial_step
        self.data = data
        self.pde_params = pde_params
        print("pde params shape when initialising TrajDataset:", pde_params.shape)
        self.grid = grid
        if num_steps is not None:
            self._set_num_steps(num_steps)
        else:
            self._set_num_steps(data.shape[-2] - 1)
        self.t = torch.arange(data.shape[-2]).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        #print("getting item: ", idx)
        ic = self.data[idx, ..., 0, :].unsqueeze(-2)
        #print("ic shape getting item: ", ic.shape)
        traj = self.data[idx, ...]
        #print("traj shape getting item: ", traj.shape)
        pde_params = self.pde_params[idx, ...]
        #print("item pde params getting item: ", pde_params)

        assert pde_params.numel() > 0, f"Empty pde_params at idx {idx}!"
        assert not torch.isnan(pde_params).any(), f"NaN in pde_params at idx {idx}!"

        return ic, traj, self.grid, pde_params, self.t[idx]

    def set_num_steps(self, num_steps):
        return TrajDataset(self.data, self.pde_params, self.grid, self.initial_step, num_steps)

    def _set_num_steps(self, num_steps):
        self.traj_len = num_steps + 1
        self.num_steps = num_steps
        self.num_sub_per_traj = (self.data.shape[-2] - num_steps)
        self.num_sub_trajs = self.num_sub_per_traj * self.data.shape[0]


class NPYDataset(TrajDataset):
    def __init__(self,
                 pde_name,
                 folders,
                 initial_step=1,
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 regexp=None,
                 max_size=None,
                 skip_initial_steps=0,
                 one_step=False,
                 ):

        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.reduced_batch = reduced_batch
        self.reduced_resolution_t = reduced_resolution_t
        self.reduced_resolution = reduced_resolution
        self.skip_initial_steps = skip_initial_steps

        if len(folders) == 1:
            folders = folders[0]
        print("Loading from ", folders)
        _data, temp, temp = jorek.JOREK_electrostatic(folders)
        _pde_par = torch.zeros((_data.shape[0], 1))  # JOREK has no pde params
        grid = jorek.get_grid(folders, 0)
        print("data shape:", _data.shape)
        super().__init__(_data, _pde_par, grid, initial_step)

        print("data shape:", _data.shape)
        print("pde par shape ", _pde_par.shape)
        print("pde par ", _pde_par)
        print("grid shape ", grid.shape)
        print("initial step", initial_step)
        print("number of trajectories in data:", len(_data))
