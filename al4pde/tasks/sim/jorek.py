import torch
import jax
import jax.numpy as jnp
from jax import device_put, lax
import numpy as np
from al4pde.tasks.sim.sim import Simulator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import tqdm
import h5py
import glob
import os
from typing import List, Tuple


def get_grid(data_path, n):
    """
    n : number of initial conditions
    """
    # Find any available JOREK file in the data path
    files = glob.glob(os.path.join(data_path, "*.h5"))
    if not files:
        raise FileNotFoundError(f"No JOREK files found in {data_path}")
    
    path_1 = files[0]  # Use the first available file
    fields, x, y = JOREK_electrostatic_single(path_1)

    print("x shape ", x.shape, "y shape", y.shape)
    grid = torch.stack([x, y], dim=-1)
    print("Grid shape ", grid.shape)

    # Add batch 
    if n:
        grid = grid.expand([n, ] + list(grid.shape)) 
        print("Grid shape ", grid.shape)

    grid = grid.to(device)  # Move to device
    return grid    # [bs, nx, ny, ]


def get_jorek_file_path_subfolders(jorek_data_dir, run_number):
    """
    Returns the full path to a JOREK output file based on the parent directory and run number.

    Assumes subdirectories are organized in ranges of 500 runs each, named like '0001-0500', '0501-1000', etc.,
    and that files are named 'jorek_runXXXX.h5' (zero-padded to 4 digits).

    Parameters:
        jorek_data_dir (str): Path to the directory containing run subdirectories.
        run_number (int): The run number to locate (e.g., 42).

    Returns:
        str: Full path to the corresponding 'jorek_runXXXX.h5' file.

    Raises:
        FileNotFoundError: If the expected file does not exist.
    """
    run_str = f"{run_number:04d}"

    # Compute range start and end
    group_start = ((run_number - 1) // 500) * 500 + 1
    group_end = group_start + 499

    subdir_name = f"{group_start:04d}-{group_end:04d}"
    subdir_path = os.path.join(jorek_data_dir, subdir_name)

    filename = f"jorek_run{run_str}.h5"
    filepath = os.path.join(subdir_path, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Expected file not found: {filepath}")

    return filepath

def get_jorek_file_path(jorek_data_dir, run_number):
    """
    Returns the full path to a JOREK output file based on the parent directory and run number.

    Parameters:
        jorek_data_dir (str): Path to the directory containing runs.
        run_number (str/int): The run number to locate (e.g., 42).

    Returns:
        str: Full path to the corresponding 'jorek_runXXXX.h5' file.

    Raises:
        FileNotFoundError: If the expected file does not exist.
    """

    if not isinstance(run_number, str):
        run_str = f"{run_number:04d}"
    else:
        run_str = run_number.zfill(4)

    filename = f"jorek_run{run_str}.h5"
    filepath = os.path.join(jorek_data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Expected file not found: {filepath}")

    return filepath


def stacked_fields(variables: List[np.ndarray]) -> torch.Tensor:
    """Stack list of field tensors into a single tensor with compatible dimensions.

        Args:
            variables (List[np.ndarray, np.ndarray, np.ndarray])- list containing 
                rho, phi and T values for a run
        
        Returns:
            torch.Tensor: Fields tensor of size (BS, Nx, Ny, Nt, Nc)"""

    stack = []
    for var in variables:
        var = torch.from_numpy(var) #Converting to Torch
        var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=-1) # BS, Nx, Ny, Nt, Nc
    return stack

def JOREK_electrostatic(data_loc: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load in JOREK electrostatic data from subfolders of data_loc.
        
        Args:
            data_loc (str) - path to where JOREK data is saved
        
        Returns:
             Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Torch tensors
                containing fields of dimensions BS, Nx, Ny, Nt, Nc with channels rho, phi, T,
                x grid, y grid and timestep. Runs from each file are stacked in batch dimension."""

    files = glob.glob(os.path.join(data_loc, "*.h5"))
    if not files:
        raise ValueError("No files found in ", data_loc)

    rho_array, phi_array, T_array = [], [], []
    for file_path in tqdm.tqdm(files, desc="Loading JOREK files"):
        with h5py.File(file_path, 'r') as f:
            rho = f['rho']
            phi = f['Phi']
            T = f['T']
            Rgrid = f['R_mesh(nR,nZ)']
            Zgrid = f['Z_mesh(nR,nZ)']

            rho_array.append(np.asarray(rho, dtype=np.float32))
            phi_array.append(np.asarray(phi, dtype=np.float32))
            T_array.append(np.asarray(T, dtype=np.float32))
            x = np.asarray(Rgrid, dtype=np.float32)
            y = np.asarray(Zgrid, dtype=np.float32)

    rho = np.asarray(rho_array)*10**-(20)
    phi = np.asarray(phi_array)
    T = np.asarray(T_array)

    print("Concatenating...")
    fields = stacked_fields([rho, phi, T])

    x, y = torch.tensor(x), torch.tensor(y)

    return fields, x, y

def JOREK_electrostatic_list(data_loc: str, ixs: list[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load in JOREK electrostatic data from subfolders of data_loc.
        
        Args:
            data_loc (str) - path to where JOREK data is saved
            ixs (list[int]) - run indices to be extracted
        
        Returns:
             Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Torch tensors
                containing fields of dimensions BS, Nx, Ny, Nt, Nc with channels rho, phi, T,
                x grid, y grid and timestep. Runs from each file are stacked in batch dimension."""

    rho_array, phi_array, T_array = [], [], []
    files = glob.glob(data_loc + folders[ii] + '/*.h5')

    for run_number in tqdm.tqdm(ixs, desc="Loading JOREK runs"):
        file_path = get_jorek_file_path(data_loc, run_number)

        with h5py.File(files[jj], 'r') as f:
            #keys = list(f.keys())
            rho = f['rho']
            phi = f['Phi']
            T = f['T']
            Rgrid = f['R_mesh(nR,nZ)']
            Zgrid = f['Z_mesh(nR,nZ)']

            rho_array.append(np.asarray(rho, dtype=np.float32))
            phi_array.append(np.asarray(phi, dtype=np.float32))
            T_array.append(np.asarray(T, dtype=np.float32))
            x = np.asarray(Rgrid, dtype=np.float32)
            y = np.asarray(Zgrid, dtype=np.float32)
    
    rho = np.asarray(rho_array)*10**-(20)
    phi = np.asarray(phi_array)
    T = np.asarray(T_array)

    print("Concatenating...")
    fields = stacked_fields([rho, phi, T])

    x, y = torch.tensor(x), torch.tensor(y)


    return fields, x, y

def JOREK_electrostatic_single(data_loc: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load in JOREK electrostatic data from file specified with path in data_loc.
        
        Args:
            data_loc (str) - path to file where JOREK data is saved
        
        Returns:
             Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Torch tensors
                containing fields of dimensions 1, Nx, Ny, Nt, Nc with channels rho, phi, T,
                x grid, y grid and timestep."""


    with h5py.File(data_loc, 'r') as f:
        #keys = list(f.keys())
        rho = np.asarray(f['rho'], dtype=np.float32)*10**-(20)
        phi = np.asarray(f['Phi'], dtype=np.float32)
        T = np.asarray(f['T'])
        Rgrid = f['R_mesh(nR,nZ)']
        Zgrid = f['Z_mesh(nR,nZ)']

        x = np.asarray(Rgrid, dtype=np.float32)
        y = np.asarray(Zgrid, dtype=np.float32)

        fields = stacked_fields([np.expand_dims(rho, axis=0), np.expand_dims(phi, axis=0), np.expand_dims(T, axis=0)])

        x, y = torch.tensor(x), torch.tensor(y)


    return fields, x, y

class JOREKSim(Simulator):
    """Class to load in presimulated data from the JOREK dataset (Neural-Parareal electrostatic Dataset 
        (Zenodo 10.5281/zenodo.11099659) to use for active learning experiments."""

    def __init__(self, pool_path):
        dt = 1.5e-6 #1.5 microseconds
        ini_time = 0
        fin_time = dt*200 #double check if not 200
        channel_names=["rho", "phi", "T"]
        super().__init__(pde_name="jorek", num_pde_params=1, spatial_dim=2, num_channels=3, dt=dt, ini_time=ini_time, fin_time=fin_time, channel_names=channel_names)
        self.pool_path = pool_path
        self.max_step = 200


    def n_step_sim(self, ic, ic_params):
        
        raise NotImplementedError #Shouldn't be used, as we load in pre-simulated data 


