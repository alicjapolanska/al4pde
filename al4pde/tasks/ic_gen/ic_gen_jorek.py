import torch
from tensordict import TensorDict
from al4pde.tasks.ic_gen.ic_gen import ICGenerator
import al4pde.tasks.sim.jorek as jorek
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ICGenJOREK(ICGenerator):
    
    def __init__(self, pool_path, requires_grad=True, single_fixed=False):
        super().__init__(requires_grad, single_fixed)
        self.pool_path = pool_path
        self.grid = self.get_grid(1)

    def get_grid(self, n):
        
        return jorek.get_grid(self.pool_path, n)


    def generate_initial_conditions(self, ic_param: TensorDict, pde_params = None) -> torch.Tensor:
        """Load in JOREK initial conditions parametrised by the run index, contained in ixs.
            
            Args:
                ic_param (int) - Run index to be extracted

                pde_params - Only included for compatibility with hydra. Defaults to None 
                    as params are the same with all runs. Not used here.
            
            Returns:

             torch.Tensor Tensor containing fields of dimensions BS, Nx, Ny, Nt, Nc with channels rho, phi, T,
        """
        
        file_path = jorek.get_jorek_file_path(self.pool_path, ic_param)

        # Load in JOREK simulations from ixs
        fields, x, y = jorek.JOREK_electrostatic_single(file_path)

        # Extract first timestep
        u = fields[..., 0, :]
        
        return u   # [bs, nx, ny, 1, nc]

