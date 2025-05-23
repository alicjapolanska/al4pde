import torch
from tensordict import TensorDict
from al4pde.tasks.ic_gen.ic_gen import ICGenerator
import al4pde.tasks.sim.jorek as jorek

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ICGenJOREK(ICGenerator):
    
    def __init__(self, train_data_path, requires_grad=True, single_fixed=False):
        super().__init__(requires_grad, single_fixed)
        self.data_path = data_path
        self.grid = self.get_grid()

    def get_grid(self, n):
        
        return jorek.get_grid(self.data_path, n)


    def generate_initial_conditions(self, ixs: list[int], pde_params = None) -> torch.Tensor:
        """Load in JOREK initial conditions parametrised by the run index, contained in ixs.
            
            Args:
                ixs (list[int]) - Run indices to be extracted

                pde_params - Only included for compatibility with hydra. Defaults to None 
                    as params are the same with all runs. Not used here.
            
            Returns:

             torch.Tensor Tensor containing fields of dimensions BS, Nx, Ny, Nt, Nc with channels rho, phi, T,
        """
        # Load in JOREK simulations from ixs
        fields, x, y = jorek.JOREK_electrostatic_list(self.data_path, ixs)

        # Extract first timestep
        u = fields[..., 0, :]
        
        return u   # [bs, nx, ny, 1, nc]

