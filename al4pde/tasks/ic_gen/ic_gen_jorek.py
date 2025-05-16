import torch
from tensordict import TensorDict
from al4pde.tasks.ic_gen.ic_gen import ICGenerator
from al4pde.tasks.sim.jorek import JOREK_electrostatic_single, get_jorek_file_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ICGenJOREK(ICGenerator):
    
    def __init__(self, train_data_path, requires_grad=True, single_fixed=False):
        super().__init__(requires_grad, single_fixed)
        self.data_path = data_path
        self.grid = self.get_grid()

    def get_grid(self, n):
        path_0 = get_jorek_file_path(self.data_path, 0)
        fields, gridx, gridy = JOREK_electrostatic_single(path_0)

        print("x shape ", gridx.shape, "y shape", gridy.shape)

        grid = torch.stack([gridx, gridy], dim=-1)
        print("Grid shape ", grid.shape)

        # Add batch 
        grid = grid.expand([n, ] + list(grid.shape)) 
        print("Grid shape ", grid.shape)

        return grid.expand([n, ] + list(grid.shape))     # [bs, nx, ny, nc]


        

    def generate_initial_conditions(self, ixs: list[int], pde_params = None) -> torch.Tensor:
        
        for ix in ixs:

            
        
        
        
        return u   # [bs, nx, nt, nc]

