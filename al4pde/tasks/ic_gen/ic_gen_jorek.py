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
        fields, x, y = JOREK_electrostatic_single(path_0)
        

    def generate_initial_conditions(self, ic_params: TensorDict, pde_params: torch.Tensor) -> torch.Tensor:
        
        
        
        return u   # [bs, nx, nt, nc]

