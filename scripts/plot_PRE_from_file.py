import numpy as np
import sys
sys.path.append("/Users/alicjapolanska/Documents/ucl/ukaea/al4pde")
print(sys.path)

from al4pde.prob_models.PRE_model import PREModel
from al4pde.utils import load_checkpoint
from al4pde.prob_models.build_prob_model import build_prob_model

path_to_models = "plots/cnr1ahtx/checkpoints"
num_al_iter = 4


model = PREModel()

for i in range(num_al_iter-1):

    load_checkpoint(path_to_models, str(i), model)

    print(model.current_ground_truths)