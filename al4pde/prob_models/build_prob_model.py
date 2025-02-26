from al4pde.prob_models.ensemble import build_ensemble
from al4pde.prob_models.PRE_model import build_PREModel

def build_prob_model(task, cfg):
    if cfg._target_ == "al4pde.prob_models.ensemble.Ensemble":
        return build_ensemble(task, cfg)
    if cfg._target_ == "al4pde.prob_models.PRE_model.PREModel":
        return build_PREModel(task, cfg.model)
    else:
        raise ValueError(f"Unknown prob_model target: {cfg._target_}")
