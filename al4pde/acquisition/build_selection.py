import hydra
from al4pde.acquisition.distance_pool_based import build_distance_pool_based


def build_strategy(task, cfg):
    if cfg._target_ == "al4pde.acquisition.distance_pool_based.DistancePoolBased":
        return build_distance_pool_based(task, cfg)
    else:
        cfg.data_schedule = hydra.utils.instantiate(cfg.data_schedule)
        return hydra.utils.instantiate(cfg, task=task)
