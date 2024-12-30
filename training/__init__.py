from .trainer import Trainer


def get_trainer(cfg):
    return Trainer.init_from_cfg(cfg)
