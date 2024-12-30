from .dl3dv import DL3DVDataModule
from .shapenet import ShapenetDataModule
from .cambridge import CamDataModule
from .NEU import NEUDataModule
from .SIASUN import SIASUNDataModule
def get_data(cfg):
    dataset_name = cfg["name"]

    if dataset_name == "NEU":
        print(f"loading NEU dataset \n")
        return NEUDataModule(cfg)
    elif dataset_name == "Cambridge":
        print(f"loading Cambridge dataset \n")
        return CamDataModule(cfg)
    elif dataset_name == "SIASUN":
        print(f"loading Cambridge dataset \n")
        return SIASUNDataModule(cfg)
    elif dataset_name == "shapenet":
        print(f"loading shapenet dataset \n")
        return ShapenetDataModule(cfg)
    else:
        RuntimeError("dataset is not implemeneted!")
