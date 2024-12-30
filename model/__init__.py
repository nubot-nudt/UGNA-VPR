from .network import Network
from .renderer import Renderer


def get_model(cfg):
    print(f"loading model \n")
    network = Network(cfg["network"])
    renderer = Renderer.init_from_cfg(cfg["renderer"])

    return network, renderer
