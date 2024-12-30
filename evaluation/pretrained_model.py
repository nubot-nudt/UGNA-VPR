import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import get_model
import warnings

warnings.filterwarnings("ignore")


class PretrainedModel:
    def __init__(self, model_config, checkpoint_file, device, gpu_id):
        self.device = device
        self.network, self.renderer = self.load_pretrained_model(
            model_config, checkpoint_file
        )
        self.renderer_par = self.renderer.parallelize(self.network, gpu_id).eval()

    def load_pretrained_model(self, model_config, checkpoint_file):
        print("------ configure model ------")

        network, renderer = get_model(model_config)

        network = network.to(self.device).eval()
        renderer = renderer.to(self.device).eval()

        print("------ load model parameters ------")

        network.load_state_dict(checkpoint_file["network_state_dict"])
        #renderer.load_state_dict(checkpoint_file["renderer_state_dict"])

        return network, renderer
