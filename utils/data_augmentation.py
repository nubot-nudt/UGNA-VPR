import torchvision.transforms.functional as TF
import numpy as np


class ColorJitterTransform:
    """
    use the same color jittering for images of one scene
    """

    def __init__(self, cfg):
        hue_range = cfg["hue_range"]
        saturation_range = cfg["saturation_range"]
        brightness_range = cfg["brightness_range"]
        contrast_range = cfg["contrast_range"]
        self.hue_range = [-hue_range, hue_range]
        self.saturation_range = [1 - saturation_range, 1 + saturation_range]
        self.brightness_range = [1 - brightness_range, 1 + brightness_range]
        self.contrast_range = [1 - contrast_range, 1 + contrast_range]

    def __call__(self, images):
        hue_factor = np.random.uniform(*self.hue_range)
        saturation_factor = np.random.uniform(*self.saturation_range)
        brightness_factor = np.random.uniform(*self.brightness_range)
        contrast_factor = np.random.uniform(*self.contrast_range)

        for i in range(len(images)):
            tmp = (images[i] + 1.0) * 0.5
            tmp = TF.adjust_saturation(tmp, saturation_factor)
            tmp = TF.adjust_hue(tmp, hue_factor)
            tmp = TF.adjust_contrast(tmp, contrast_factor)
            tmp = TF.adjust_brightness(tmp, brightness_factor)
            images[i] = tmp * 2.0 - 1.0
        return images


def get_transformation(cfg):
    augmentations = list(cfg.keys())
    transformations = []
    for aug in augmentations:
        if aug == "color_jitter":
            cfg_aug = cfg["color_jitter"]
            transformations.append(ColorJitterTransform(cfg_aug))

    return transformations
