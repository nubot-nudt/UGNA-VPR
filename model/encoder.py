import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.autograd.profiler as profiler
from utils import util



class Encoder(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=3,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        use_first_pool=True,
        norm_type="batch",
    ):
        """
        Inits Encoder instance.

        Args:
            backbone: encoder model resnest34.
            pretrained: whether to use model weights pretrained on ImageNet.
            num_layers: number of resnet layers to use (1-5).
            index_interp: interpolation to use for feature map indexing.
            index_padding: padding mode to use for indexing (border, zeros, reflection).
            upsample_interp: interpolation to use for upscaling latent code.
            use_first_pool: whether to use first maxpool layer.
            norm_type: norm type to use. usually "batch"
        """

        super().__init__()
        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.use_first_pool = use_first_pool
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        norm_layer = util.get_norm_layer(norm_type)
        self.model = getattr(torchvision.models, backbone)(
            pretrained=pretrained, norm_layer=norm_layer
        )

    def forward(self, x):
        """
        Get feature maps of RGB image inputs.

        Args:
            x: image (RN, C, H, W).

        Returns:
            latent: features (RN, L, H, W), L is feature map channel length.
        """

        # with profiler.record_function("encoder_inference"):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        latents = [x]
        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.model.maxpool(x)
            x = self.model.layer1(x)
            latents.append(x)
        if self.num_layers > 2:
            x = self.model.layer2(x)
            latents.append(x)
        if self.num_layers > 3:
            x = self.model.layer3(x)
            latents.append(x)
        if self.num_layers > 4:
            x = self.model.layer4(x)
            latents.append(x)

        align_corners = None if self.index_interp == "nearest " else True
        latent_sz = latents[0].shape[-2:]

        # unpsample feature map from different layers to the same dimension
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode=self.upsample_interp,
                align_corners=align_corners,
            )
        latent = torch.cat(latents, dim=1)
        return latent

    @classmethod
    def init_from_cfg(cls, cfg):
        return cls(
            backbone=cfg["backbone"],
            pretrained=cfg["pretrained"],
            num_layers=cfg["num_layers"],
            index_interp=cfg["index_interp"],
            index_padding=cfg["index_padding"],
            upsample_interp=cfg["upsample_interp"],
            use_first_pool=cfg["use_first_pool"],
            norm_type=cfg["norm_type"],
        )
if __name__ == '__main__':
    tea = Encoder()
    inputs = torch.rand((1, 3, 224, 224))
    outputs_tea = tea(inputs)
    print(outputs_tea.shape)
