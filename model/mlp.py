import torch
from torch import nn
import torch.autograd.profiler as profiler
from .code import PositionalEncoding


def km_init(l):
    if isinstance(l, nn.Linear):
        nn.init.kaiming_normal_(l.weight, a=0, mode="fan_in")
        nn.init.constant_(l.bias, 0.0)


class MLPFeature(nn.Module):
    def __init__(
        self, d_latent, d_feature, block_num, use_encoding, pe_config, use_view
    ):
        """
        Inits MLP_feature model.

        Args:
            d_latent: encoder latent size.
            d_feature: mlp feature size.
            use_encoding: whether use positional encoding
            pe_config: configuration for positional encoding
        """

        super().__init__()

        self.d_latent = d_latent
        self.d_pose = 3  # (x, y, z, vx, vy, vz)
        self.d_feature = d_feature
        self.block_num = block_num
        self.use_view = use_view

        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.use_encoding = use_encoding
        if self.use_encoding:
            self.positional_encoding = PositionalEncoding.init_from_cfg(pe_config)
            self.d_pose = self.positional_encoding.d_out

        if self.use_view:
            self.d_pose += 3

        self.lin_in_p = nn.Sequential(
            nn.Linear(self.d_pose, self.d_feature), self.activation
        )
        self.out_feat = nn.Sequential(
            nn.Linear(self.d_feature, self.d_feature), self.activation
        )
        self.out_weight = nn.Sequential(
            nn.Linear(self.d_feature, self.d_feature), self.sigmoid
        )

        self.lin_in_p.apply(km_init)
        self.out_feat.apply(km_init)

        self.blocks = nn.ModuleList()
        self.lin_in_z = nn.ModuleList()
        for _ in range(self.block_num):
            lin_z = nn.Sequential(
                nn.Linear(self.d_latent, self.d_feature), self.activation
            )
            lin_z.apply(km_init)
            self.lin_in_z.append(lin_z)

            self.blocks.append(ResnetBlock(self.d_feature))

    def forward(self, z, x):
        if self.use_encoding:
            p = self.positional_encoding(x[..., :3])
            if self.use_view:
                p = torch.cat((p, x[..., 3:]), dim=-1)
        #print(p.dtype)
        p=p.to(torch.float32)
        p = self.lin_in_p(p)
        for i in range(self.block_num):
            tz = self.lin_in_z[i](z)
            p = p + tz
            p = self.blocks[i](p)

        out = self.out_feat(p)  # (ON*RN*RB, d_feature)
        weight = self.out_weight(p)  # (ON*RN*RB, 1)

        return out, weight

    @classmethod
    def init_from_cfg(cls, cfg):
        return cls(
            d_latent=cfg["d_latent"],
            d_feature=cfg["d_feature"],
            use_encoding=cfg["use_encoding"],
            use_view=cfg["use_view"],
            block_num=cfg["block_num"],
            pe_config=cfg["positional_encoding"],
        )


class MLPOut(nn.Module):
    def __init__(self, d_feature, d_out, block_num):
        """
        Inits MLP_out model.

        Args:
            d_feature: feature size.
            d_out: output size.
            block_num: number of Resnet blocks.
        """

        super().__init__()
        self.d_feature = d_feature
        self.d_out = d_out
        self.block_num = block_num

        self.lin_out = nn.Linear(self.d_feature, self.d_out)

        self.blocks = nn.ModuleList()
        for _ in range(self.block_num):
            self.blocks.append(ResnetBlock(self.d_feature))

    def forward(self, x):
        for blkid in range(self.block_num):
            x = self.blocks[blkid](x)

        out = self.lin_out(x)
        return out

    @classmethod
    def init_from_cfg(cls, cfg):
        return cls(
            d_feature=cfg["d_feature"],
            block_num=cfg["block_num"],
            d_out=cfg["d_out"],
        )


class ResnetBlock(nn.Module):
    """
    Fully connected ResNet Block class.
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
        """
        Inits Resnet block.

        Args:
            size_in: input dimension.
            size_out: output dimension.
            size_h: hidden dimension.
        """

        super().__init__()

        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)
        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        self.fc_0.apply(km_init)
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            self.shortcut.apply(km_init)

    def forward(self, x):
        with profiler.record_function("resblock"):
            res = self.fc_0(x)
            res = self.activation(res)
            res = self.fc_1(res)

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            out = self.activation(x_s + res)
            return out


