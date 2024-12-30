import torch
from torch import nn
from .code import PositionalEncoding
import torch.autograd.profiler as profiler
from dotmap import DotMap
from utils import util


def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.kaiming_normal_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name:
            continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.0)


class RayMarcher(nn.Module):
    def __init__(
        self,
        d_in,
        d_hidden,
        raymarch_steps,
        use_encoding,
        pe_config,
    ):
        """
        Inits LSTM ray marcher

        Args:
            d_in: input feature size.
            d_hidden: hidden feature size of LSTM cell.
            raymarch_steps: number of iteration
            use_encoding: whether use positional encoding for lstm input
            pe_config: positional encoding (pe) configuration
        """

        super().__init__()

        self.lstm_d_in = d_in
        self.lstm_d_hidden = d_hidden
        self.lstm_d_out = 1

        self.raymarch_steps = raymarch_steps
        self.use_encoding = use_encoding

        if use_encoding:
            self.positional_encoding = PositionalEncoding.init_from_cfg(pe_config)
            self.lstm_d_in += self.positional_encoding.d_out

        self.lstm = nn.LSTMCell(
            input_size=self.lstm_d_in, hidden_size=self.lstm_d_hidden
        )
        self.lstm.apply(init_recurrent_weights)
        lstm_forget_gate_init(self.lstm)
        self.out_layer = nn.Linear(self.lstm_d_hidden, self.lstm_d_out)
        nn.init.constant_(self.out_layer.bias, 0.0)
        nn.init.kaiming_normal_(self.out_layer.weight, a=0, mode="fan_in")

        self.sigmoid = nn.Sigmoid()

    def forward(self, network, rays):
        """
        Predicting surface sampling points using LSTM

        Args:
            network: network model
            rays: ray (ON, RB, 8)

        Returns:
            sample_points: predicted surface points (x, y, z) in world coordinate, (ON, RB, 3)
            depth: distance to the camera center point, is the depth prediction, (ON, RB, 1)
            depth_confidence: accumulated confidence of current depth prediction (ON. RB, 1)

        """
        with profiler.record_function("ray_marching"):
            ON, RB, _ = rays.shape
            RN = network.num_ref_views
            ray_dirs = rays[:, :, 3:6]  # (ON, RB, 3)
            z_near = rays[:, :, 6:7]
            z_far = rays[:, :, 7:8]
            z_scale = z_far - z_near

            scaled_depth = [0 * z_scale]  # scaled_depth should range from 0 -1
            sample_points = rays[:, :, :3] + z_near * ray_dirs  # (ON, RB, 3)
            states = [None]

            for _ in range(self.raymarch_steps):
                with torch.no_grad():
                    # print("1", sample_points.shape)
                    latent, p_feature = network.get_features(
                        sample_points, ray_dirs
                    )  # (ON*RN*RB, d_latent)

                    feature, weight = network.mlp_feature(
                        latent,
                        p_feature,
                    )
                    # (ON*RN*RB, d_feature)
                    lstm_feature = util.weighted_pooling(
                        feature, inner_dims=(RN, RB), weight=weight
                    ).reshape(
                        ON * RB, -1
                    )  # (ON*RB, 2*d_feature)

                state = self.lstm(lstm_feature, states[-1])  # (2, ON*RB, d_hidden)

                if state[0].requires_grad:
                    state[0].register_hook(lambda x: x.clamp(min=-5, max=5))

                states.append(state)

                lstm_out = self.out_layer(state[0]).view(
                    ON, RB, self.lstm_d_out
                )  # (ON, RB, 1)
                signed_distance = lstm_out
                depth_scaling = 1.0 / (1.0 * self.raymarch_steps)
                signed_distance = depth_scaling * signed_distance
                scaled_depth.append(self.sigmoid(scaled_depth[-1] + signed_distance))
                depth = scaled_depth[-1] * z_scale + z_near  # (ON, RB, 1)
                sample_points = rays[:, :, :3] + depth * ray_dirs  # (ON, RB, 3)

        return sample_points, depth, scaled_depth


class _RenderWrapper(nn.Module):
    def __init__(self, network, renderer):
        super().__init__()
        self.network = network
        self.renderer = renderer

    def forward(self, rays):
        outputs = self.renderer(self.network, rays)
        return outputs.toDict()


class Renderer(nn.Module):
    def __init__(
        self,
        d_in,
        d_hidden,
        raymarch_steps,
        trainable,
        use_encoding,
        pe_config,
    ):
        super().__init__()
        self.ray_marcher = RayMarcher(
            d_in, d_hidden, raymarch_steps, use_encoding, pe_config
        )
        self.is_trainable = trainable
        self.sigmoid = nn.Sigmoid()

    def forward(self, network, rays):
        """
        Ray marching rendering.

        Args:
            network: network model
            rays: ray (ON, RB, 8)

        Returns:
            render dict
        """

        with profiler.record_function("rendering"):
            assert len(rays.shape) == 3

            (
                sample_points,
                depth_final,
                scaled_depth,
            ) = self.ray_marcher(
                network, rays
            )  # (ON, RB, 3), (ON, RB, 1), (ON, RB, 1), (step, ON, RB, 1)
            render_dict = {
                "depth": depth_final,
                "scaled_depth": torch.stack(scaled_depth),
            }

            out = network(sample_points, rays[:, :, 3:6])  # (ON, RB, 4)

            logit_mean = out[:, :, :3]  # (ON, RB, 3)
            logit_log_var = out[:, :, 3:]  # (ON, RB , 3)

            render_dict["logit_mean"] = logit_mean
            render_dict["logit_log_var"] = logit_log_var

            with torch.no_grad():
                sampled_predictions = util.get_samples(
                    logit_mean, torch.sqrt(torch.exp(logit_log_var)), 100
                )
                rgb_mean = torch.mean(sampled_predictions, axis=0)
                rgb_std = torch.std(sampled_predictions, axis=0)
                render_dict["rgb"] = rgb_mean
                render_dict["uncertainty"] = torch.mean(rgb_std, dim=-1)

            return DotMap(render_dict)

    def parallelize(self, network, gpus=None):
        """
        Returns a wrapper module compatible with DataParallel.
        Specify a list of GPU ids in 'gpus' to apply DataParallel automatically.

        Args:
            network: network model
            gpus: list of GPU ids to parallize to. No parallelization if gpus length is 1.

        Returns:
            wrapper module
        """

        wrapped = _RenderWrapper(network, self)
        if gpus is not None and len(gpus) > 1:
            print("Using multi-GPU", gpus)
            wrapped = nn.DataParallel(wrapped, gpus, dim=1)
        return wrapped

    @classmethod
    def init_from_cfg(cls, cfg):
        return cls(
            d_in=cfg["d_in"],
            d_hidden=cfg["d_hidden"],
            raymarch_steps=cfg["raymarch_steps"],
            trainable=cfg["trainable"],
            use_encoding=cfg["use_encoding"],
            pe_config=cfg["positional_encoding"],
        )
