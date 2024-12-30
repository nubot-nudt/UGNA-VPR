import torch
from torch import nn
import numpy as np
import torch.autograd.profiler as profiler


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs, d_in, include_input, freq_factor):
        """
        Init poistional encoding instance.

        Args:
            num_freqs: frequency level for positional encoding.
            d_in: input dimension, by dedault should be 3 for x, y ,z or 6 for x y z vx, vy, vz.
            include_input: whether use pure input embedding vector.
            freq_factor: coefficient for positional encoding.
        """

        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in

        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding.

        Args:
            x: (batch, self.d_in), pose information.

        Returns:
            embed: (batch, self.d_out), postional embedding.
        """

        # with profiler.record_function("positional_encoding"):
        embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
        embed=embed.to(self._phases.dtype)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        embed = embed.view(x.shape[0], -1)
        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)
        return embed

    @classmethod
    def init_from_cfg(cls, cfg):
        return cls(
            num_freqs=cfg["num_freqs"],
            d_in=cfg["d_in"],
            include_input=cfg["include_input"],
            freq_factor=cfg["freq_factor"],
        )
