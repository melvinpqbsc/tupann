"""
Adaptive Fourier Neural Operator (AFNO) for 2D fields.

Based on Guibas et al., "Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers" (ICLR 2022)
and the FourCastNet implementation (Pathak et al., 2022). The input lags are stacked along the channel dimension,
split into patches, mixed in the Fourier domain by block-diagonal MLPs and decoded back to all lead times at once.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(F.gelu(self.fc1(x)))
        return self.drop(self.fc2(x))


class AFNO2D(nn.Module):
    """
    Fourier token mixer: FFT over the token grid, block-diagonal complex MLP per frequency, soft-shrinkage, inverse FFT.

    Args:
        hidden_size: channel dimension of the tokens.
        num_blocks: number of diagonal blocks of the channel-mixing weights.
        sparsity_threshold: soft-shrinkage threshold applied to the frequency activations.
        hard_thresholding_fraction: fraction of the lowest frequencies that are kept.
        hidden_size_factor: expansion ratio of the frequency MLP.
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        hidden_size_factor: int = 1,
    ):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} must be divisible by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor

        scale = 0.02
        bs, hidden = self.block_size, self.block_size * hidden_size_factor
        # index 0: real part, index 1: imaginary part
        self.w1 = nn.Parameter(scale * torch.randn(2, num_blocks, bs, hidden))
        self.b1 = nn.Parameter(scale * torch.randn(2, num_blocks, hidden))
        self.w2 = nn.Parameter(scale * torch.randn(2, num_blocks, hidden, bs))
        self.b2 = nn.Parameter(scale * torch.randn(2, num_blocks, bs))

    def forward(self, x):
        # x: (B, H, W, C)
        bias = x
        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
        h_slice = slice(total_modes - kept_modes, total_modes + kept_modes)
        w_slice = slice(None, kept_modes)
        xk = x[:, h_slice, w_slice]

        o1_real = F.relu(
            torch.einsum("...bi,bio->...bo", xk.real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", xk.imag, self.w1[1])
            + self.b1[0]
        )
        o1_imag = F.relu(
            torch.einsum("...bi,bio->...bo", xk.imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", xk.real, self.w1[1])
            + self.b1[1]
        )
        o2_real = (
            torch.einsum("...bi,bio->...bo", o1_real, self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag, self.w2[1])
            + self.b2[0]
        )
        o2_imag = (
            torch.einsum("...bi,bio->...bo", o1_imag, self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real, self.w2[1])
            + self.b2[1]
        )

        o2 = torch.zeros(B, H, W // 2 + 1, self.num_blocks, self.block_size, 2, device=x.device)
        o2[:, h_slice, w_slice] = F.softshrink(torch.stack([o2_real, o2_imag], dim=-1), lambd=self.sparsity_threshold)
        x = torch.view_as_complex(o2).reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")

        return x.type(dtype) + bias


class AFNOBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        double_skip: bool = True,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.filter(self.norm1(x))
        if self.double_skip:
            x = x + residual
            residual = x
        x = self.mlp(self.norm2(x))
        x = self.drop_path(x)
        return x + residual


class AFNONet(nn.Module):
    """
    Args:
        img_size: (H, W) of the input fields; must be divisible by patch_size.
        patch_size: side of the square patches.
        in_channels: number of input channels (stacked input lags).
        out_channels: number of output channels (stacked lead times).
        embed_dim: token dimension.
        depth: number of AFNO blocks.
    """

    def __init__(
        self,
        img_size: tuple,
        patch_size: int,
        in_channels: int,
        out_channels: int,
        embed_dim: int = 256,
        depth: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ):
        super().__init__()
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0, f"Image size {img_size} not divisible by {patch_size}"
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.h = H // patch_size
        self.w = W // patch_size

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.h, self.w, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                AFNOBlock(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    num_blocks=num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_channels * patch_size * patch_size, bias=False)

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (B, C_in, H, W) -> tokens (B, h, w, D)
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.head(self.norm(x))
        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels,
        )
        return x
