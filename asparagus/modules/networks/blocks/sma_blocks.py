from typing import Optional, Type, Tuple, Union, Sequence, List
import torch
import torch.nn as nn

from .utils import _DropoutNd, to_3tuple, make_nonlin, expand_spacing, make_dropout, make_norm, match_spatial_shape
from .layers.film import SpacingModalityContextFiLM
from .layers.physical_conv3d import PhysicalGaussianMaskedConv3d


class SpacingModalityConvDropoutNormNonlin(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        modality_embed_dim: int,
        conv_op: Type[nn.Module] = PhysicalGaussianMaskedConv3d,
        conv_kwargs: Optional[dict] = None,
        norm_op: Optional[Type[nn.Module]] = nn.InstanceNorm3d,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        film_hidden_dim: int = 128,
        stage_index: float = 0.0,
    ):
        super().__init__()

        conv_kwargs = {} if conv_kwargs is None else dict(conv_kwargs)

        self.conv = conv_op(
            input_channels,
            output_channels,
            **conv_kwargs,
        )

        self.dropout = make_dropout(dropout_op, dropout_op_kwargs)
        self.norm = make_norm(norm_op, output_channels, norm_op_kwargs)

        self.film = SpacingModalityContextFiLM(
            channels=output_channels,
            modality_embed_dim=modality_embed_dim,
            hidden_dim=film_hidden_dim,
            stage_index=stage_index,
        )

        self.nonlin = make_nonlin(nonlin, nonlin_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_context: torch.Tensor,
        film_spacing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        spacing:
            spacing of the input feature grid, used by the convolution mask.

        film_spacing:
            spacing of the output feature grid, used by FiLM.
            This matters when the convolution has stride > 1.
        """

        x = self.conv(x, spacing)
        x = self.dropout(x)
        x = self.norm(x)

        if film_spacing is None:
            film_spacing = spacing

        x = self.film(x, film_spacing, modality_context)
        x = self.nonlin(x)

        return x

class MultiLayerSpacingModalityConvDropoutNormNonlin(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        modality_embed_dim: int,
        num_layers: int = 2,
        conv_op: Type[nn.Module] = PhysicalGaussianMaskedConv3d,
        conv_kwargs: Optional[dict] = None,
        norm_op: Optional[Type[nn.Module]] = nn.InstanceNorm3d,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        film_hidden_dim: int = 128,
        stage_index: float = 0.0,
    ):
        super().__init__()

        assert num_layers >= 1, f"num_layers must be at least 1, got {num_layers}"

        self.layers = nn.ModuleList()

        self.layers.append(
            SpacingModalityConvDropoutNormNonlin(
                input_channels=input_channels,
                output_channels=output_channels,
                modality_embed_dim=modality_embed_dim,
                conv_op=conv_op,
                conv_kwargs=conv_kwargs,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                film_hidden_dim=film_hidden_dim,
                stage_index=stage_index,
            )
        )

        for _ in range(1, num_layers):
            self.layers.append(
                SpacingModalityConvDropoutNormNonlin(
                    input_channels=output_channels,
                    output_channels=output_channels,
                    modality_embed_dim=modality_embed_dim,
                    conv_op=conv_op,
                    conv_kwargs=conv_kwargs,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    film_hidden_dim=film_hidden_dim,
                    stage_index=stage_index,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_context: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, spacing, modality_context)
        return x

class SpacingModalityResidualBlock(nn.Module):
    def __init__(
        self,
        conv_op: Type[nn.Module],
        pool_op: Type[nn.Module],
        input_channels: int,
        output_channels: int,
        modality_embed_dim: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = nn.InstanceNorm3d,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        film_hidden_dim: int = 128,
        stage_index: float = 0.0,
        sigma_mm: float = 1.5,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = to_3tuple(stride)

        self.has_stride = self.stride != (1, 1, 1)
        self.requires_projection = input_channels != output_channels

        self.register_buffer(
            "stride_tensor",
            torch.tensor(self.stride, dtype=torch.float32)[None, :],
        )

        self.conv1 = SpacingModalityConvDropoutNormNonlin(
            input_channels=input_channels,
            output_channels=output_channels,
            modality_embed_dim=modality_embed_dim,
            conv_op=conv_op,
            conv_kwargs={
                "kernel_size": kernel_size,
                "stride": self.stride,
                "bias": conv_bias,
                "init_sigma_mm": sigma_mm,
            },
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            film_hidden_dim=film_hidden_dim,
            stage_index=stage_index,
        )

        self.conv2 = SpacingModalityConvDropoutNormNonlin(
            input_channels=output_channels,
            output_channels=output_channels,
            modality_embed_dim=modality_embed_dim,
            conv_op=conv_op,
            conv_kwargs={
                "kernel_size": kernel_size,
                "stride": 1,
                "bias": conv_bias,
                "init_sigma_mm": sigma_mm,
            },
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=None,
            nonlin_kwargs=None,
            film_hidden_dim=film_hidden_dim,
            stage_index=stage_index,
        )

        self.nonlin2 = make_nonlin(nonlin, nonlin_kwargs)

        if self.has_stride:
            self.skip_pool = pool_op(kernel_size=self.stride, stride=self.stride)
        else:
            self.skip_pool = nn.Identity()

        if self.requires_projection:
            self.skip_proj = SpacingModalityConvDropoutNormNonlin(
                input_channels=input_channels,
                output_channels=output_channels,
                modality_embed_dim=modality_embed_dim,
                conv_op=conv_op,
                conv_kwargs={
                    "kernel_size": 1,
                    "stride": 1,
                    "bias": False,
                    "init_sigma_mm": sigma_mm,
                },
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=None,
                dropout_op_kwargs=None,
                nonlin=None,
                nonlin_kwargs=None,
                film_hidden_dim=film_hidden_dim,
                stage_index=stage_index,
            )
        else:
            self.skip_proj = None

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = x.shape[0]
        spacing = expand_spacing(spacing, B, x.device)

        spacing_after = spacing * self.stride_tensor.to(
            device=spacing.device,
            dtype=spacing.dtype,
        )

        residual = self.skip_pool(x)

        if self.skip_proj is not None:
            residual = self.skip_proj(
                residual,
                spacing_after,
                modality_context,
                film_spacing=spacing_after,
            )

        out = self.conv1(
            x,
            spacing,
            modality_context,
            film_spacing=spacing_after,
        )

        out = self.conv2(
            out,
            spacing_after,
            modality_context,
            film_spacing=spacing_after,
        )

        residual = match_spatial_shape(residual, tuple(out.shape[2:]))

        out = out + residual
        out = self.nonlin2(out)

        return out, spacing_after

class StackedSpacingModalityResidualBlocks(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        conv_op: Type[nn.Module],
        pool_op: Type[nn.Module],
        input_channels: int,
        output_channels: Union[int, List[int], Tuple[int, ...]],
        modality_embed_dim: int,
        kernel_size: Union[int, Sequence[int]],
        initial_stride: Union[int, Sequence[int]],
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = nn.InstanceNorm3d,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        block: Type[SpacingModalityResidualBlock] = SpacingModalityResidualBlock,
        film_hidden_dim: int = 128,
        stage_index: float = 0.0,
        sigma_mm: float = 1.5,
    ):
        super().__init__()

        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks

        self.blocks = nn.ModuleList()

        self.blocks.append(
            block(
                conv_op=conv_op,
                pool_op=pool_op,
                input_channels=input_channels,
                output_channels=output_channels[0],
                modality_embed_dim=modality_embed_dim,
                kernel_size=kernel_size,
                stride=initial_stride,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                film_hidden_dim=film_hidden_dim,
                stage_index=stage_index,
                sigma_mm=sigma_mm,
            )
        )

        for n in range(1, n_blocks):
            self.blocks.append(
                block(
                    conv_op=conv_op,
                    pool_op=pool_op,
                    input_channels=output_channels[n - 1],
                    output_channels=output_channels[n],
                    modality_embed_dim=modality_embed_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    film_hidden_dim=film_hidden_dim,
                    stage_index=stage_index,
                    sigma_mm=sigma_mm,
                )
            )

        self.output_channels = output_channels[-1]

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        for block in self.blocks:
            x, spacing = block(x, spacing, modality_context)

        return x, spacing
