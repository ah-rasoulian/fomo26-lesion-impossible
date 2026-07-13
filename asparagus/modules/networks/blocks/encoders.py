from typing import List, Optional, Tuple, Union, Type, Sequence

import torch
import torch.nn as nn
from .layers.physical_conv3d import PhysicalGaussianMaskedConv3d
from .layers.film import ChannelModalityEmbeddingStem
from .utils import _DropoutNd, normalize_list, normalize_stage_strides, Spacing, expand_spacing
from .sma_blocks import (SpacingModalityResidualBlock, MultiLayerSpacingModalityConvDropoutNormNonlin,
                         StackedSpacingModalityResidualBlocks)


class SpacingModalityResidualUNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_modalities: int,
        conv_op: Type[nn.Module] = PhysicalGaussianMaskedConv3d,
        pool_op: Type[nn.Module] = nn.AvgPool3d,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int], Sequence[Sequence[int]]] = 2,
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]] = 2,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = nn.InstanceNorm3d,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        block: Type[SpacingModalityResidualBlock] = SpacingModalityResidualBlock,
        modality_embed_dim: int = 32,
        modality_aggregation: str = "sqrt_sum",
        padding_modality_idx: Optional[int] = None,
        film_hidden_dim: int = 128,
        sigma_mm_per_stage: Optional[Sequence[float]] = None,
    ):
        super().__init__()

        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage]

        features_per_stage = list(features_per_stage)
        n_stages = len(features_per_stage)

        n_blocks_per_stage = normalize_list(
            n_blocks_per_stage,
            n_stages,
            "n_blocks_per_stage",
        )

        strides_per_stage = normalize_stage_strides(stride, n_stages)

        if sigma_mm_per_stage is None:
            sigma_mm_per_stage = [1.5 * (2 ** s) for s in range(n_stages)]
        else:
            sigma_mm_per_stage = normalize_list(
                sigma_mm_per_stage,
                n_stages,
                "sigma_mm_per_stage",
            )

        if nonlin_kwargs is None and nonlin is nn.LeakyReLU:
            nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}

        if norm_op_kwargs is None and norm_op is nn.InstanceNorm3d:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}

        stem_channels = features_per_stage[0]

        self.modality_stem = ChannelModalityEmbeddingStem(
            input_channels=input_channels,
            output_channels=stem_channels,
            num_modalities=num_modalities,
            modality_embed_dim=modality_embed_dim,
            aggregation=modality_aggregation,
            padding_idx=padding_modality_idx,
            bias=conv_bias,
        )

        self.stem = MultiLayerSpacingModalityConvDropoutNormNonlin(
            input_channels=stem_channels,
            output_channels=stem_channels,
            modality_embed_dim=modality_embed_dim,
            num_layers=1,
            conv_op=conv_op,
            conv_kwargs={
                "kernel_size": kernel_size,
                "stride": 1,
                "bias": conv_bias,
                "init_sigma_mm": sigma_mm_per_stage[0],
            },
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            film_hidden_dim=film_hidden_dim,
            stage_index=0.0,
        )

        input_channels_stage = stem_channels

        stages = []
        for s in range(n_stages):
            stage = StackedSpacingModalityResidualBlocks(
                n_blocks=n_blocks_per_stage[s],
                conv_op=conv_op,
                pool_op=pool_op,
                input_channels=input_channels_stage,
                output_channels=features_per_stage[s],
                modality_embed_dim=modality_embed_dim,
                kernel_size=kernel_size,
                initial_stride=strides_per_stage[s],
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                block=block,
                film_hidden_dim=film_hidden_dim,
                stage_index=float(s),
                sigma_mm=sigma_mm_per_stage[s],
            )

            stages.append(stage)
            input_channels_stage = features_per_stage[s]

        self.stages = nn.ModuleList(stages)

        self.features_per_stage = features_per_stage
        self.strides_per_stage = strides_per_stage
        self.modality_embed_dim = modality_embed_dim

    def forward(
        self,
        x: torch.Tensor,
        spacing: Spacing,
        modality_ids: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        return_spacings: bool = False,
        return_modality_context: bool = False,
    ):
        """
        x:
            [B, C, D, H, W]

        spacing:
            [B, 3] or [3], order [D, H, W]

        modality_ids:
            [B, C], [C], or [B] if C=1

        modality_mask:
            optional [B, C], 1 for available channel, 0 for missing/padded channel

        Returns:
            ret by default:
                list of stage features, same style as ResidualUNetEncoder

            optionally:
                ret, ret_spacings
                ret, ret_spacings, modality_context
        """

        B = x.shape[0]
        spacing = expand_spacing(spacing, B, x.device)

        x, modality_context = self.modality_stem(
            x=x,
            modality_ids=modality_ids,
            modality_mask=modality_mask,
        )

        x = self.stem(
            x=x,
            spacing=spacing,
            modality_context=modality_context,
        )

        ret = []
        ret_spacings = []

        for stage in self.stages:
            x, spacing = stage(x, spacing, modality_context)
            ret.append(x)
            ret_spacings.append(spacing)

        if return_spacings and return_modality_context:
            return ret, ret_spacings, modality_context

        if return_spacings:
            return ret, ret_spacings

        if return_modality_context:
            return ret, modality_context

        return ret
