import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Sequence, Optional, Type
from torch.nn.modules.dropout import _DropoutNd

from .utils import normalize_list, to_3tuple

def interpolate_to_shape(
    x: torch.Tensor,
    target_shape: Tuple[int, int, int],
    mode: str = "trilinear",
) -> torch.Tensor:
    if tuple(x.shape[2:]) == tuple(target_shape):
        return x

    if mode == "trilinear":
        return F.interpolate(
            x,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )

    return F.interpolate(x, size=target_shape, mode="nearest")


class SpacingModalityResidualUNetDecoder(nn.Module):
    """
    Spacing + modality-aware U-Net decoder.

    Input:
        skips:
            list of encoder features in high-to-low order:
                [stage0, stage1, ..., bottleneck]

        skip_spacings:
            list of effective spacings for each skip:
                [spacing0, spacing1, ..., bottleneck_spacing]

        modality_context:
            [B, modality_embed_dim]

    Output:
        logits or deep-supervision logits.
    """

    def __init__(
        self,
        output_channels: int,
        features_per_stage: List[int],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        modality_embed_dim: int,
        basic_block: Type[nn.Module],
        conv_op: Type[nn.Module],
        conv_kwargs: dict,
        stride_for_transpose_conv: Union[
            int,
            Sequence[int],
            Sequence[Sequence[int]],
        ],
        upsample_op: Type[nn.Module] = nn.ConvTranspose3d,
        deep_supervision: bool = False,
        norm_op: Optional[Type[nn.Module]] = nn.InstanceNorm3d,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        use_skip_connections: bool = True,
        film_hidden_dim: int = 128,
        sigma_mm_per_stage: Optional[Sequence[float]] = None,
        final_conv_op: Type[nn.Module] = nn.Conv3d,
    ):
        super().__init__()

        self.num_classes = output_channels
        self.features_per_stage = list(features_per_stage)
        self.deep_supervision = deep_supervision
        self.use_skip_connections = use_skip_connections
        self.modality_embed_dim = modality_embed_dim

        n_stages = len(self.features_per_stage)

        if n_stages < 2:
            raise ValueError("Decoder requires at least two stages.")

        n_decoder_stages = n_stages - 1

        n_conv_per_stage = normalize_list(
            n_conv_per_stage,
            n_decoder_stages,
            "n_conv_per_stage",
        )

        if (
            isinstance(stride_for_transpose_conv, (list, tuple))
            and len(stride_for_transpose_conv) == n_decoder_stages
            and all(isinstance(s, (list, tuple)) for s in stride_for_transpose_conv)
        ):
            up_strides = [to_3tuple(s) for s in stride_for_transpose_conv]
        else:
            up_strides = [to_3tuple(stride_for_transpose_conv)] * n_decoder_stages

        if sigma_mm_per_stage is None:
            sigma_mm_per_stage = [1.5 * (2 ** i) for i in range(n_stages)]
        else:
            sigma_mm_per_stage = normalize_list(
                sigma_mm_per_stage,
                n_stages,
                "sigma_mm_per_stage",
            )

        self.up_strides = up_strides

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        for d in range(n_decoder_stages):
            in_ch = self.features_per_stage[d]
            skip_ch = self.features_per_stage[d + 1]
            out_ch = skip_ch
            up_stride = up_strides[d]

            self.upsamples.append(
                upsample_op(
                    in_ch,
                    out_ch,
                    kernel_size=up_stride,
                    stride=up_stride,
                )
            )

            block_in_ch = out_ch * (2 if use_skip_connections else 1)

            conv_kwargs_d = dict(conv_kwargs)
            conv_kwargs_d.setdefault("init_sigma_mm", sigma_mm_per_stage[d + 1])

            self.decoder_blocks.append(
                basic_block(
                    input_channels=block_in_ch,
                    output_channels=out_ch,
                    modality_embed_dim=modality_embed_dim,
                    num_layers=n_conv_per_stage[d],
                    conv_op=conv_op,
                    conv_kwargs=conv_kwargs_d,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    film_hidden_dim=film_hidden_dim,
                    stage_index=float(n_decoder_stages - d),
                )
            )

            self.seg_layers.append(
                final_conv_op(
                    out_ch,
                    output_channels,
                    kernel_size=1,
                )
            )

        self.out_conv = self.seg_layers[-1]

    def forward(
        self,
        skips: List[torch.Tensor],
        skip_spacings: List[torch.Tensor],
        modality_context: torch.Tensor,
    ):
        """
        skips:
            encoder features in high-to-low order.

        skip_spacings:
            effective spacings corresponding to skips.

        modality_context:
            pooled modality embedding from encoder stem.
        """

        if len(skips) != len(skip_spacings):
            raise ValueError(
                f"Expected same number of skips and spacings. "
                f"Got {len(skips)} skips and {len(skip_spacings)} spacings."
            )

        n_stages = len(skips)

        x = skips[-1]
        current_spacing = skip_spacings[-1]

        decoder_outputs = []

        for d in range(n_stages - 1):
            skip_index = n_stages - 2 - d

            skip = skips[skip_index]
            target_spacing = skip_spacings[skip_index]

            x = self.upsamples[d](x)

            # ConvTranspose can overshoot by one voxel for odd input sizes.
            x = interpolate_to_shape(x, tuple(skip.shape[2:]))

            # After upsampling, the feature grid should match the skip grid.
            current_spacing = target_spacing

            if self.use_skip_connections:
                x = torch.cat([x, skip], dim=1)

            x = self.decoder_blocks[d](
                x=x,
                spacing=current_spacing,
                modality_context=modality_context,
            )

            decoder_outputs.append(self.seg_layers[d](x))

        if self.deep_supervision and torch.is_grad_enabled():
            # Return high-resolution output first, then lower-resolution outputs.
            return decoder_outputs[::-1]

        return decoder_outputs[-1]
