from typing import Union, Sequence, List, Type, Optional, Tuple
import torch
import torch.nn as nn

from gardening_tools.modules.networks.BaseNet import BaseNet
from .blocks.utils import _DropoutNd, MODALITY_TO_ID
from .blocks.encoders import SpacingModalityResidualBlock, MultiLayerSpacingModalityConvDropoutNormNonlin, PhysicalGaussianMaskedConv3d, SpacingModalityResidualUNetEncoder
from .blocks.decoders import SpacingModalityResidualUNetDecoder


class SpacingModalityResidualEncoderUNet(BaseNet):
    """
    Full spacing + modality-aware residual encoder-decoder U-Net.

    Supports:
        - single-channel MRI
        - multi-channel MRI
        - missing modalities via modality_mask
        - channel-level modality embeddings
        - spacing-aware Gaussian masked convolutions
        - spacing-aware encoder and decoder
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: str,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int], Sequence[Sequence[int]]],
        features_per_stage: List[int],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = True,
        deep_supervision: bool = False,
        encoder_basic_block: Type[nn.Module] = SpacingModalityResidualBlock,
        decoder_basic_block: Type[nn.Module] = MultiLayerSpacingModalityConvDropoutNormNonlin,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        use_skip_connections: bool = True,
        modality_embed_dim: int = 32,
        modality_aggregation: str = "sqrt_sum",
        padding_modality_idx: Optional[int] = None,
        film_hidden_dim: int = 128,
        sigma_mm_per_stage: Optional[Sequence[float]] = None,
    ):
        super().__init__()

        if dimensions != "3D":
            raise ValueError(
                "This spacing-aware implementation is designed for 3D tensors "
                "[B, C, D, H, W]. Use dimensions='3D'."
            )

        conv_op = PhysicalGaussianMaskedConv3d
        final_conv_op = nn.Conv3d
        norm_op = nn.InstanceNorm3d
        upsample_op = nn.ConvTranspose3d
        pool_op = nn.AvgPool3d

        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}

        if nonlin_kwargs is None:
            if nonlin is nn.LeakyReLU:
                nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
            else:
                nonlin_kwargs = {}

        self.num_classes = output_channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_modalities = len(set(MODALITY_TO_ID.values()))
        self.modality_embed_dim = modality_embed_dim

        # Useful if you load pretrained weights by stem name.
        self.stem_weight_name = "encoder.modality_stem.value_proj.weight"

        self.encoder = SpacingModalityResidualUNetEncoder(
            input_channels=input_channels,
            features_per_stage=features_per_stage,
            num_modalities=self.num_modalities,
            conv_op=conv_op,
            pool_op=pool_op,
            kernel_size=kernel_size,
            stride=stride,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            block=encoder_basic_block,
            modality_embed_dim=modality_embed_dim,
            modality_aggregation=modality_aggregation,
            padding_modality_idx=padding_modality_idx,
            film_hidden_dim=film_hidden_dim,
            sigma_mm_per_stage=sigma_mm_per_stage,
        )

        # Encoder strides are in high-to-low order:
        #   [stage0_stride, stage1_stride, ..., stageN_stride]
        #
        # Decoder needs them reversed, excluding stage0.
        # Example:
        #   encoder strides: [(1,1,1), (1,2,2), (1,2,2), (2,2,2)]
        #   decoder strides: [(2,2,2), (1,2,2), (1,2,2)]
        decoder_strides = self.encoder.strides_per_stage[:0:-1]

        if sigma_mm_per_stage is None:
            sigma_mm_per_stage = [1.5 * (2 ** s) for s in range(len(features_per_stage))]

        decoder_sigma_mm = list(sigma_mm_per_stage)[::-1]

        self.decoder = SpacingModalityResidualUNetDecoder(
            output_channels=output_channels,
            features_per_stage=features_per_stage[::-1],
            n_conv_per_stage=n_conv_per_stage_decoder,
            modality_embed_dim=modality_embed_dim,
            basic_block=decoder_basic_block,
            deep_supervision=deep_supervision,
            conv_op=conv_op,
            conv_kwargs={
                "kernel_size": kernel_size,
                "bias": conv_bias,
            },
            stride_for_transpose_conv=decoder_strides,
            upsample_op=upsample_op,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            use_skip_connections=use_skip_connections,
            film_hidden_dim=film_hidden_dim,
            sigma_mm_per_stage=decoder_sigma_mm,
            final_conv_op=final_conv_op,
        )

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_ids: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ):
        """
        x:
            [B, C, D, H, W]

        spacing:
            [B, 3] or [3], in [D, H, W] order.

        modality_ids:
            [B, C], [C], or [B] if C == 1.

        modality_mask:
            optional [B, C], 1 for available channel, 0 for missing/padded channel.
        """

        skips, skip_spacings, modality_context = self.encoder(
            x=x,
            spacing=spacing,
            modality_ids=modality_ids,
            modality_mask=modality_mask,
            return_spacings=True,
            return_modality_context=True,
        )

        return self.decoder(
            skips=skips,
            skip_spacings=skip_spacings,
            modality_context=modality_context,
        )

    def forward_with_features(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_ids: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ):
        skips, skip_spacings, modality_context = self.encoder(
            x=x,
            spacing=spacing,
            modality_ids=modality_ids,
            modality_mask=modality_mask,
            return_spacings=True,
            return_modality_context=True,
        )

        output = self.decoder(
            skips=skips,
            skip_spacings=skip_spacings,
            modality_context=modality_context,
        )

        bottleneck = skips[-1]

        return output, bottleneck

    def load_state_dict(self, target_state_dict, *args, **kwargs):
        """
        Shape-safe loading, same idea as your BaseNet.
        Useful when loading partial pretrained weights.
        """

        current_state_dict = self.state_dict()

        filtered_state_dict = {
            k: v
            for k, v in target_state_dict.items()
            if k in current_state_dict
            and current_state_dict[k].shape == v.shape
        }

        super().load_state_dict(filtered_state_dict, *args, **kwargs)


# Encoder 29M parameters
# Full model 42M parameters
# This is the "classic" unet, but with residual encoder blocks
def sma_resunet_s(
    dimensions,
    input_channels,
    output_channels,
    deep_supervision=False,
    use_skip_connections=True,
):
    return SpacingModalityResidualEncoderUNet(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        kernel_size=3,
        stride=2,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        n_blocks_per_stage=(2, 2, 2, 2, 2, 2),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        deep_supervision=deep_supervision,
        use_skip_connections=use_skip_connections,
    )
