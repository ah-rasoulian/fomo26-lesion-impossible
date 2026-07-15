from typing import Union, Sequence, List, Type, Optional, Tuple
import torch
import torch.nn as nn

from gardening_tools.modules.networks.components.heads import ClsRegHead
from .blocks.utils import _DropoutNd, MODALITY_TO_ID
from .blocks.encoders import SpacingModalityResidualBlock, MultiLayerSpacingModalityConvDropoutNormNonlin, PhysicalGaussianMaskedConv3d, SpacingModalityResidualUNetEncoder
from .blocks.decoders import SpacingModalityResidualUNetDecoder
from .sma_basenet import SpacingModalityAwareBaseNet


class SpacingModalityResidualEncoderUNet(SpacingModalityAwareBaseNet):
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


class SpacingModalityResidualEncoderUNetCLSREG(SpacingModalityAwareBaseNet):
    """
    Spacing- and modality-aware classification/regression network.

    Supports two fusion strategies:

    Early fusion:
        All modalities are processed jointly by the modality-aware encoder.

        x:
            [B, C, D, H, W]

    Late fusion:
        Each modality is processed independently using the same shared encoder.
        Encoder features are then concatenated along the channel dimension.

        x:
            [B, C, D, H, W]

        Internally:
            [B, C, D, H, W] -> [B*C, 1, D, H, W]

    Additional inputs:
        spacing:
            [B, 3] or [3], in [D, H, W] order.

        modality_ids:
            [B, C], [C], or [B] when C == 1.

        modality_mask:
            Optional [B, C] tensor:
                1 = modality is available
                0 = modality is missing or padded
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
        conv_bias: bool = True,
        encoder_basic_block: Type[nn.Module] = SpacingModalityResidualBlock,
        decoder: Type[nn.Module] = ClsRegHead,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        late_fusion: bool = False,
        modality_embed_dim: int = 32,
        modality_aggregation: str = "sqrt_sum",
        padding_modality_idx: Optional[int] = None,
        film_hidden_dim: int = 128,
        sigma_mm_per_stage: Optional[Sequence[float]] = None,
    ):
        super().__init__()

        if dimensions != "3D":
            raise ValueError(
                "SpacingModalityResidualEncoderUNetCLSREG is designed for "
                "3D tensors [B, C, D, H, W]. Use dimensions='3D'."
            )

        if input_channels < 1:
            raise ValueError(
                f"input_channels must be at least 1, got {input_channels}."
            )

        if norm_op_kwargs is None:
            norm_op_kwargs = {
                "eps": 1e-5,
                "affine": True,
            }

        if nonlin_kwargs is None:
            if nonlin is nn.LeakyReLU:
                nonlin_kwargs = {
                    "negative_slope": 1e-2,
                    "inplace": True,
                }
            else:
                nonlin_kwargs = {}

        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}

        # These follow the same convention as ResidualEncoderUNetCLSREG.
        encoder_dropout_rate = dropout_op_kwargs.get(
            "encoder_dropout_rate",
            dropout_op_kwargs.get("p", 0.0),
        )
        decoder_dropout_rate = dropout_op_kwargs.get(
            "decoder_dropout_rate",
            0.0,
        )
        dropout_inplace = dropout_op_kwargs.get("inplace", True)

        if encoder_dropout_rate > 0.0 and dropout_op is None:
            dropout_op = nn.Dropout3d

        encoder_dropout_kwargs = {
            "p": encoder_dropout_rate,
            "inplace": dropout_inplace,
        }

        conv_op = PhysicalGaussianMaskedConv3d
        norm_op = nn.InstanceNorm3d
        pool_op = nn.AvgPool3d
        clsreg_pool_op = nn.AdaptiveAvgPool3d

        self.num_classes = output_channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.late_fusion = late_fusion

        self.num_modalities = len(set(MODALITY_TO_ID.values()))
        self.modality_embed_dim = modality_embed_dim

        # With late fusion, the shared encoder receives one modality at a time.
        encoder_input_channels = 1 if late_fusion else input_channels

        self.encoder = SpacingModalityResidualUNetEncoder(
            input_channels=encoder_input_channels,
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
            dropout_op_kwargs=encoder_dropout_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            block=encoder_basic_block,
            modality_embed_dim=modality_embed_dim,
            modality_aggregation=modality_aggregation,
            padding_modality_idx=padding_modality_idx,
            film_hidden_dim=film_hidden_dim,
            sigma_mm_per_stage=sigma_mm_per_stage,
        )

        bottleneck_channels = features_per_stage[-1]

        if late_fusion:
            bottleneck_channels *= input_channels

        self.decoder = decoder(
            pool_op=clsreg_pool_op,
            input_channels=bottleneck_channels,
            output_channels=output_channels,
            dropout_rate=decoder_dropout_rate,
        )

        # Useful for loading pretrained weights.
        self.stem_weight_name = "encoder.modality_stem.value_proj.weight"

    @staticmethod
    def _prepare_spacing(
        spacing: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Convert spacing into a [B, 3] tensor.
        """

        spacing = torch.as_tensor(
            spacing,
            device=device,
            dtype=dtype,
        )

        if spacing.ndim == 1:
            if spacing.numel() != 3:
                raise ValueError(
                    "A one-dimensional spacing tensor must contain exactly "
                    f"three values, but got shape {tuple(spacing.shape)}."
                )

            spacing = spacing.unsqueeze(0).expand(batch_size, -1)

        elif spacing.ndim == 2:
            if spacing.shape[1] != 3:
                raise ValueError(
                    "spacing must have shape [B, 3], but got "
                    f"{tuple(spacing.shape)}."
                )

            if spacing.shape[0] == 1 and batch_size > 1:
                spacing = spacing.expand(batch_size, -1)

            elif spacing.shape[0] != batch_size:
                raise ValueError(
                    f"spacing batch size is {spacing.shape[0]}, while the "
                    f"image batch size is {batch_size}."
                )

        else:
            raise ValueError(
                "spacing must have shape [3] or [B, 3], but got "
                f"{tuple(spacing.shape)}."
            )

        return spacing

    @staticmethod
    def _prepare_modality_ids(
        modality_ids: torch.Tensor,
        batch_size: int,
        num_channels: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert modality IDs into a [B, C] tensor.
        """

        modality_ids = torch.as_tensor(
            modality_ids,
            device=device,
            dtype=torch.long,
        )

        if modality_ids.ndim == 0:
            if num_channels != 1:
                raise ValueError(
                    "A scalar modality ID is only valid for single-channel input."
                )

            modality_ids = modality_ids.view(1, 1).expand(batch_size, 1)

        elif modality_ids.ndim == 1:
            if modality_ids.shape[0] == num_channels:
                # One shared list of channel modality IDs for the batch.
                modality_ids = modality_ids.unsqueeze(0).expand(
                    batch_size,
                    -1,
                )

            elif num_channels == 1 and modality_ids.shape[0] == batch_size:
                # One modality ID per sample for single-channel input.
                modality_ids = modality_ids.unsqueeze(1)

            else:
                raise ValueError(
                    "One-dimensional modality_ids must have length C, or "
                    "length B when C == 1. Got shape "
                    f"{tuple(modality_ids.shape)} for B={batch_size}, "
                    f"C={num_channels}."
                )

        elif modality_ids.ndim == 2:
            if modality_ids.shape == (1, num_channels) and batch_size > 1:
                modality_ids = modality_ids.expand(batch_size, -1)

            elif modality_ids.shape != (batch_size, num_channels):
                raise ValueError(
                    "modality_ids must have shape [B, C]. Got "
                    f"{tuple(modality_ids.shape)} for B={batch_size}, "
                    f"C={num_channels}."
                )

        else:
            raise ValueError(
                "modality_ids must be a scalar, [C], [B] when C == 1, "
                f"or [B, C]. Got {tuple(modality_ids.shape)}."
            )

        return modality_ids

    @staticmethod
    def _prepare_modality_mask(
        modality_mask: Optional[torch.Tensor],
        batch_size: int,
        num_channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Convert modality mask into a [B, C] tensor.
        """

        if modality_mask is None:
            return torch.ones(
                batch_size,
                num_channels,
                device=device,
                dtype=dtype,
            )

        modality_mask = torch.as_tensor(
            modality_mask,
            device=device,
            dtype=dtype,
        )

        if modality_mask.ndim == 1:
            if modality_mask.shape[0] == num_channels:
                modality_mask = modality_mask.unsqueeze(0).expand(
                    batch_size,
                    -1,
                )

            elif num_channels == 1 and modality_mask.shape[0] == batch_size:
                modality_mask = modality_mask.unsqueeze(1)

            else:
                raise ValueError(
                    "One-dimensional modality_mask must have length C, or "
                    "length B when C == 1. Got shape "
                    f"{tuple(modality_mask.shape)}."
                )

        elif modality_mask.ndim == 2:
            if modality_mask.shape == (1, num_channels) and batch_size > 1:
                modality_mask = modality_mask.expand(batch_size, -1)

            elif modality_mask.shape != (batch_size, num_channels):
                raise ValueError(
                    "modality_mask must have shape [B, C]. Got "
                    f"{tuple(modality_mask.shape)} for B={batch_size}, "
                    f"C={num_channels}."
                )

        else:
            raise ValueError(
                "modality_mask must have shape [C], [B] when C == 1, "
                f"or [B, C]. Got {tuple(modality_mask.shape)}."
            )

        return modality_mask

    def _encode_early_fusion(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_ids: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Jointly encode all modalities.
        """

        skips = self.encoder(
            x=x,
            spacing=spacing,
            modality_ids=modality_ids,
            modality_mask=modality_mask,
        )

        # In case the encoder is configured to return additional metadata.
        if isinstance(skips, tuple):
            skips = skips[0]

        return skips

    def _encode_late_fusion(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_ids: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Encode each modality independently using a shared encoder, then
        concatenate corresponding feature levels along the channel dimension.
        """

        batch_size, num_channels = x.shape[:2]
        spatial_shape = x.shape[2:]

        # [B, C, D, H, W] -> [B*C, 1, D, H, W]
        x_flat = x.reshape(
            batch_size * num_channels,
            1,
            *spatial_shape,
        )

        # Each modality/sample pair receives its own spacing.
        # [B, 3] -> [B*C, 3]
        spacing_flat = spacing.repeat_interleave(
            num_channels,
            dim=0,
        )

        # [B, C] -> [B*C, 1]
        modality_ids_flat = modality_ids.reshape(
            batch_size * num_channels,
            1,
        )

        modality_mask_flat = modality_mask.reshape(
            batch_size * num_channels,
            1,
        )

        skips = self.encoder(
            x=x_flat,
            spacing=spacing_flat,
            modality_ids=modality_ids_flat,
            modality_mask=modality_mask_flat,
        )

        if isinstance(skips, tuple):
            skips = skips[0]

        fused_skips = []

        for feature in skips:
            # feature:
            #     [B*C, F, d, h, w]
            #
            # Restore modality dimension:
            #     [B, C, F, d, h, w]
            feature = feature.reshape(
                batch_size,
                num_channels,
                feature.shape[1],
                *feature.shape[2:],
            )

            # Explicitly suppress features belonging to missing modalities.
            #
            # This is important because convolutional biases, normalization,
            # and FiLM layers can otherwise produce nonzero features even when
            # the missing input channel contains zeros.
            feature_mask = modality_mask.view(
                batch_size,
                num_channels,
                1,
                1,
                1,
                1,
            )

            feature = feature * feature_mask

            # [B, C, F, d, h, w] -> [B, C*F, d, h, w]
            feature = feature.reshape(
                batch_size,
                num_channels * feature.shape[2],
                *feature.shape[3:],
            )

            fused_skips.append(feature)

        return fused_skips

    def _encode(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_ids: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Validate and normalize inputs, then run early or late fusion.
        """

        if x.ndim != 5:
            raise ValueError(
                "x must have shape [B, C, D, H, W], but got "
                f"{tuple(x.shape)}."
            )

        batch_size, num_channels = x.shape[:2]

        if num_channels != self.input_channels:
            raise ValueError(
                f"The model was created with input_channels="
                f"{self.input_channels}, but received {num_channels} channels."
            )

        spacing = self._prepare_spacing(
            spacing=spacing,
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )

        modality_ids = self._prepare_modality_ids(
            modality_ids=modality_ids,
            batch_size=batch_size,
            num_channels=num_channels,
            device=x.device,
        )

        modality_mask = self._prepare_modality_mask(
            modality_mask=modality_mask,
            batch_size=batch_size,
            num_channels=num_channels,
            device=x.device,
            dtype=x.dtype,
        )

        if self.late_fusion:
            return self._encode_late_fusion(
                x=x,
                spacing=spacing,
                modality_ids=modality_ids,
                modality_mask=modality_mask,
            )

        return self._encode_early_fusion(
            x=x,
            spacing=spacing,
            modality_ids=modality_ids,
            modality_mask=modality_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_ids: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        skips = self._encode(
            x=x,
            spacing=spacing,
            modality_ids=modality_ids,
            modality_mask=modality_mask,
        )

        return self.decoder(skips)

    def forward_with_features(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor,
        modality_ids: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ):
        skips = self._encode(
            x=x,
            spacing=spacing,
            modality_ids=modality_ids,
            modality_mask=modality_mask,
        )

        output = self.decoder(skips)
        bottleneck = skips[-1]

        return output, bottleneck

    def freeze_backbone(self):
        """
        Freeze the spacing- and modality-aware encoder for linear probing.
        """

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.encoder.eval()

    def unfreeze_backbone(self):
        """
        Unfreeze the encoder after linear probing.
        """

        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

        self.encoder.train()

    def load_state_dict(
        self,
        target_state_dict,
        *args,
        **kwargs,
    ):
        """
        Shape-safe partial state-dict loading.

        This permits loading compatible encoder weights from segmentation or
        classification checkpoints while ignoring incompatible prediction-head
        parameters.
        """

        current_state_dict = self.state_dict()

        filtered_state_dict = {
            key: value
            for key, value in target_state_dict.items()
            if key in current_state_dict
            and current_state_dict[key].shape == value.shape
        }

        return super().load_state_dict(
            filtered_state_dict,
            *args,
            **kwargs,
        )


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


def sma_resunet_s_clsreg(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    dropout_op_kwargs: dict = None,
    late_fusion: bool = False,
):
    # input_channels is the number of modalities (e.g. 2 for T1+T2).
    # Each modality is a single-channel volume, so the encoder always uses input_channels=1.
    return SpacingModalityResidualEncoderUNetCLSREG(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        kernel_size=3,
        stride=2,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        n_blocks_per_stage=(2, 2, 2, 2, 2, 2),
        dropout_op_kwargs=dropout_op_kwargs,
        late_fusion=late_fusion,
    )
