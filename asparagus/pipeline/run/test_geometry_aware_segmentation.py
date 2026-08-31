from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
from asparagus.modules.networks.blocks.layers.physical_conv3d import PhysicalGaussianMaskedConv3d
from asparagus.modules.networks.vit_task_model import LightUNetDecoder


ROOT = Path(__file__).resolve().parent

def test_physical_projection_metadata() -> None:
    layer = PhysicalGaussianMaskedConv3d(
        in_channels=1,
        out_channels=4,
        kernel_size=2,
        grid_size=(4, 4, 4),
        offset_mode="center",
    )
    image = torch.randn(2, 1, 10, 16, 16)
    output = layer(
        image,
        spacing=torch.ones(2, 3),
        valid_spatial_shapes=torch.tensor(((10, 16, 16), (6, 8, 10))),
        return_metadata=True,
    )
    assert output.features.shape == (2, 4, 4, 4, 4)
    assert output.original_spatial_shapes.tolist() == [
        [10, 16, 16],
        [6, 8, 10],
    ]
    assert output.processed_spatial_shapes.tolist() == [
        [8, 8, 8],
        [6, 8, 8],
    ]
    assert output.stride_vox.tolist() == [[2, 2, 2], [1, 2, 2]]
    assert output.effective_kernel_size_vox.tolist() == [[2, 2, 2], [2, 2, 2]]
    assert output.interpolation_applied.tolist() == [True, True]


def test_heterogeneous_geometry_aware_decoder() -> None:
    decoder = LightUNetDecoder(
        embed_dim=8,
        output_channels=2,
        hidden_dim=32,
        dropout=0.0,
    )
    feature_maps = tuple(
        torch.randn(2, 8, 4, 4, 4, requires_grad=True)
        for _ in range(4)
    )
    logits = decoder(
        feature_maps=feature_maps,
        output_size=(10, 16, 16),
        original_spatial_shapes=torch.tensor(((10, 16, 16), (6, 8, 10))),
        processed_spatial_shapes=torch.tensor(((8, 8, 8), (6, 8, 8))),
        stride_vox=torch.tensor(((2, 2, 2), (1, 2, 2))),
        effective_kernel_size_vox=torch.tensor(((2, 2, 2), (2, 2, 2))),
        padding_vox=torch.tensor(
            (
                ((0, 0), (0, 0), (0, 0)),
                ((0, 0), (0, 0), (0, 0)),
            )
        ),
        cropping_vox=torch.tensor(
            (
                ((0, 0), (0, 0), (0, 0)),
                ((0, 1), (0, 0), (0, 0)),
            )
        ),
    )
    assert logits.shape == (2, 2, 10, 16, 16)
    assert torch.isfinite(logits).all()
    # Subject 1 is reconstructed to 6x8x10 before high-end batch padding.
    assert torch.count_nonzero(logits[1, :, 6:]) == 0
    assert torch.count_nonzero(logits[1, :, :, 8:]) == 0
    assert torch.count_nonzero(logits[1, :, :, :, 10:]) == 0
    logits.square().mean().backward()
    assert all(feature.grad is not None for feature in feature_maps)


if __name__ == "__main__":
    test_physical_projection_metadata()
    test_heterogeneous_geometry_aware_decoder()
    print("All geometry-aware segmentation tests passed.")
