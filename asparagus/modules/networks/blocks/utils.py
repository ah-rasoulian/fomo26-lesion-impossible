import math
from typing import List, Tuple, Union, Sequence, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import _DropoutNd
from asparagus.functional.loading import MODALITY_TO_ID


Spacing = Union[torch.Tensor, Sequence[float]]

def to_3tuple(x) -> Tuple[int, int, int]:
    if isinstance(x, int):
        return (x, x, x)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if len(x) != 3:
        raise ValueError(f"Expected 3 values, got {x}")
    return tuple(int(v) for v in x)

def expand_spacing(
    spacing: Spacing,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    spacing = torch.as_tensor(spacing, dtype=torch.float32, device=device)

    if spacing.ndim == 1:
        if spacing.numel() != 3:
            raise ValueError(f"spacing must have 3 values, got {spacing}")
        spacing = spacing[None, :].expand(batch_size, 3)

    elif spacing.ndim == 2:
        if spacing.shape != (batch_size, 3):
            raise ValueError(
                f"spacing must be [B, 3]. Got {spacing.shape}, "
                f"expected {(batch_size, 3)}"
            )

    else:
        raise ValueError(f"spacing must be [3] or [B, 3], got {spacing.shape}")

    return spacing


def inverse_sigmoid(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def make_norm(
    norm_op: Optional[Type[nn.Module]],
    channels: int,
    norm_op_kwargs: Optional[dict],
) -> nn.Module:
    if norm_op is None:
        return nn.Identity()

    kwargs = {} if norm_op_kwargs is None else dict(norm_op_kwargs)

    if norm_op is nn.GroupNorm:
        kwargs.setdefault("num_groups", min(8, channels))
        kwargs["num_channels"] = channels
        return norm_op(**kwargs)

    return norm_op(channels, **kwargs)


def make_dropout(
    dropout_op: Optional[Type[_DropoutNd]],
    dropout_op_kwargs: Optional[dict],
) -> nn.Module:
    if dropout_op is None:
        return nn.Identity()

    kwargs = {} if dropout_op_kwargs is None else dict(dropout_op_kwargs)
    return dropout_op(**kwargs)


def make_nonlin(
    nonlin: Optional[Type[nn.Module]],
    nonlin_kwargs: Optional[dict],
) -> nn.Module:
    if nonlin is None:
        return nn.Identity()

    kwargs = {} if nonlin_kwargs is None else dict(nonlin_kwargs)
    return nonlin(**kwargs)


def normalize_stage_strides(
    stride: Union[int, Sequence[int], Sequence[Sequence[int]]],
    n_stages: int,
) -> List[Tuple[int, int, int]]:
    """
    Supports:
        stride=2
        stride=(1, 2, 2)
        stride=[(1,1,1), (1,2,2), (1,2,2), (2,2,2)]
    """

    if (
        isinstance(stride, (list, tuple))
        and len(stride) == n_stages
        and all(isinstance(s, (list, tuple)) for s in stride)
    ):
        return [to_3tuple(s) for s in stride]

    base_stride = to_3tuple(stride)
    return [(1, 1, 1)] + [base_stride for _ in range(n_stages - 1)]


def normalize_list(x, n: int, name: str):
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError(f"{name} must have length {n}, got {len(x)}")
        return list(x)
    return [x] * n


def match_spatial_shape(x: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(x.shape[2:]) == tuple(target_shape):
        return x
    return F.interpolate(x, size=target_shape, mode="nearest")
