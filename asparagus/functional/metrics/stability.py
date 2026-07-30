"""
Training stability and monitoring metrics for SSL pretraining.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _contains_nan(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isnan(value).any().item())

    if isinstance(value, Mapping):
        return any(_contains_nan(item) for item in value.values())

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_nan(item) for item in value)

    return False


def _contains_inf(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isinf(value).any().item())

    if isinstance(value, Mapping):
        return any(_contains_inf(item) for item in value.values())

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_inf(item) for item in value)

    return False


def compute_on_backward(model: nn.Module, grad_clip_val: Optional[float] = None) -> Dict[str, float]:
    """Metrics computed after backward pass."""
    return compute_gradient_metrics(model, grad_clip_val)

def compute_nan_inf_metrics(
    loss: Optional[torch.Tensor] = None,
    pred: Optional[Any] = None,
    activations: Optional[Any] = None,
    model: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Monitor tensors for NaN and Inf values.

    `pred` and `activations` may be tensors or nested containers of tensors,
    such as the per-crop outputs produced during DINO/iBOT pretraining.
    """
    metrics: Dict[str, float] = {}

    if loss is not None:
        metrics["nan_loss"] = float(_contains_nan(loss))
        metrics["inf_loss"] = float(_contains_inf(loss))

    if pred is not None:
        metrics["nan_predictions"] = float(_contains_nan(pred))
        metrics["inf_predictions"] = float(_contains_inf(pred))

    if activations is not None:
        metrics["nan_activations"] = float(_contains_nan(activations))
        metrics["inf_activations"] = float(_contains_inf(activations))

    if model is not None:
        has_nan_grad = False
        has_inf_grad = False

        for parameter in model.parameters():
            if parameter.grad is None:
                continue

            has_nan_grad |= _contains_nan(parameter.grad)
            has_inf_grad |= _contains_inf(parameter.grad)

            if has_nan_grad and has_inf_grad:
                break

        metrics["nan_gradients"] = float(has_nan_grad)
        metrics["inf_gradients"] = float(has_inf_grad)

    return metrics


def compute_gradient_metrics(model: nn.Module, grad_clip_value: Optional[float] = None) -> Dict[str, float]:
    """
    Gradient flow diagnostics: norm, clipping frequency, and layer-wise statistics.
    High clipping frequency (>0.5) suggests learning rate or architecture issues.
    """
    total_norm = 0.0
    num_parameters = 0
    gradient_clipped = False

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num_parameters += 1

    total_norm = total_norm**0.5

    # Check if gradients would be clipped
    if grad_clip_value is not None and total_norm > grad_clip_value:
        gradient_clipped = True

    return {
        "gradient_norm": total_norm,
        "gradient_clipping_events": 1.0 if gradient_clipped else 0.0,
        "num_params_with_grad": num_parameters,
    }


# TODO: Add proper queue
def compute_feature_stability(
    current_features: torch.Tensor, previous_features: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute feature stability across epochs (cosine similarity).

    Args:
        current_features: Current epoch's features (B, C, ...)
        previous_features: Previous epoch's features (B, C, ...)

    Returns:
        Dictionary with feature stability metrics
    """
    metrics = {}

    if current_features is None or previous_features is None:
        return metrics

    # Flatten spatial dimensions if present
    if current_features.dim() > 2:
        B = current_features.shape[0]
        current_features = current_features.reshape(B, -1).float()
    else:
        current_features = current_features.float()

    if previous_features.dim() > 2:
        B = previous_features.shape[0]
        previous_features = previous_features.reshape(B, -1).float()
    else:
        previous_features = previous_features.float()

    # Ensure same batch size (take minimum)
    min_batch = min(current_features.shape[0], previous_features.shape[0])
    current_features = current_features[:min_batch]
    previous_features = previous_features[:min_batch]

    # L2 normalize
    current_norm = torch.nn.functional.normalize(current_features, p=2, dim=1)
    previous_norm = torch.nn.functional.normalize(previous_features, p=2, dim=1)

    # Compute cosine similarity per sample
    cos_similarities = (current_norm * previous_norm).sum(dim=1)

    metrics["feature_stability_mean"] = cos_similarities.mean().item()
    metrics["feature_stability_std"] = cos_similarities.std().item()
    metrics["feature_stability_min"] = cos_similarities.min().item()
    metrics["feature_stability_max"] = cos_similarities.max().item()

    # Compute feature drift (1 - cosine similarity)
    feature_drift = 1 - cos_similarities.mean()
    metrics["feature_drift"] = feature_drift.item()

    # Compute mean feature magnitude change
    current_mag = current_features.norm(dim=1).mean()
    previous_mag = previous_features.norm(dim=1).mean()
    magnitude_change = (current_mag - previous_mag).abs() / (previous_mag + 1e-8)
    metrics["feature_magnitude_change"] = magnitude_change.item()

    return metrics
