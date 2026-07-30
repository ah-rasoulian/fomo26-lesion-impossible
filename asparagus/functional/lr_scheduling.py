from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from typing import Any, Dict, List
import torch


def separate_encoder_decoder_weights(named_parameters) -> List[Dict[str, Any]]:
    """Separate the encoder and decoder weights of a model.

    Args:
        named_parameters (List[tuple]): List of named parameters from the model.
    Returns:
        List[Dict[str, Any]]: A list containing two dictionaries, one for encoder parameters and one for decoder parameters.
    """
    encoder_params = []
    decoder_params = []
    for name, param in named_parameters:
        name = name.replace("_orig_mod.", "")  # if params are compilled we laugh and do this
        if "model.encoder" in name:
            encoder_params.append(param)
        elif "model.decoder" in name:
            decoder_params.append(param)
        else:
            # Default to encoder params for any other parameters (e.g., stem)
            encoder_params.append(param)

    # All hail the almighty assert which saved my ass. Twice.
    assert len(encoder_params) > 0 and len(decoder_params) > 0, "Encoder or decoder parameters not found."
    return [
        {"params": encoder_params, "name": "encoder"},
        {"params": decoder_params, "name": "decoder"},
    ]


def sawtooth_warmup_cosine_decay_schedule(
    optimizer,
    decoder_warmup_epochs,
    warmup_epochs,
    steps_per_epoch,
    cosine_period_ratio,  # cosine_half_period is from max to min
    max_epochs,
):
    """
    Phase 1: Decoder warmup, encoder frozen
    Phase 2: Both encoder and decoder warmup
    Phase 3: Cosine annealing for both
    """
    assert max_epochs > 0 and steps_per_epoch > 0, "max_epochs and steps_per_epoch must be greater than 0"
    print(f"Using separate warmup: decoder for {decoder_warmup_epochs} epochs, then both for {warmup_epochs} epochs")

    decoder_warmup_steps = int(decoder_warmup_epochs * steps_per_epoch)
    encoder_decoder_warmup_steps = int(warmup_epochs * steps_per_epoch)
    total_warmup_steps = decoder_warmup_steps + encoder_decoder_warmup_steps
    cosine_steps = int(cosine_period_ratio * (max_epochs * steps_per_epoch - total_warmup_steps))

    def encoder_phase1_lambda(_step):
        return 0.0  # Encoder frozen during phase 1

    def decoder_phase1_lambda(step):
        return 0.999 * step / decoder_warmup_steps + 0.001

    # LambdaLR depends on the order of the param groups
    assert optimizer.param_groups[0]["name"] == "encoder" and optimizer.param_groups[1]["name"] == "decoder", (
        "Param groups are not in the expected order."
    )

    phase1_scheduler = LambdaLR(optimizer, lr_lambda=[encoder_phase1_lambda, decoder_phase1_lambda])
    phase2_scheduler = LinearLR(optimizer, start_factor=1.0 / 1000, total_iters=encoder_decoder_warmup_steps)
    phase3_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)

    return SequentialLR(
        optimizer,
        schedulers=[phase1_scheduler, phase2_scheduler, phase3_scheduler],
        milestones=[decoder_warmup_steps, total_warmup_steps],
    )


def simple_warmup_cosine_decay_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.10,
    cosine_period_ratio: float = 1.0,
    minimum_lr: float = 1.0e-6,
):
    """
    Step-based learning-rate schedule.

    Phase 1:
        Linear warmup from 1/1000 of the peak LR to the peak LR.

    Phase 2:
        Cosine decay from the peak LR to minimum_lr.
    """
    if total_steps <= 0:
        raise ValueError(
            f"total_steps must be positive, but received {total_steps}."
        )

    if not 0.0 < warmup_ratio < 1.0:
        raise ValueError(
            "warmup_ratio must be strictly between 0 and 1."
        )

    if not 0.0 < cosine_period_ratio <= 1.0:
        raise ValueError(
            "cosine_period_ratio must be in the interval (0, 1]."
        )

    warmup_steps = max(
        1,
        round(total_steps * warmup_ratio),
    )

    remaining_steps = total_steps - warmup_steps
    cosine_steps = max(
        1,
        round(remaining_steps * cosine_period_ratio),
    )

    peak_lrs = [
        parameter_group["lr"]
        for parameter_group in optimizer.param_groups
    ]

    if any(minimum_lr >= peak_lr for peak_lr in peak_lrs):
        raise ValueError(
            "minimum_lr must be smaller than every optimizer-group "
            f"learning rate. Received minimum_lr={minimum_lr}, "
            f"peak_lrs={peak_lrs}."
        )

    # CosineAnnealingLR uses one eta_min for all parameter groups.
    # All current parameter groups have the same peak LR.
    if len(set(peak_lrs)) != 1:
        raise ValueError(
            "This scheduler expects all parameter groups to have "
            "the same peak learning rate."
        )

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1.0 / 1000.0,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=minimum_lr,
    )

    print("Learning-rate schedule configured as:")
    print(f"  - Total optimizer steps: {total_steps:,}")
    print(
        f"  - Warmup: {warmup_steps:,} steps "
        f"({warmup_steps / total_steps:.1%})"
    )
    print(f"  - Cosine decay: {cosine_steps:,} steps")
    print(f"  - Peak learning rate: {peak_lrs[0]:.3e}")
    print(f"  - Minimum learning rate: {minimum_lr:.3e}")

    return SequentialLR(
        optimizer,
        schedulers=[
            warmup_scheduler,
            cosine_scheduler,
        ],
        milestones=[warmup_steps],
    )

def cosine_decay_schedule(optimizer, steps_per_epoch, cosine_period_ratio, max_epochs=-1, max_steps=-1):
    """
    Phase 1: Cosine annealing for both encoder and decoder
    """
    # cosine_half_period is from max to min
    if max_epochs > 0:
        max_steps = max_epochs * steps_per_epoch
    cosine_steps = int(cosine_period_ratio * max_steps)
    assert cosine_steps > 0, "Cosine steps must be greater than 0 for cosine decay schedule."
    return CosineAnnealingLR(optimizer, T_max=cosine_steps)
