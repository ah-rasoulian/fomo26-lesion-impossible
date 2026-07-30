from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from asparagus.modules.transforms.crop import Torch_Crop
from asparagus.modules.transforms.pad import Torch_Pad
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from torchvision import transforms


SpatialShape = Tuple[int, int, int]


def _shape3(value: Sequence[int], name: str) -> SpatialShape:
    value = tuple(int(v) for v in value)
    if len(value) != 3 or any(v <= 0 for v in value):
        raise ValueError(f"{name} must contain three positive integers.")
    return value


def _clone_metadata(info: Mapping[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in info.items():
        result[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
    return result


def _pad_to_shape(x: torch.Tensor, target: SpatialShape) -> torch.Tensor:
    """Symmetrically pad [B, C, D, H, W] to at least target."""
    current = x.shape[-3:]
    padding = []
    for size, wanted in reversed(tuple(zip(current, target))):
        total = max(wanted - size, 0)
        padding.extend((total // 2, total - total // 2))
    return F.pad(x, padding, mode="constant", value=0.0)


def _crop(
    x: torch.Tensor,
    crop_shape: SpatialShape,
    random: bool,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Crop [B, C, D, H, W] independently for each subject.

    All channels belonging to one subject receive the same crop because they
    are spatially aligned modalities. Different subjects receive independent
    crop locations during training.

    Validation uses a deterministic center crop.
    """
    if x.ndim != 5:
        raise ValueError(
            "Expected x with shape [B, C, D, H, W], "
            f"but received {tuple(x.shape)}."
        )

    x = _pad_to_shape(x, crop_shape)

    batch_size = x.shape[0]
    spatial_shape = x.shape[-3:]

    if not random:
        starts = [
            (size - crop_size) // 2
            for size, crop_size in zip(spatial_shape, crop_shape)
        ]

        d, h, w = starts
        cd, ch, cw = crop_shape

        return x[
            ...,
            d : d + cd,
            h : h + ch,
            w : w + cw,
        ]

    maximum_starts = [
        size - crop_size
        for size, crop_size in zip(spatial_shape, crop_shape)
    ]

    starts = [
        torch.randint(
            low=0,
            high=maximum + 1,
            size=(batch_size,),
            generator=generator,
            device=x.device,
        )
        if maximum > 0
        else torch.zeros(
            batch_size,
            dtype=torch.long,
            device=x.device,
        )
        for maximum in maximum_starts
    ]

    depth_starts, height_starts, width_starts = starts
    cd, ch, cw = crop_shape

    crops = [
        x[
            batch_index,
            :,
            depth_starts[batch_index] : depth_starts[batch_index] + cd,
            height_starts[batch_index] : height_starts[batch_index] + ch,
            width_starts[batch_index] : width_starts[batch_index] + cw,
        ]
        for batch_index in range(batch_size)
    ]

    return torch.stack(crops, dim=0)


@dataclass
class BrainDinoIntensityAugmentation:
    """
    Lightweight GPU-safe MRI intensity augmentation.

    Parameters are independently sampled per sample and channel. Spatial
    operations are deliberately excluded so spacing metadata remains valid.
    """

    noise_probability: float = 0.2
    noise_std: float = 0.05
    gamma_probability: float = 0.2
    gamma_range: Tuple[float, float] = (0.7, 1.5)
    bias_probability: float = 0.2
    bias_strength: float = 0.25

    def __call__(
            self,
            x: torch.Tensor,
            channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected image with shape [B, C, D, H, W].")

        batch_channels = x.shape[:2]
        device = x.device
        dtype = x.dtype

        if self.noise_probability > 0:
            apply = (
                torch.rand((*batch_channels, 1, 1, 1), device=device)
                < self.noise_probability
            )
            noise = torch.randn_like(x) * self.noise_std
            x = torch.where(apply, x + noise, x)

        if self.gamma_probability > 0:
            apply = (
                torch.rand((*batch_channels, 1, 1, 1), device=device)
                < self.gamma_probability
            )
            gamma = torch.empty(
                (*batch_channels, 1, 1, 1), device=device, dtype=dtype
            ).uniform_(*self.gamma_range)
            minimum = x.amin(dim=(-3, -2, -1), keepdim=True)
            maximum = x.amax(dim=(-3, -2, -1), keepdim=True)
            normalized = (x - minimum) / (maximum - minimum).clamp_min(1e-6)
            gamma_x = normalized.clamp_min(0).pow(gamma) * (maximum - minimum) + minimum
            x = torch.where(apply, gamma_x, x)

        if self.bias_probability > 0:
            apply = (
                torch.rand((*batch_channels, 1, 1, 1), device=device)
                < self.bias_probability
            )
            coordinates = [
                torch.linspace(-1, 1, size, device=device, dtype=dtype)
                for size in x.shape[-3:]
            ]
            zz, yy, xx = torch.meshgrid(*coordinates, indexing="ij")
            coefficients = torch.empty(
                (*batch_channels, 3), device=device, dtype=dtype
            ).uniform_(-self.bias_strength, self.bias_strength)
            field = (
                coefficients[..., 0, None, None, None] * zz
                + coefficients[..., 1, None, None, None] * yy
                + coefficients[..., 2, None, None, None] * xx
            ).exp()
            x = torch.where(apply, x * field, x)

        if channel_mask is not None:
            if channel_mask.shape != x.shape[:2]:
                raise ValueError(
                    "channel_mask must have shape [B, C], got "
                    f"{tuple(channel_mask.shape)} for image {tuple(x.shape)}."
                )

            x = x * channel_mask[
                :, :, None, None, None
            ].to(device=x.device, dtype=x.dtype)

        return x


class BrainDinoViewTransform:
    """
    Convert the collated dataset batch into the BrainDinoModule batch contract.

    Input:
        {
            "image": Tensor[B, C, D, H, W],
            "info": {"spacing": Tensor[B, 3], "modality": Tensor[B, C], ...},
            ...
        }

    Output:
        {
            "global_crops": list[Tensor[B, C, Dg, Hg, Wg]],
            "local_crops": list[Tensor[B, C, Dl, Hl, Wl]],
            "global_masks": list[BoolTensor[B, N]],
            "global_info": list[dict],
            "local_info": list[dict],
            ...
        }
    """

    def __init__(
        self,
        global_crop_size: Sequence[int],
        local_crop_size: Sequence[int],
        token_stride: Sequence[int],
        n_global_crops: int = 2,
        n_local_crops: int = 4,
        mask_ratio: float = 0.5,
        training: bool = True,
        flip_probability: float = 0.5,
        intensity_augmentation: Optional[BrainDinoIntensityAugmentation] = None,
        channel_drop_probability: float = 0.35,
        one_unknown_probability: float = 0.25,
        all_unknown_probability: float = 0.10,
        unknown_modality_id: int = 0,
    ) -> None:
        self.global_crop_size = _shape3(global_crop_size, "global_crop_size")
        self.local_crop_size = _shape3(local_crop_size, "local_crop_size")
        self.token_stride = _shape3(token_stride, "token_stride")

        if n_global_crops < 2:
            raise ValueError("DINO requires at least two global crops.")
        if n_local_crops < 0:
            raise ValueError("n_local_crops cannot be negative.")
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1].")

        for crop_size, stride in zip(self.global_crop_size, self.token_stride):
            if crop_size % stride:
                raise ValueError(
                    "Every global_crop_size dimension must be divisible by "
                    "the corresponding token_stride."
                )

        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
        self.mask_ratio = mask_ratio
        self.training = training
        self.flip_probability = flip_probability
        self.intensity_augmentation = (
            intensity_augmentation
            if intensity_augmentation is not None
            else BrainDinoIntensityAugmentation()
        )

        for name, probability in (
                ("channel_drop_probability", channel_drop_probability),
                ("one_unknown_probability", one_unknown_probability),
                ("all_unknown_probability", all_unknown_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")

        if one_unknown_probability + all_unknown_probability > 1.0:
            raise ValueError(
                "one_unknown_probability + all_unknown_probability "
                "must not exceed 1."
            )
        self.channel_drop_probability = channel_drop_probability
        self.one_unknown_probability = one_unknown_probability
        self.all_unknown_probability = all_unknown_probability
        self.unknown_modality_id = unknown_modality_id

    @property
    def global_token_grid(self) -> SpatialShape:
        return tuple(
            crop // stride
            for crop, stride in zip(self.global_crop_size, self.token_stride)
        )

    def _augment_view(
            self,
            image: torch.Tensor,
            shape: SpatialShape,
            channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        view = _crop(
            image,
            shape,
            random=self.training,
        )

        if self.training:
            for dimension in (-3, -2, -1):
                flip_samples = (
                        torch.rand(
                            view.shape[0],
                            device=view.device,
                        )
                        < self.flip_probability
                )

                if flip_samples.any():
                    view[flip_samples] = torch.flip(
                        view[flip_samples],
                        dims=(dimension,),
                    )

            view = self.intensity_augmentation(
                view,
                channel_mask=channel_mask,
            )

        # Guarantees that padding remains exactly zero in validation too.
        view = view * channel_mask[
            :, :, None, None, None
        ].to(device=view.device, dtype=view.dtype)

        return view.contiguous()

    def _make_mask(self, batch_size: int, device: torch.device, mask_index: int = 0) -> torch.Tensor:
        number_of_tokens = math.prod(self.global_token_grid)
        number_masked = round(self.mask_ratio * number_of_tokens)
        mask = torch.zeros(
            (batch_size, number_of_tokens), dtype=torch.bool, device=device
        )

        if number_masked == 0:
            return mask

        if self.training:
            scores = torch.rand((batch_size, number_of_tokens), device=device)
            indices = scores.topk(number_masked, dim=1, largest=False).indices
        else:
            generator = torch.Generator(device=device)
            generator.manual_seed(12_345 + mask_index)

            permutation = torch.randperm(number_of_tokens, generator=generator, device=device)
            indices = permutation[:number_masked]
            indices = indices.unsqueeze(0).expand(batch_size, -1)

        return mask.scatter_(1, indices, True)

    def _augment_student_info(
            self,
            info: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Independently augment channel availability and modality identity
        for one student view.

        A dropped channel becomes invalid.
        An UNKNOWN channel remains valid but loses its modality identity.
        """
        augmented = _clone_metadata(info)

        modality = torch.as_tensor(
            augmented["modality"],
            device=augmented["channel_mask"].device,
            dtype=torch.long,
        ).clone()

        channel_mask = torch.as_tensor(
            augmented["channel_mask"],
            device=modality.device,
            dtype=torch.bool,
        ).clone()

        if not self.training:
            augmented["modality"] = modality
            augmented["channel_mask"] = channel_mask
            return augmented

        for batch_index in range(channel_mask.shape[0]):
            valid = torch.where(channel_mask[batch_index])[0]

            # Drop at most one channel and never drop the only channel.
            if (
                    valid.numel() > 1
                    and torch.rand((), device=channel_mask.device)
                    < self.channel_drop_probability
            ):
                selected = valid[
                    torch.randint(
                        low=0,
                        high=valid.numel(),
                        size=(),
                        device=channel_mask.device,
                    )
                ]

                channel_mask[batch_index, selected] = False

                # The network replaces masked IDs with its padding ID.
                modality[batch_index, selected] = -1

            valid = torch.where(channel_mask[batch_index])[0]

            identity_choice = torch.rand(
                (),
                device=channel_mask.device,
            )

            if identity_choice < self.all_unknown_probability:
                modality[batch_index, valid] = self.unknown_modality_id

            elif identity_choice < (
                    self.all_unknown_probability
                    + self.one_unknown_probability
            ):
                selected = valid[
                    torch.randint(
                        low=0,
                        high=valid.numel(),
                        size=(),
                        device=channel_mask.device,
                    )
                ]

                modality[batch_index, selected] = (
                    self.unknown_modality_id
                )

        augmented["modality"] = modality
        augmented["channel_mask"] = channel_mask

        return augmented

    @staticmethod
    def _apply_channel_mask(
            view: torch.Tensor,
            channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        if channel_mask.shape != view.shape[:2]:
            raise ValueError(
                "channel_mask must have shape "
                f"{tuple(view.shape[:2])}, got "
                f"{tuple(channel_mask.shape)}."
            )

        return (
                view
                * channel_mask[:, :, None, None, None].to(
            device=view.device,
            dtype=view.dtype,
        )
        ).contiguous()

    def __call__(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        image = batch["image"]
        if image.ndim != 5:
            raise ValueError(
                "BrainDinoViewTransform must run after collation and expects "
                "batch['image'] with shape [B, C, D, H, W]."
            )

        info = batch.get("info", {})

        if "spacing" not in info:
            raise KeyError("batch['info']['spacing'] is required.")

        channel_mask = info.get("channel_mask")

        if channel_mask is None:
            channel_mask = torch.ones(
                image.shape[:2],
                device=image.device,
                dtype=torch.bool,
            )
        else:
            channel_mask = torch.as_tensor(
                channel_mask,
                device=image.device,
                dtype=torch.bool,
            )

        if channel_mask.shape != image.shape[:2]:
            raise ValueError(
                "info['channel_mask'] must have shape [B, C], got "
                f"{tuple(channel_mask.shape)}."
            )
        batch_size, channels = image.shape[:2]
        if "modality" not in info or info["modality"] is None:
            modality = torch.full(
                (batch_size, channels),
                fill_value=self.unknown_modality_id,
                device=image.device,
                dtype=torch.long,
            )
        else:
            modality = torch.as_tensor(
                info["modality"],
                device=image.device,
                dtype=torch.long,
            )

        if modality.shape != (batch_size, channels):
            raise ValueError(
                "info['modality'] must have shape "
                f"{(batch_size, channels)}, got {tuple(modality.shape)}."
            )

        # Work with normalized tensor metadata from this point forward.
        info = _clone_metadata(info)
        info["modality"] = modality
        info["channel_mask"] = channel_mask

        # Spatial/intensity augmentation is sampled once for each global crop.
        # The resulting crop is shared by its teacher and student pair.
        base_global_crops = [
            self._augment_view(image,self.global_crop_size,channel_mask,)
            for _ in range(self.n_global_crops)
        ]

        teacher_global_info = [
            _clone_metadata(info)
            for _ in range(self.n_global_crops)
        ]

        student_global_info = [
            self._augment_student_info(info)
            for _ in range(self.n_global_crops)
        ]

        # Teacher retains all original channels.
        teacher_global_crops = [
            self._apply_channel_mask(crop,crop_info["channel_mask"],)
            for crop, crop_info in zip(base_global_crops,teacher_global_info)
        ]

        # Each student global view gets an independent modality subset.
        global_crops = [
            self._apply_channel_mask(crop,crop_info["channel_mask"])
            for crop, crop_info in zip(base_global_crops,student_global_info)
        ]

        # Local crops are student-only.
        base_local_crops = [
            self._augment_view(image,self.local_crop_size,channel_mask,)
            for _ in range(self.n_local_crops)
        ]

        student_local_info = [
            self._augment_student_info(info)
            for _ in range(self.n_local_crops)
        ]

        local_crops = [
            self._apply_channel_mask(crop,crop_info["channel_mask"])
            for crop, crop_info in zip(base_local_crops,student_local_info)
        ]

        global_masks = [
            self._make_mask(batch_size=image.shape[0], device=image.device, mask_index=crop_index)
            for crop_index in range(self.n_global_crops)
        ]

        result = dict(batch)
        result.update(
            {
                # Student inputs
                "global_crops": global_crops,
                "local_crops": local_crops,
                "global_info": student_global_info,
                "local_info": student_local_info,

                # Teacher inputs
                "teacher_global_crops": teacher_global_crops,
                "teacher_global_info": teacher_global_info,

                # iBOT masks correspond to the paired global crops.
                "global_masks": global_masks,
            }
        )

        result.pop("image", None)
        result.pop("label", None)
        return result


def braindino_GPU_train_transforms(
    global_crop_size: Sequence[int],
    local_crop_size: Sequence[int],
    token_stride: Sequence[int],
    mask_ratio: float = 0.5,
    n_global_crops: int = 2,
    n_local_crops: int = 4,
    channel_drop_probability: float = 0.35,
    one_unknown_probability: float = 0.25,
    all_unknown_probability: float = 0.10,
    unknown_modality_id: int = 0,
) -> BrainDinoViewTransform:
    return BrainDinoViewTransform(
        global_crop_size=global_crop_size,
        local_crop_size=local_crop_size,
        token_stride=token_stride,
        n_global_crops=n_global_crops,
        n_local_crops=n_local_crops,
        mask_ratio=mask_ratio,
        channel_drop_probability=channel_drop_probability,
        one_unknown_probability=one_unknown_probability,
        all_unknown_probability=all_unknown_probability,
        unknown_modality_id=unknown_modality_id,
        training=True,
    )


def braindino_GPU_val_transforms(
    global_crop_size: Sequence[int],
    local_crop_size: Sequence[int],
    token_stride: Sequence[int],
    mask_ratio: float = 0.5,
    n_global_crops: int = 2,
    n_local_crops: int = 4,
) -> BrainDinoViewTransform:
    return BrainDinoViewTransform(
        global_crop_size=global_crop_size,
        local_crop_size=local_crop_size,
        token_stride=token_stride,
        n_global_crops=n_global_crops,
        n_local_crops=n_local_crops,
        mask_ratio=mask_ratio,
        flip_probability=0.0,
        channel_drop_probability=0.0,
        one_unknown_probability=0.0,
        all_unknown_probability=0.0,
        training=False,
    )


def braindino_CPU_train_transforms(
    source_patch_size: Sequence[int],
    normalize: bool = True,
):
    """
    Add this function to presets.py where the existing Torch_* transforms live.

    The source crop must be at least as large as global_crop_size. The two
    independent global crops are subsequently made on GPU.
    """
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Pad(patch_size=source_patch_size),
            Torch_Crop(
                patch_size=source_patch_size,
                p_oversample_foreground=0.0,
            ),
        ]
    )


def braindino_CPU_val_transforms(
    source_patch_size: Sequence[int],
    normalize: bool = True,
):
    return transforms.Compose(
        [
            Torch_Normalize(normalize=normalize),
            Torch_Pad(patch_size=source_patch_size),
            Torch_Crop(
                patch_size=source_patch_size,
                p_oversample_foreground=0.0,
            ),
        ]
    )