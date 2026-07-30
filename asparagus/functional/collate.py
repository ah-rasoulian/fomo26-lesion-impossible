import torch
from torch.utils.data._utils.collate import default_collate


def subjectwise_pretrain_collate(batch):
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    spatial_shapes = {
        tuple(sample["image"].shape[1:])
        for sample in batch
    }

    if len(spatial_shapes) != 1:
        raise ValueError(
            "All images must have the same spatial shape after CPU transforms. "
            f"Received: {sorted(spatial_shapes)}"
        )

    max_channels = max(
        sample["image"].shape[0]
        for sample in batch
    )

    padded_batch = []

    for original_sample in batch:
        sample = original_sample.copy()
        sample["info"] = original_sample["info"].copy()

        image = original_sample["image"]
        num_channels = image.shape[0]

        modality = torch.as_tensor(
            sample["info"]["modality"],
            dtype=torch.long,
        ).reshape(-1)

        if modality.numel() != num_channels:
            raise ValueError(
                "Expected one modality ID per image channel, but received "
                f"image={tuple(image.shape)}, "
                f"modality={tuple(modality.shape)}."
            )

        channel_mask = torch.zeros(
            max_channels,
            dtype=torch.bool,
        )
        channel_mask[:num_channels] = True

        padding_channels = max_channels - num_channels

        if padding_channels > 0:
            image_padding = image.new_zeros(
                padding_channels,
                *image.shape[1:],
            )
            image = torch.cat(
                (image, image_padding),
                dim=0,
            )

            modality_padding = torch.full(
                (padding_channels,),
                fill_value=-1,
                dtype=torch.long,
            )
            modality = torch.cat(
                (modality, modality_padding),
                dim=0,
            )

        sample["image"] = image
        sample["info"]["modality"] = modality
        sample["info"]["channel_mask"] = channel_mask

        padded_batch.append(sample)

    # Variable-length per-subject metadata must not be passed through
    # default_collate, because it tries to transpose and collate the lists.
    session_paths = [
        sample.pop("session_path")
        for sample in padded_batch
    ]

    collated_batch = default_collate(padded_batch)

    # List of length B; each element contains that subject's selected files.
    collated_batch["session_path"] = session_paths

    return collated_batch

def collate_return(x):
    x = x[0]
    x["image"] = x["image"].unsqueeze(0)
    if x.get("label") is not None:
        x["label"] = x["label"].unsqueeze(0)
    return x
