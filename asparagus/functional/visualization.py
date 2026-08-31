import logging
import numpy as np
import torch
import wandb
from asparagus.functional.decorators import depends_on_mlflow
from math import ceil, floor
from PIL import Image
from typing import Any, Dict, List

try:
    import mlflow
except ImportError:
    logging.debug("MLFlow not found. Logging with MLFlow will not work.")

FPS = 4
DPI = 100


def finish():
    wandb.finish()


def update_config(dct):
    wandb.config.update(dct)


def save_tensor(x, name):
    torch.save(x, name)
    wandb.save(name)
    print(f"Saved {name} to wandb")


def normalize_array_to_pil(img: np.ndarray) -> Image.Image:
    """
    Normalize numpy array to 0-255 range and convert to PIL Image.
    Handles medical imaging data that may have negative values or wide ranges.

    Args:
        img: Numpy array representing image data

    Returns:
        PIL Image in grayscale mode ('L')
    """
    img = img.copy()
    # Normalize to 0-1 range
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:  # Avoid division by zero
        img = (img - img_min) / (img_max - img_min)
    # Convert to 0-255 range
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img, mode="L")


def get_logger_compatible_imgs(
    x, y, y_hat, slice_dim, n=1, desc="", titles=["input", "target", "prediction"]
) -> List[Dict[str, Any]]:
    """
    Generate images in a format compatible with MLflow and Wandb loggers.
    Returns a list of dictionaries with numpy arrays ready for logging.
    """
    batch_idx = 0  # we always just plot the first batch element

    x = x[batch_idx].squeeze().detach().cpu()
    y = y[batch_idx].squeeze().detach().cpu()
    y_hat = y_hat[batch_idx].squeeze().detach().cpu()

    # we might use mixed precision, so we cast to ensure tensor is compatible with numpy
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    y_hat = y_hat.to(torch.float32)

    images = []

    if n is None:
        index_range = torch.arange(x.shape[slice_dim])
    else:
        # we take the middle n images
        middle = x.shape[slice_dim] // 2
        index_range = torch.arange(middle - floor(n / 2), middle + ceil(n / 2))
        assert len(index_range) == n

    for i in index_range:
        index = (
            i if slice_dim == 0 else slice(None),
            i if slice_dim == 1 else slice(None),
        )

        xi, yi, yi_hat = (t[index].numpy() for t in (x, y, y_hat))

        # Add all three arrays to the result
        images.append(
            {
                "images": [
                    {"title": titles[0], "array": xi},
                    {"title": titles[1], "array": yi},
                    {"title": titles[2], "array": yi_hat},
                ],
                "slice_idx": i.item(),
                "caption": f"Slice {i.item()}: {desc}",
            }
        )

    return images


def log_images_to_logger(loggers, images, step, prefix=""):
    """
    Log images to MLflow or Wandb logger.

    Args:
        loggers: List of lightning logger instances
        images: List of image dictionaries from get_logger_compatible_imgs
        step: The current step/epoch (optional, if None, logger handles step automatically)
        prefix: Prefix for the image key
    """
    if not images:
        return

    for i, img_dict in enumerate(images):
        image0 = normalize_array_to_pil(img_dict["images"][0]["array"])
        image1 = normalize_array_to_pil(img_dict["images"][1]["array"])
        image2 = normalize_array_to_pil(img_dict["images"][2]["array"])

        title0 = img_dict["images"][0]["title"]
        title1 = img_dict["images"][1]["title"]
        title2 = img_dict["images"][2]["title"]

        for logger in loggers:
            logger_type = type(logger).__name__

            # WANDB
            if "WandbLogger" in logger_type:
                import wandb

                # Batch all images into a single log call to avoid hanging
                log_data = {}
                log_data[f"{prefix}/slice_{img_dict['slice_idx']}"] = [
                    wandb.Image(image0, caption=title0),
                    wandb.Image(image1, caption=title1),
                    wandb.Image(image2, caption=title2),
                ]

                logger.experiment.log(log_data, commit=False)

            # MLFLOW
            if "MLFlowLogger" in logger_type:
                prefix = prefix.replace("/", "_")
                slice_idx = img_dict["slice_idx"]
                _log_images_to_mlflow(
                    logger=logger,
                    prefix=prefix,
                    slice_idx=slice_idx,
                    step=step,
                    image0=image0,
                    image1=image1,
                    image2=image2,
                    title0=title0,
                    title1=title1,
                    title2=title2,
                )


@depends_on_mlflow()
def _log_images_to_mlflow(logger, prefix, slice_idx, step, image0, image1, image2, title0, title1, title2):
    logger.experiment.log_image(
        run_id=logger.run_id,
        image=mlflow.Image(image0),
        key=f"{prefix}_{title0}_slice_{slice_idx}",
        step=step,
    )
    logger.experiment.log_image(
        run_id=logger.run_id,
        image=mlflow.Image(image1),
        key=f"{prefix}_{title1}_slice_{slice_idx}",
        step=step,
    )
    logger.experiment.log_image(
        run_id=logger.run_id,
        image=mlflow.Image(image2),
        key=f"{prefix}_{title2}_slice_{slice_idx}",
        step=step,
    )


def get_logger_compatible_image_output_target(
    image,
    output,
    target,
    task_type: str = "segmentation",
):
    image = np.asarray(image)
    output = np.asarray(output)
    target = np.asarray(target)

    if image.ndim < 3:
        raise ValueError(
            f"Expected an image with spatial dimensions, got {image.shape}."
        )

    # Select the channel with the greatest spatial variation. This avoids
    # visualizing a zero-filled missing modality.
    channel_variances = np.asarray(
        [
            np.nanvar(image[channel].astype(np.float32))
            for channel in range(image.shape[0])
        ],
        dtype=np.float32,
    )

    finite_channels = np.isfinite(channel_variances)
    if finite_channels.any():
        safe_variances = np.where(
            finite_channels,
            channel_variances,
            -np.inf,
        )
        channel_idx = int(np.argmax(safe_variances))
    else:
        channel_idx = 0

    if image.ndim == 4:
        # Arrays use [C, H, W, D]. Always select a slice along D,
        # which is the final axis.
        depths = [image.shape[-1]]

        if target.ndim >= 3:
            depths.append(target.shape[-1])
        if output.ndim >= 3:
            depths.append(output.shape[-1])

        common_depth = min(int(depth) for depth in depths)

        if common_depth < 1:
            raise ValueError(
                "Cannot visualize an empty volume: "
                f"image={image.shape}, output={output.shape}, "
                f"target={target.shape}."
            )

        if task_type == "segmentation":
            # Convert target into one [H, W, D] integer label map.
            if target.ndim == 4:
                if target.shape[0] == 1:
                    target_volume = target[0]
                else:
                    # Also supports one-hot encoded targets.
                    target_volume = target.argmax(axis=0)
            elif target.ndim == 3:
                target_volume = target
            else:
                raise ValueError(
                    "Segmentation target must be [H, W, D], "
                    "[1, H, W, D], or [C, H, W, D], but got "
                    f"{target.shape}."
                )

            target_volume = target_volume[..., :common_depth]

            # For multiclass segmentation, IDs greater than zero are
            # foreground. Choose the slice with the greatest total
            # foreground area.
            foreground_per_slice = (
                target_volume > 0
            ).reshape(-1, common_depth).sum(axis=0)

            if foreground_per_slice.max() > 0:
                slice_to_visualize = int(
                    foreground_per_slice.argmax()
                )
            else:
                slice_to_visualize = common_depth // 2
        else:
            slice_to_visualize = common_depth // 2

        slice_to_visualize = int(
            np.clip(
                slice_to_visualize,
                0,
                common_depth - 1,
            )
        )

        image = image[..., slice_to_visualize]

        if target.ndim in (3, 4):
            target = target[..., slice_to_visualize]

        if output.ndim in (3, 4):
            output = output[..., slice_to_visualize]

    image = normalize_array_to_pil(image[channel_idx])

    if task_type == "classification":
        target = np.round(target.squeeze(0), decimals=3)
        output = np.round(output.argmax(0), decimals=3)

    elif task_type == "regression":
        target = np.round(target.squeeze(0), decimals=3)
        output = np.round(output.squeeze(0), decimals=3)

    elif task_type == "segmentation":
        # Target can be an integer map with one channel or a one-hot map.
        if target.ndim == 3:
            if target.shape[0] == 1:
                target = target[0]
            else:
                target = target.argmax(axis=0)
        elif target.ndim != 2:
            raise ValueError(
                "The sliced segmentation target must be [H, W], "
                f"[1, H, W], or [C, H, W], got {target.shape}."
            )

        # Output is normally multiclass probabilities [C, H, W].
        if output.ndim == 3:
            if output.shape[0] == 1:
                # Compatibility with an older binary one-channel model.
                output = (output[0] >= 0.5).astype(np.int32)
            else:
                output = output.argmax(axis=0)
        elif output.ndim != 2:
            raise ValueError(
                "The sliced segmentation output must be [H, W], "
                f"[1, H, W], or [C, H, W], got {output.shape}."
            )

        # W&B segmentation masks should contain integer class IDs.
        target = np.rint(target).astype(np.int32)
        output = np.rint(output).astype(np.int32)

    elif task_type == "self-supervised":
        target = normalize_array_to_pil(target[channel_idx])
        output = normalize_array_to_pil(output[channel_idx])

    else:
        logging.warning(
            "Unknown task type %r. Expected classification, regression, "
            "segmentation, or self-supervised.",
            task_type,
        )

    return image, output, target


def log_image_output_target_to_wandb(
    logger,
    image,
    output,
    target,
    log_key: str,
    fig_title: str,
    step,
    task_type: str = "segmentation",
):
    """
    Log a random image from the imagedict to wandb
    """
    if task_type in ["classification", "regression"]:
        fig = wandb.Image(image, mode="L", caption=f"P: {output} | GT: {target} | {fig_title}")
    elif task_type == "segmentation":
        output = np.asarray(output).astype(np.int32)
        target = np.asarray(target).astype(np.int32)

        if output.ndim != 2 or target.ndim != 2:
            raise ValueError(
                "W&B segmentation masks must be 2-D after slice selection; "
                f"output={output.shape}, target={target.shape}."
            )

        # Register only foreground classes. Class 0 is intentionally omitted so
        # W&B treats it as background instead of drawing a blue overlay.
        foreground_ids = sorted(
            {
                int(class_id)
                for class_id in np.concatenate(
                [np.unique(output), np.unique(target)]
            )
                if int(class_id) > 0
            }
        )

        class_labels = {
            class_id: (
                "foreground"
                if len(foreground_ids) == 1
                else f"class_{class_id}"
            )
            for class_id in foreground_ids
        }

        fig = [
            wandb.Image(
                image,
                mode="L",
                masks={
                    "predictions": {
                        "mask_data": output,
                        "class_labels": class_labels,
                    },
                    "ground_truth": {
                        "mask_data": target,
                        "class_labels": class_labels,
                    },
                },
                caption=fig_title,
            ),
            wandb.Image(
                normalize_array_to_pil(output.astype(np.float32)),
                caption="predicted class map",
            ),
            wandb.Image(
                normalize_array_to_pil(target.astype(np.float32)),
                caption="ground-truth class map",
            ),
        ]
    elif task_type == "self-supervised":
        fig = [
            wandb.Image(image, mode="L", caption=fig_title),
            wandb.Image(output, mode="L", caption="output"),
            wandb.Image(target, mode="L", caption="target"),
        ]
    logger.experiment.log({log_key: fig})


def log_image_output_target_to_mlflow(
    logger,
    image,
    output,
    target,
    log_key: str,
    fig_title: str,
    step,
    task_type: str = "segmentation",
):
    """
    Log a random image from the imagedict to wandb
    """
    log_key = log_key.replace("/", "_")
    fig_title = fig_title.replace("/", "_")

    if task_type in ["classification", "regression"]:
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=image,
            key=f"{log_key}_P:_{output}_|_GT:_{target}",
            step=step,
        )
    elif task_type == "segmentation":
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=mlflow.Image(image),
            key=f"{log_key}_input",
            step=step,
        )
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=mlflow.Image(output),
            key=f"{log_key}_output",
            step=step,
        )
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=mlflow.Image(target),
            key=f"{log_key}_target",
            step=step,
        )
    elif task_type == "self-supervised":
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=image,
            key=f"{fig_title}",
            step=step,
        )
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=output,
            key="output",
            step=step,
        )
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=target,
            key="target",
            step=step,
        )
