import torch
from abc import abstractmethod
from torch import nn
from gardening_tools.modules.networks.utils import get_steps_for_sliding_window
from gardening_tools.modules.networks.BaseNet import BaseNet


class SpacingModalityAwareBaseNet(BaseNet):
    def __init__(self):
        super(SpacingModalityAwareBaseNet, self).__init__()

    @abstractmethod
    def forward(self):
        """
        implement in individual trainers.
        DO NOT INCLUDE FINAL SOFTMAX/SIGMOID ETC.
        WILL BE HANDLED BY LOSS FUNCTIONS
        """

    def sliding_window_predict(
        self,
        data,
        patch_size,
        overlap,
        spacing=None,
        modality=None,
        mirror=False,
    ):
        if spacing is None or modality is None:
            raise ValueError("Spacing and Modality are required.")

        if len(patch_size) == 3:
            mode = "3D"
        elif len(patch_size) == 2:
            mode = "2D"
        else:
            raise ValueError(
                f"patch_size must be of length 2 or 3, but got length: {len(patch_size)} from: {patch_size}"
            )

        if mode == "3D":
            predict_fn = self._sliding_window_predict3D
        elif mode == "2D":
            predict_fn = self._sliding_window_predict2D

        pred = predict_fn(data, patch_size, overlap, spacing, modality)
        if mirror:
            pred += torch.flip(
                predict_fn(torch.flip(data, (2,)), patch_size, overlap, spacing, modality), (2,)
            )
            pred += torch.flip(
                predict_fn(torch.flip(data, (3,)), patch_size, overlap, spacing, modality), (3,)
            )
            pred += torch.flip(
                predict_fn(torch.flip(data, (2, 3)), patch_size, overlap, spacing, modality), (2, 3)
            )
            div = 4
            if mode == "3D":
                pred += torch.flip(
                    predict_fn(torch.flip(data, (4,)), patch_size, overlap, spacing, modality), (4,)
                )
                pred += torch.flip(
                    predict_fn(torch.flip(data, (2, 4)), patch_size, overlap, spacing, modality), (2, 4)
                )
                pred += torch.flip(
                    predict_fn(torch.flip(data, (3, 4)), patch_size, overlap, spacing, modality), (3, 4)
                )
                pred += torch.flip(
                    predict_fn(torch.flip(data, (2, 3, 4)), patch_size, overlap, spacing, modality),
                    (2, 3, 4),
                )
                div += 4
            pred /= div
        return pred

    def _full_image_predict(self, data, spacing=None, modality=None):
        """
        Standard prediction used in cases where models predict on full-size images.
        This is opposed to patch-based predictions where we use a sliding window approach to generate
        full size predictions.
        """
        return self.forward(data, spacing, modality)

    def _sliding_window_predict3D(self, data, patch_size, overlap, spacing=None, modality=None):
        """
        Sliding window prediction implementation
        """
        canvas = torch.zeros(
            (1, self.num_classes, *data.shape[2:]),
            device=data.device,
        )

        x_steps, y_steps, z_steps = get_steps_for_sliding_window(
            data.shape[2:], patch_size, overlap
        )
        px, py, pz = patch_size

        for xs in x_steps:
            for ys in y_steps:
                for zs in z_steps:
                    # check if out of bounds
                    out = self.forward(
                        data[:, :, xs : xs + px, ys : ys + py, zs : zs + pz], spacing, modality
                    )
                    canvas[:, :, xs : xs + px, ys : ys + py, zs : zs + pz] += out
        return canvas

    def _sliding_window_predict2D(self, data, patch_size, overlap, spacing=None, modality=None):
        """
        Sliding window prediction implementation
        """
        canvas = torch.zeros(
            (1, self.num_classes, *data.shape[2:]),
            device=data.device,
        )

        px, py = patch_size

        # If we have 5 dimensions we are working with 3D data, and need to predict each slice.
        if len(data.shape) == 5:
            x_steps, y_steps = get_steps_for_sliding_window(
                data.shape[3:], patch_size, overlap
            )
            for idx in range(data.shape[2]):
                for xs in x_steps:
                    for ys in y_steps:
                        out = self.forward(data[:, :, idx, xs : xs + px, ys : ys + py], spacing, modality)
                        canvas[:, :, idx, xs : xs + px, ys : ys + py] += out
            return canvas

        # else we proceed with the data as 2D
        x_steps, y_steps = get_steps_for_sliding_window(
            data.shape[2:], patch_size, overlap
        )

        for xs in x_steps:
            for ys in y_steps:
                # check if out of bounds
                out = self.forward(data[:, :, xs : xs + px, ys : ys + py], spacing, modality)
                canvas[:, :, xs : xs + px, ys : ys + py] += out
        return canvas
