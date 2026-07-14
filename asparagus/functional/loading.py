from typing import Dict, Any
import nibabel as nib
from nibabel.orientations import aff2axcodes
import numpy as np
import torch
import warnings
import pickle
import os


# Modality-family IDs for stem modality embeddings.
# Similar modalities intentionally share the same ID so they use the same learned embedding.
MODALITY_TO_ID: Dict[str, int] = {
    # -------------------------------------------------------------------------
    # Unknown / fallback
    # -------------------------------------------------------------------------
    "UNKNOWN": 0,       # Unknown or unrecognized modality.

    # -------------------------------------------------------------------------
    # Diffusion family
    # -------------------------------------------------------------------------
    "dwi": 1,          # Diffusion-weighted MRI; raw diffusion acquisition.
    "dwi_b1000": 1,    # Diffusion-weighted MRI; raw diffusion acquisition.
    "adc": 1,          # Apparent diffusion coefficient map; derived from DWI.
    "ADC": 1,          # Same as adc, uppercase filename variant.

    # -------------------------------------------------------------------------
    # T1-weighted / T1-like anatomical family
    # -------------------------------------------------------------------------
    "T1w": 2,          # T1-weighted anatomical MRI.
    "t1": 2,
    "t1w": 2,
    "UNIT1": 2,        # Uniform T1-weighted image, commonly derived from MP2RAGE.
    "MP2RAGE": 2,      # MP2RAGE T1-like/qT1 acquisition.
    "mp2rage": 2,      # Same as MP2RAGE, lowercase filename variant.
    "T1map": 2,        # Quantitative T1 relaxation map.
    "R1map": 2,        # Quantitative R1 map; inverse of T1.

    # -------------------------------------------------------------------------
    # Contrast-enhanced T1
    # -------------------------------------------------------------------------
    "T1c": 3,          # Contrast-enhanced T1-weighted MRI; post-gadolinium T1.

    # -------------------------------------------------------------------------
    # T2 / fluid-sensitive anatomical family
    # -------------------------------------------------------------------------
    "T2w": 4,          # T2-weighted anatomical MRI.
    "t2w": 4,
    "FLAIR": 4,        # T2-like image with CSF suppression.
    "flair": 4,        # T2-like image with CSF suppression, lowercase variant.
    "PDw": 4,          # Proton-density-weighted MRI; often structurally similar to T2/PD scans.
    "MESE": 4,         # Multi-echo spin-echo acquisition; often used for T2 mapping.

    # -------------------------------------------------------------------------
    # Susceptibility / gradient-echo family
    # -------------------------------------------------------------------------
    "swi": 5,          # Susceptibility-weighted imaging.
    "gre": 5,          # Gradient-echo image; broad susceptibility-sensitive family.
    "T2starw": 5,      # T2*-weighted MRI.
    "t2s": 5,
    "R2starmap": 5,    # Quantitative R2* map; inverse of T2*.

    # -------------------------------------------------------------------------
    # Perfusion / ASL family
    # -------------------------------------------------------------------------
    "asl": 6,          # Arterial spin labeling acquisition.
    "m0scan": 6,       # ASL M0 calibration image.
    "cbf": 6,          # Cerebral blood flow map; usually derived from ASL.
    "att": 6,          # Arterial transit time / arrival time map; usually derived from ASL.

    # -------------------------------------------------------------------------
    # Angiography family
    # -------------------------------------------------------------------------
    "angio": 7,        # MR angiography image.

    # -------------------------------------------------------------------------
    # Other MRI acquisition families
    # -------------------------------------------------------------------------
    "FLASH": 8,        # Fast low-angle shot gradient-echo sequence; dataset-specific contrast.
    "UTE": 9,          # Ultrashort echo time MRI.
}


def load_image_file(file: str) -> torch.Tensor:
    if file.endswith(".pt"):
        return torch.load(file)
    elif file.endswith(".nii.gz") or file.endswith(".nii"):
        nii = nib.load(file)
        data = nii.get_fdata(dtype=np.float32)
        tensor = torch.from_numpy(data)
        return tensor.unsqueeze(0)  # (H,W,D) -> (1,H,W,D) to match .pt channel convention
    else:
        raise ValueError(f"Unsupported file format: {file}. Expected .pt, .nii, or .nii.gz")

def get_file_info(file):
    file_name = os.path.basename(file)

    modality_name = "UNKNOWN"
    for mod in MODALITY_TO_ID.keys():
        if mod in file_name:
            modality_name = mod
            break

    if modality_name == "UNKNOWN":
        warnings.warn(f"UNKNOWN modality: {file_name}")
    modality_id = torch.tensor(MODALITY_TO_ID[modality_name], dtype=torch.long)

    if file_name.endswith(".pt"):
        with open(file.replace(".pt", ".pkl"), "rb") as file_info:
            info = pickle.load(file_info)

        return {
            "affine": torch.as_tensor(info["nifti_metadata"]["affine"], dtype=torch.float32),
            "spacing": torch.as_tensor(info["new_spacing"], dtype=torch.float32),
            "direction": info["new_direction"],
            "modality": modality_id,
        }

    elif file_name.endswith(".nii.gz") or file_name.endswith(".nii"):

        nii = nib.load(file)
        header = nii.header

        affine = torch.as_tensor(nii.affine, dtype=torch.float32)
        spacing = torch.as_tensor(header.get_zooms()[:3], dtype=torch.float32)

        direction = "".join(aff2axcodes(nii.affine))
        return {
            "affine": affine,
            "spacing": spacing,
            "direction": direction,
            "modality": modality_id,
        }

    else:
        raise ValueError(
            f"Unsupported file format: {file}. Expected .pt, .nii, or .nii.gz"
        )
