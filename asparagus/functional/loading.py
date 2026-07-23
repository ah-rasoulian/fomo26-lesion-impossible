from typing import Dict, Any, List
import json
import nibabel as nib
from nibabel.orientations import aff2axcodes
import numpy as np
import torch
import warnings
import pickle
import os


# Modality-family IDs for FOMO26 stem embeddings.
#
# Design rule:
# - Modalities that appear as separate task inputs get separate IDs.
# - Modalities that are explicitly interchangeable in the task definition share an ID.
# - Generic or uncertain MR inputs fall back to UNKNOWN unless the filename gives more detail.

MODALITY_TO_ID: Dict[str, int] = {
    # -------------------------------------------------------------------------
    # Unknown / generic MR input
    # -------------------------------------------------------------------------
    "UNKNOWN": 0,  # Unknown modality or generic "--input: Path to an MR image".

    # -------------------------------------------------------------------------
    # FLAIR
    # Used explicitly in Task 1 and Task 2.
    # Keep separate from T2w because the task treats FLAIR as its own input.
    # -------------------------------------------------------------------------
    "FLAIR": 1,        # T2 FLAIR image; fluid-attenuated inversion recovery.
    "flair": 1,

    # -------------------------------------------------------------------------
    # DWI
    # Used explicitly in Task 1 and Task 2.
    # Keep separate from ADC because Task 1 uses both DWI and ADC.
    # -------------------------------------------------------------------------
    "dwi": 2,          # Diffusion-weighted image, usually b1000 in this challenge.

    # -------------------------------------------------------------------------
    # ADC
    # Used explicitly in Task 1.
    # Derived from DWI, but should have a separate embedding because it is a
    # separate task input with different intensity meaning.
    # -------------------------------------------------------------------------
    "adc": 3,          # Apparent diffusion coefficient map.
    "ADC": 3,          # Same as ADC, uppercase filename variant.

    # -------------------------------------------------------------------------
    # Susceptibility / T2* family
    # Task says T2* and SWI are optional replacements for each other, so they
    # should share one embedding.
    # -------------------------------------------------------------------------
    "T2starw": 4,      # T2*-weighted image.
    "t2s": 4,          # Possible filename token for T2*.
    "T2S": 4,          # Possible uppercase filename token for T2*.
    "swi": 4,          # Susceptibility-weighted image.
    "SWI": 4,          # Same as SWI, uppercase filename variant.
    "gre": 4,          # Gradient-echo image; often susceptibility/T2*-related.
    "R2starmap": 4,    # R2* map; related susceptibility/T2* quantitative map.

    # -------------------------------------------------------------------------
    # T1-weighted family
    # Used explicitly in Task 3 and Task 5.
    # -------------------------------------------------------------------------
    "T1w": 5,          # T1-weighted anatomical image.
    "t1": 5,           # Possible task-style filename token.
    "t1w": 5,           # Possible task-style filename token.
    "T1": 5,           # Possible uppercase filename token.
    "UNIT1": 5,        # Uniform T1-weighted image, commonly from MP2RAGE.
    "MP2RAGE": 5,      # MP2RAGE T1-like/qT1 acquisition.
    "mp2rage": 5,      # Same as MP2RAGE, lowercase filename variant.
    "T1map": 5,        # Quantitative T1 map.
    "R1map": 5,        # Quantitative R1 map; inverse of T1.

    # -------------------------------------------------------------------------
    # T1 contrast-enhanced
    # Not listed in the task inputs, but clinically different enough from T1w
    # that I would keep it separate if present.
    # -------------------------------------------------------------------------
    "T1c": 6,          # Contrast-enhanced T1-weighted image.

    # -------------------------------------------------------------------------
    # T2-weighted family
    # Used explicitly in Task 4.
    # Keep separate from FLAIR because Task 4 asks for T2, not FLAIR.
    # -------------------------------------------------------------------------
    "T2w": 7,          # T2-weighted anatomical image.
    "t2": 7,           # Possible task-style filename token.
    "t2w": 7,           # Possible task-style filename token.
    "T2": 7,           # Possible uppercase filename token.
    "PDw": 7,          # Proton-density-weighted image; structurally closer to T2/PD family.
    "MESE": 7,         # Multi-echo spin echo; often related to T2 mapping.

    # -------------------------------------------------------------------------
    # Perfusion / ASL family
    # Not in the explicit task inputs, but keep separate if present.
    # -------------------------------------------------------------------------
    "asl": 8,          # Arterial spin labeling acquisition.
    "m0scan": 8,       # ASL M0 calibration scan.
    "cbf": 8,          # Cerebral blood flow map, usually derived from ASL.
    "att": 8,          # Arterial transit time / arrival time map.

    # -------------------------------------------------------------------------
    # Angiography
    # -------------------------------------------------------------------------
    "angio": 9,        # MR angiography image.

    # -------------------------------------------------------------------------
    # Magnetization transfer
    # -------------------------------------------------------------------------
    "MTRmap": 10,      # Magnetization transfer ratio map.
    "MTsat": 10,       # Magnetization transfer saturation map, if present.

    # -------------------------------------------------------------------------
    # Other MRI acquisition families
    # These are dataset-specific and not explicit task inputs.
    # -------------------------------------------------------------------------
    "FLASH": 11,  # Fast low-angle shot gradient-echo sequence.
    "UTE": 12,  # Ultrashort echo time MRI.
}

def get_modality_id(files: str | List[str]) -> torch.Tensor:
    def get_single_file_id(file):
        file_name = os.path.basename(file)

        modality_name = "UNKNOWN"
        for mod in MODALITY_TO_ID.keys():
            if mod in file_name:
                modality_name = mod
                break

        if modality_name == "UNKNOWN":
            warnings.warn(f"UNKNOWN modality: {file_name}")
        return MODALITY_TO_ID[modality_name]

    ids = []
    if isinstance(files, str):
        files = [files]

    for f in files:
        ids.append(get_single_file_id(f))
    modality_id = torch.tensor(ids, dtype=torch.long)
    return modality_id

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
    modality_id = get_modality_id(file)

    if file.endswith(".pt"):
        with open(file.replace(".pt", ".pkl"), "rb") as file_info:
            info = pickle.load(file_info)

        return {
            "affine": torch.as_tensor(info["nifti_metadata"]["affine"], dtype=torch.float32),
            "spacing": torch.as_tensor(info["new_spacing"], dtype=torch.float32),
            "direction": info["new_direction"],
            "modality": modality_id,
        }

    elif file.endswith(".nii.gz") or file.endswith(".nii"):

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

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)
