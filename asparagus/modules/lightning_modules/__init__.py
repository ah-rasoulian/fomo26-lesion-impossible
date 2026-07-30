from .clsreg_module import ClassificationModule, RegressionModule
from .linear_probe_module import LinearProbeModule
from .segmentation_module import SegmentationModule
from .self_supervised import SelfSupervisedModule
from .braindino_module import BrainDinoModule

__all__ = [
    "SegmentationModule",
    "ClassificationModule",
    "RegressionModule",
    "SelfSupervisedModule",
    "LinearProbeModule",
    "BrainDinoModule",
]
