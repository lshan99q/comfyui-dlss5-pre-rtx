"""Pure-PyTorch DLSS 5 neural-rendering runtime used by the ComfyUI nodes.

Implementation is derived from the reverse-engineered MLX-DLSS project and
adapted to live directly in this repository. See THIRD_PARTY_NOTICES.md.
"""

from .pipeline import EnhanceResult, NeuralRenderingPipeline, NeuralRenderingSession
from .temporal import BLEND_SCALE, make_temporal_features, normalize_pixel_motion

__all__ = [
    "EnhanceResult",
    "NeuralRenderingPipeline",
    "NeuralRenderingSession",
    "BLEND_SCALE",
    "make_temporal_features",
    "normalize_pixel_motion",
]
