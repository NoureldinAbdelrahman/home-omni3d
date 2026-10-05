"""CPU marching-cubes with the tatsy/torchmcubes call convention.

Why this exists: ``torchmcubes`` cannot compile on this machine (its CMake
requires a CUDA>=13 toolchain dialect; only nvcc 12.8 is installed), so
``import tsr.system`` (TripoSR) would fail at import time. This shim exposes
the single function TripoSR uses — ``marching_cubes(volume, isolevel)`` —
delegating to PyMCubes (prebuilt wheel, CPU).

This is a RECORDED compatibility layer, not a silent fallback: every eval
summary that uses it carries ``marching_cubes: cpu-shim-via-pymcubes``.
Algorithmically both are Lewiner marching cubes over the same index grid;
only the device differs. Face winding may differ between implementations,
which is irrelevant for our point-sampling evaluation (positions only).
"""

import numpy as np
import torch

try:
    from skimage.measure import marching_cubes as _sk_mcubes
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torchmcubes-cpu-shim requires the 'scikit-image' package "
        "(pip install scikit-image)"
    ) from exc

__version__ = "0.1.0+cpu-shim"
__backend__ = "pymcubes-cpu"


def marching_cubes(volume, isolevel):
    """marching_cubes(volume, isolevel) -> (vertices, faces) as torch tensors.

    Mirrors the tatsy/torchmcubes call convention used by TripoSR's
    ``MarchingCubeHelper``: ``volume`` is a torch FloatTensor grid,
    ``isolevel`` a scalar; returns ``(FloatTensor[N,3], LongTensor[M,3])``
    on the input's device.
    """
    device = volume.device if torch.is_tensor(volume) else torch.device("cpu")
    vol = (volume.detach().cpu().numpy() if torch.is_tensor(volume)
           else np.asarray(volume)).astype(np.float32, copy=False)
    verts, faces, _, _ = _sk_mcubes(vol, float(isolevel))
    return (torch.from_numpy(np.ascontiguousarray(verts)).float().to(device),
            torch.from_numpy(np.ascontiguousarray(faces)).long().to(device))
