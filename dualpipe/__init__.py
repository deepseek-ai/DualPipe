__version__ = "1.0.0"

from dualpipe.dualpipe import DualPipe
from dualpipe.dualpipev import DualPipeV
from dualpipe.comm import (
    set_p2p_tensor_shapes,
    set_p2p_tensor_dtype,
)
from dualpipe.utils import WeightGradStore

__all__ = [
    "DualPipe",
    "DualPipeV",
    "WeightGradStore",
    "set_p2p_tensor_shapes",
    "set_p2p_tensor_dtype",
]
