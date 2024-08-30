from enum import IntEnum

import numpy.typing as npt

WBTuple = tuple[npt.NDArray, npt.NDArray]
WBTuples = list[WBTuple]


class Architecture(IntEnum):
    linear = 0
    conv2d = 1
    conv2d_linear = 2


class Format(IntEnum):
    torch = 0
    tf = 1
    keras = 2
    tf_onnx = 3
    torch_onnx = 4
