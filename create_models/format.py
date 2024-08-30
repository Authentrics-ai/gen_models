from enum import IntEnum


class Format(IntEnum):
    torch = 0
    tf = 1
    keras = 2
    tf_onnx = 3
    torch_onnx = 4
