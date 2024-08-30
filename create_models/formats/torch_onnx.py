from pathlib import Path

import click
import torch

from create_models.architecture import Architecture

from .torch import create as create_torch
from ..model_weights.common import WBTuples

def _get_linear_inputs():
    from ..model_weights.linear import DUMMY_INPUT
    return DUMMY_INPUT

def _get_conv2d_inputs():
    from ..model_weights.conv2d import DUMMY_INPUT
    return DUMMY_INPUT

def _get_conv2d_linear_inputs():
    from ..model_weights.conv2d_linear import DUMMY_INPUT
    return DUMMY_INPUT

get_inputs = {Architecture.linear: _get_linear_inputs, Architecture.conv2d: _get_conv2d_inputs, Architecture.conv2d_linear: _get_conv2d_linear_inputs}


def create(architecture: Architecture, weights: WBTuples, save_dir: Path) -> Path:
    torch_file = create_torch(architecture, weights, save_dir)

    inputs = get_inputs[architecture]()
    converted_file = save_dir / f"onnx_torch_{architecture.name}.onnx"

    model = torch.load(torch_file)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError(
            f"File does not contain a PyTorch module: {torch_file.absolute}"
        )

    try:
        torch.onnx.export(model, torch.as_tensor(inputs), str(converted_file))
    except torch.onnx.errors.OnnxExporterError as e:
        click.echo(
            f"Warning: Failed to convert file: {torch_file}\n  Traceback: {e.__traceback__}",
            err=True,
        )

    return converted_file