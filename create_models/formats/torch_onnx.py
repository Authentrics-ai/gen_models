from pathlib import Path

import click
import torch

from .torch import create as create_torch
from .._types import WBTuples, Architecture

def _get_linear_inputs():
    from ..model_weights.linear import DUMMY_INPUT
    return DUMMY_INPUT

def _get_conv2d_inputs():
    from ..model_weights.conv2d import DUMMY_INPUT
    return DUMMY_INPUT

get_inputs = {Architecture.linear: _get_linear_inputs, Architecture.conv2d: _get_conv2d_inputs, Architecture.conv2d_linear: _get_conv2d_inputs}


def create(architecture: Architecture, weights: WBTuples, save_path: Path):
    torch_file = save_path.with_name(f"torch_{architecture.name}.pt")
    if not torch_file.exists():
        create_torch(architecture, weights, torch_file)

    inputs = get_inputs[architecture]()

    model = torch.load(torch_file)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError(
            f"File does not contain a PyTorch module: {torch_file.absolute}"
        )

    try:
        torch.onnx.export(model, torch.as_tensor(inputs), str(save_path))
    except torch.onnx.errors.OnnxExporterError as e:
        click.echo(
            f"Warning: Failed to convert file: {torch_file}\n  Traceback: {e.__traceback__}",
            err=True,
        )