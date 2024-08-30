from pathlib import Path

import click
import numpy as np

from ._types import Format, Architecture, WBTuples
from .model_weights import linear, conv2d


FORMATS = Format._member_names_
MODELS = Architecture._member_names_
EXT = {
    Format.keras: ".keras",
    Format.tf: "",
    Format.torch: ".pt",
    Format.tf_onnx: ".onnx",
    Format.torch_onnx: ".onnx",
}

# NOTE: the keras save-file is the same regardless of backend (it actually loads
# TF libs to save the file regardless of backend).gen
# TODO: keras wrapping a PyTorch module (if possible? Can't find keras.layers.TorchModuleWrapper?)

def copy(wb: WBTuples) -> WBTuples:
    return [(np.copy(w), np.copy(b)) for w, b in wb]

conv2d_linear_wb = copy(conv2d.WEIGHTS_BIASES) + copy(linear.WEIGHTS_BIASES)

@click.command()
@click.argument(
    "destination",
    nargs=1,
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=Path("."),
)
@click.help_option("-h", "--help")
@click.option(
    "--all",
    "generate_all",
    is_flag=True,
    help="Shortcut for '-f all -m all'",
)
@click.option(
    "--overwrite/--no-overwrite",
    help="Overwrite existing models or skip if they exist",
    default=False,
)
@click.option(
    "-f",
    "--formats",
    "format_names",
    type=click.Choice(FORMATS + ["all"], case_sensitive=False),
    multiple=True,
    help="File formats to generate with each specified model",
)
@click.option(
    "-m",
    "--models",
    "model_names",
    type=click.Choice(MODELS + ["all"], case_sensitive=False),
    multiple=True,
    help="Model architectures to generate in each specified format",
)
def cli(destination: Path, generate_all: bool, overwrite: bool, format_names: list[str], model_names: list[str]):
    if generate_all:
        format_names = ["all"]
        model_names = ["all"]

    if len(format_names) == 0:
        raise click.BadOptionUsage("formats", "Option '--formats' should always be set")
    if len(model_names) == 0:
        raise click.BadOptionUsage("models", "Option '--models' should always be set")

    if "all" in format_names:
        format_names = FORMATS
    if "all" in model_names:
        model_names = MODELS
    click.echo(f"Formats: {', '.join(format_names)}")
    click.echo(f"Models: {', '.join(model_names)}")

    if not destination.exists():
        click.echo(f"Creating output directory: {destination.absolute}")
        destination.mkdir()

    # Transform strings to enums
    formats: list[Format] = list()
    for f in Format:
        if f.name in format_names:
            formats.append(f)
    models: list[Architecture] = list()
    for a in Architecture:
        if a.name in model_names:
            models.append(a)

    for format in formats:
        if format == Format.keras:
            from .formats.keras import create as create_model
        elif format == Format.torch:
            from .formats.torch import create as create_model
        elif format == Format.tf:
            from .formats.tf import create as create_model
        elif format == Format.torch_onnx:
            from .formats.torch_onnx import create as create_model
        elif format == Format.tf_onnx:
            from .formats.tf_onnx import create as create_model


        for model in models:
            if model == Architecture.linear:
                weights = linear.WEIGHTS_BIASES
            elif model == Architecture.conv2d:
                weights = conv2d.WEIGHTS_BIASES
            elif model == Architecture.conv2d_linear:
                weights = conv2d_linear_wb

            save_file = destination / f"{format.name}_{model.name}{EXT[format]}"
            if overwrite or not save_file.exists():
                create_model(model, copy(weights), save_file)
