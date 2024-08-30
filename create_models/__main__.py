from pathlib import Path

import click

from .format import Format
from .architecture import Architecture


FORMATS = Format._member_names_
MODELS = Architecture._member_names_

# NOTE: the keras save-file is the same regardless of backend (it actually loads
# TF libs to save the file regardless of backend).gen
# TODO: keras wrapping a PyTorch module (if possible? Can't find keras.layers.TorchModuleWrapper?)


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
def cli(destination: Path, generate_all: bool, format_names: list[str], model_names: list[str]):
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
    formats = [Format._member_map_[f.replace('-', '_')] for f in format_names]
    models = [Architecture._member_map_[m.replace('-', '_')] for m in model_names]

    if Format.torch in formats and Format.torch_onnx in formats:
        formats.remove(Format.torch)

    if Format.tf in formats and Format.tf_onnx in formats:
        formats.remove(Format.tf)

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
                from .model_weights import linear

                weights = linear.WEIGHTS_BIASES
            elif model == Architecture.conv2d:
                from .model_weights import conv2d

                weights = conv2d.WEIGHTS_BIASES
            elif model == Architecture.conv2d_linear:
                from .model_weights import conv2d_linear

                weights = conv2d_linear.WEIGHTS_BIASES

            create_model(model, weights, destination)
