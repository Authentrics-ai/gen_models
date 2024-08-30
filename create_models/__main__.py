from pathlib import Path
import click

FORMATS = ["torch", "tf", "onnx", "keras"]
MODELS = ["linear", "conv2d", "conv2d_linear"]

# NOTE: the keras save-file is the same regardless of backend (it actually loads
# TF libs to save the file regardless of backend).gen
# TODO: keras wrapping a PyTorch module (if possible? Can't find keras.layers.TorchModuleWrapper?)


@click.command()
@click.help_option("-h", "--help")
@click.option(
    "-f",
    "--formats",
    type=click.Choice(FORMATS + ["all"], case_sensitive=False),
    multiple=True,
    required=True,
)
@click.option(
    "-m",
    "--models",
    type=click.Choice(MODELS + ["all"], case_sensitive=False),
    multiple=True,
    required=True,
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=Path(".")
)
def cli(formats: list[str], models: list[str], output_dir: Path):
    if "all" in formats:
        formats = FORMATS
    if "all" in models:
        models = MODELS
    click.echo(f"Formats: {', '.join(formats)}")
    click.echo(f"Models: {', '.join(models)}")

    if not output_dir.exists():
        click.echo(f"Creating output directory: {output_dir.absolute}")
        output_dir.mkdir()

    for format in formats:
        if format == "keras":
            from .formats import keras

            create_fn = keras.create
        elif format == "torch":
            from .formats import torch

            create_fn = torch.create
        elif format == "tf":
            from .formats import tf

            create_fn = tf.create
        elif format == "onnx":
            from .formats import onnx

            create_fn = onnx.create

        for model in models:
            if model == "linear":
                from .model_weights import linear

                weights = linear.WEIGHTS_BIASES
            elif model == "conv2d":
                from .model_weights import conv2d

                weights = conv2d.WEIGHTS_BIASES
            elif model == "conv2d_linear":
                from .model_weights import conv2d_linear

                weights = conv2d_linear.WEIGHTS_BIASES

            create_fn(model, weights, output_dir)
