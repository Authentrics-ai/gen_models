from pathlib import Path
import subprocess as spr

import click

from create_models.architecture import Architecture

from .tf import create as create_tf
from ..model_weights.common import WBTuples


def create(architecture: Architecture, weights: WBTuples, save_dir: Path) -> Path:
    tf_dir = create_tf(architecture, weights, save_dir)
    converted_file = save_dir / f"onnx_tf_{architecture.name}.onnx"

    assert tf_dir.is_dir()
    proc = spr.run(
        [
            "python",
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            str(tf_dir),
            "--output",
            str(converted_file),
        ]
    )
    if proc.returncode:
        click.echo(f"Warning: tf2onnx returned with non-zero return code.")

    return converted_file