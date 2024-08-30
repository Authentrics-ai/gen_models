from pathlib import Path
import subprocess as spr

import click

from .tf import create as create_tf
from .._types import WBTuples, Architecture


def create(architecture: Architecture, weights: WBTuples, save_path: Path):
    tf_dir = save_path.with_name(f"tf_{architecture.name}")
    if not tf_dir.exists():
        create_tf(architecture, weights, save_path)

    assert tf_dir.is_dir()
    proc = spr.run(
        [
            "python",
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            str(tf_dir),
            "--output",
            str(save_path),
        ]
    )
    if proc.returncode:
        click.echo(f"Warning: tf2onnx returned with non-zero return code.")