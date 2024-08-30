# Create Models

A central repo to construct various toy models and slightly larger models in various formats.

Currently supported formats:

- Keras (`.keras`)
- PyTorch (`.pt`)
- Tensorflow (`saved_model` format)
- ONNX (`.onnx`) created from the PyTorch and Tensorflow files.

Currently implemented model architectures:

- Linear (3 layers, 4-12 nodes per layer, various activation functions)
- Conv2D (3 layers)
- Conv2D_linear (combination of Conv2D feeding into Linear, like an image classifier)

## Installation

I've kept the required versions that we have in the Dynamic and Static repos for compatibility, so installation should be quick if you already have Keras, TF, Torch, Onnx installed:

```bash
$ pip install .
```

## Usage

Then you can simply run it as a command-line tool:

```bash
$ gen_models -h
Usage: gen_models [OPTIONS] [DESTINATION]

Options:
  -h, --help                      Show this message and exit.
  --all                           Shortcut for '-f all -m all'
  --overwrite / --no-overwrite    Overwrite existing models or skip if they
                                  exist
  -f, --formats [torch|tf|keras|tf_onnx|torch_onnx|all]
                                  File formats to generate with each specified
                                  model
  -m, --models [linear|conv2d|conv2d_linear|all]
                                  Model architectures to generate in each
                                  specified format
```