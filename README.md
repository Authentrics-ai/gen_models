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
pip install .
```