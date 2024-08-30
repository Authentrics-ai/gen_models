from pathlib import Path
import numpy as np
import keras

from ..model_weights.common import WBTuples


def transpose_linear_weights(weights: WBTuples) -> WBTuples:
    for i in range(len(weights)):
        weights[i] = (weights[i][0].T, weights[i][1])
    return weights


def transpose_conv_weights(weights: WBTuples) -> WBTuples:
    for i in range(len(weights)):
        weights[i] = (np.moveaxis(weights[i][0], (0, 1, 2, 3), (2, 3, 1, 0)), weights[i][1])
    return weights


def linear(weights: WBTuples) -> keras.Model:
    assert len(weights) == 3
    weights = transpose_linear_weights(weights)
        
    linear_layers = [
        keras.layers.Dense(12, activation=keras.activations.relu),
        keras.layers.Dense(8, activation=keras.activations.sigmoid),
        keras.layers.Dense(4, activation=keras.activations.tanh),
    ]
    model = keras.Sequential([keras.Input((6,)), *linear_layers])
    model.build()
    
    for layer, wb in zip(model.layers, weights):
        layer.set_weights(wb)

    return model


def conv2d(weights: WBTuples) -> keras.Model:
    assert len(weights) == 3
    weights = transpose_conv_weights(weights)

    conv_layers = [
        keras.layers.Conv2D(8, 3, activation=keras.activations.relu),
        keras.layers.Conv2D(16, 5),
        keras.layers.Conv2D(6, 3, activation=keras.activations.sigmoid),
    ]

    model = keras.Sequential(
        [keras.Input((None, None, 3)), *conv_layers, keras.layers.MaxPool2D()]
    )
    model.build()

    for layer, wb in zip(model.layers[:3], weights):
        layer.set_weights(wb)

    return model


def conv2d_linear(weights: WBTuples) -> keras.Model:
    assert len(weights) == 6
    conv_weights = transpose_conv_weights(weights[:3])
    linear_weights = transpose_linear_weights(weights[3:])

    conv_layers: list[keras.Layer] = [
        keras.layers.Conv2D(8, 3, activation=keras.activations.relu),
        keras.layers.Conv2D(16, 5),
        keras.layers.Conv2D(6, 3, activation=keras.activations.sigmoid),
    ]
    linear_layers = [
        keras.layers.Dense(12, activation=keras.activations.relu),
        keras.layers.Dense(8, activation=keras.activations.sigmoid),
        keras.layers.Dense(4, activation=keras.activations.tanh),
    ]
    model = keras.Sequential(
        [
            keras.Input((None, None, 3)),
            *conv_layers,
            keras.layers.MaxPool2D(),
            *linear_layers,
        ]
    )
    model.build()

    for layer, wb in zip(model.layers[:3], conv_weights):
        layer.set_weights(wb)

    for layer, wb in zip(model.layers[4:], linear_weights):
        layer.set_weights(wb)

    return model


MODELS = {"linear": linear, "conv2d": conv2d, "conv2d_linear": conv2d_linear}


def create(model_name: str, weights: WBTuples, save_dir: Path):
    model = MODELS[model_name](weights)
    model.save(save_dir / f"keras_{model_name}.keras")
