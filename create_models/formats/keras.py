from pathlib import Path
import numpy as np
import keras

from .._types import WBTuples, Architecture


def transpose_linear_weights(weights: WBTuples) -> WBTuples:
    for i in range(len(weights)):
        weights[i] = (weights[i][0].T, weights[i][1])
    return weights


def transpose_conv_weights(weights: WBTuples) -> WBTuples:
    for i in range(len(weights)):
        weights[i] = (np.moveaxis(np.copy(weights[i][0]), (0, 1, 2, 3), (3, 2, 0, 1)), weights[i][1])
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
        keras.layers.Conv2D(2, 3, activation=keras.activations.relu),
        keras.layers.Conv2D(6, 5),
        keras.layers.Conv2D(6, 3, activation=keras.activations.sigmoid),
    ]

    model = keras.Sequential(
        [keras.Input((None, None, 1)), *conv_layers, keras.layers.MaxPool2D()]
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
        keras.layers.Conv2D(2, 3, activation=keras.activations.relu),
        keras.layers.Conv2D(6, 5),
        keras.layers.Conv2D(6, 3, activation=keras.activations.sigmoid),
    ]
    linear_layers = [
        keras.layers.Dense(12, activation=keras.activations.relu),
        keras.layers.Dense(8, activation=keras.activations.sigmoid),
        keras.layers.Dense(4, activation=keras.activations.tanh),
    ]
    model = keras.Sequential(
        [
            keras.Input((None, None, 1)),
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


MODELS = {Architecture.linear: linear, Architecture.conv2d: conv2d, Architecture.conv2d_linear: conv2d_linear}


def create(architecture: Architecture, weights: WBTuples, save_path: Path):
    model = MODELS[architecture](weights)
    model.save(save_path)
