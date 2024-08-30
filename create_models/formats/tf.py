from pathlib import Path
import numpy as np
import tensorflow as tf

from .._types import WBTuples, Architecture

def apply(*fns):
    def tmp(x):
        for f in fns:
            x = f(x)
        return x
    
    return tmp

class Linear(tf.Module):
    def __init__(self, weights: WBTuples):
        super().__init__()
        assert len(weights) == 3

        self.lin1_weight = tf.Variable(weights[0][0], trainable=True)
        self.lin1_bias = tf.Variable(weights[0][1].reshape((-1, 1)), trainable=True)
        self.lin2_weight = tf.Variable(weights[1][0], trainable=True)
        self.lin2_bias = tf.Variable(weights[1][1].reshape((-1, 1)), trainable=True)
        self.lin3_weight = tf.Variable(weights[2][0], trainable=True)
        self.lin3_bias = tf.Variable(weights[2][1].reshape((-1, 1)), trainable=True)

        self.lin1 = lambda x: (
            tf.matmul(self.lin1_weight, x) + self.lin1_bias
        )
        self.lin2 = lambda x: (
            tf.matmul(self.lin2_weight, x) + self.lin2_bias
        )
        self.lin3 = lambda x: (
            tf.matmul(self.lin3_weight, x) + self.lin3_bias
        )

        self.forward = apply(
            self.lin1,
            tf.nn.relu,
            self.lin2,
            tf.nn.sigmoid,
            self.lin3,
            tf.nn.tanh,
        )
    
    @tf.function(input_signature=[tf.TensorSpec([6, None], tf.dtypes.float64, "tfModelInput")])
    def __call__(self, x: tf.Tensor):
        return self.forward(x)


def transform_kernel(x: np.ndarray) -> np.ndarray:
    y = np.moveaxis(x, (0, 1, 2, 3), (3, 2, 0, 1))
    return y


class Conv2d(tf.Module):
    def __init__(self, weights: WBTuples):
        super().__init__()
        assert len(weights) == 3

        for i in range(3):
            weights[i] = (transform_kernel(weights[i][0]), weights[i][1])

        self.conv1_filter = tf.Variable(weights[0][0], trainable=True)
        self.conv1_bias = tf.Variable(weights[0][1], trainable=True)
        self.conv2_filter = tf.Variable(weights[1][0], trainable=True)
        self.conv2_bias = tf.Variable(weights[1][1], trainable=True)
        self.conv3_filter = tf.Variable(weights[2][0], trainable=True)
        self.conv3_bias = tf.Variable(weights[2][1], trainable=True)

        self.conv1 = lambda x: (
            tf.nn.conv2d(x, self.conv1_filter, strides=1, padding="VALID")
            + self.conv1_bias
        )
        self.conv2 = lambda x: (
            tf.nn.conv2d(x, self.conv2_filter, strides=1, padding="VALID")
            + self.conv2_bias
        )
        self.conv3 = lambda x: (
            tf.nn.conv2d(x, self.conv3_filter, strides=1, padding="VALID")
            + self.conv3_bias
        )
        self.maxpool = lambda x: tf.nn.max_pool2d(x, 2, 1, "VALID")

        self.forward = apply(
            self.conv1,
            self.conv2,
            tf.nn.relu,
            self.conv3,
            tf.nn.sigmoid,
            self.maxpool,
        )
    
    @tf.function(input_signature=[tf.TensorSpec([None, 10, 10, 1], tf.dtypes.float64, "tfModelInput")])
    def __call__(self, x: tf.Tensor):
        return self.forward(x)


class Conv2dLinear(tf.Module):
    def __init__(self, weights: WBTuples):
        super().__init__()
        assert len(weights) == 6

        for i in range(3):
            weights[i] = (transform_kernel(weights[i][0]), weights[i][1])
        
        self.conv1_filter = tf.Variable(weights[0][0], trainable=True)
        self.conv1_bias = tf.Variable(weights[0][1], trainable=True)
        self.conv2_filter = tf.Variable(weights[1][0], trainable=True)
        self.conv2_bias = tf.Variable(weights[1][1], trainable=True)
        self.conv3_filter = tf.Variable(weights[2][0], trainable=True)
        self.conv3_bias = tf.Variable(weights[2][1], trainable=True)

        self.lin1_weight = tf.Variable(weights[3][0], trainable=True)
        self.lin1_bias = tf.Variable(weights[3][1].reshape(-1, 1), trainable=True)
        self.lin2_weight = tf.Variable(weights[4][0], trainable=True)
        self.lin2_bias = tf.Variable(weights[4][1].reshape(-1, 1), trainable=True)
        self.lin3_weight = tf.Variable(weights[5][0], trainable=True)
        self.lin3_bias = tf.Variable(weights[5][1].reshape(-1, 1), trainable=True)

    
    @tf.function(input_signature=[tf.TensorSpec([None, 10, 10, 1], tf.dtypes.float64, "tfModelInput")])
    def __call__(self, x: tf.Tensor):

        conv1 = lambda x: (
            tf.nn.conv2d(x, self.conv1_filter, strides=1, padding="VALID")
            + self.conv1_bias
        )
        conv2 = lambda x: (
            tf.nn.conv2d(x, self.conv2_filter, strides=1, padding="VALID")
            + self.conv2_bias
        )
        conv3 = lambda x: (
            tf.nn.conv2d(x, self.conv3_filter, strides=1, padding="VALID")
            + self.conv3_bias
        )
        maxpool = lambda x: tf.nn.max_pool2d(x, 2, 1, "VALID")
        reshape = lambda x: tf.reshape(x, (-1, 6, 1))
        lin1 = lambda x: (
            tf.matmul(self.lin1_weight, x) + self.lin1_bias
        )
        lin2 = lambda x: (
            tf.matmul(self.lin2_weight, x) + self.lin2_bias
        )
        lin3 = lambda x: (
            tf.matmul(self.lin3_weight, x) + self.lin3_bias
        )

        forward = apply(
            conv1,
            conv2,
            tf.nn.relu,
            conv3,
            tf.nn.sigmoid,
            maxpool,
            reshape,
            lin1,
            tf.nn.relu,
            lin2,
            tf.nn.sigmoid,
            lin3,
            tf.nn.tanh,
        )
        return forward(x)


MODELS = {Architecture.linear: Linear, Architecture.conv2d: Conv2d, Architecture.conv2d_linear: Conv2dLinear}

def create(architecture: Architecture, weights: WBTuples, save_path: Path):
    model = MODELS[architecture](weights)
    tf.saved_model.save(model, str(save_path), )
