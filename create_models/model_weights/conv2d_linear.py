"""NN for Conv2D feeding into a linear classifier.

Note: This will be a valid model for a 3-channel 10x10 image. 
Otherwise, there will be a shape mismatch between the convolutional
and linear layers.
"""


from . import conv2d
from . import linear

WEIGHTS_BIASES = list()
WEIGHTS_BIASES.extend(conv2d.WEIGHTS_BIASES)
WEIGHTS_BIASES.extend(linear.WEIGHTS_BIASES)