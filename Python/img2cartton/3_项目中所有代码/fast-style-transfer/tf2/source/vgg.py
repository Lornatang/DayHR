# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import tensorflow as tf
import numpy as np
import scipy.io

MEAN_PIXEL = np.array([125.555, 125.555, 125.555])


def build(data_to_path, inputs):
    """ Create VGG based neural unit model

    Args:
        data_to_path: Data set file path.
        inputs: input image tensor.

    Returns:
        The generated model file.
    """
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',    # Block 1

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',    # Block 2

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',  # Block 3
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',  # Block 4
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',  # Block 5
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    # load data.
    dataset = scipy.io.loadmat(data_to_path)
    # Neural layer weights were obtained.
    weights = dataset['layers'][0]

    model = {}
    # Read sequence Numbers and nerve layers sequentially.
    for index, layer in enumerate(layers):
        # get current layer name.
        layer_name = layer[:4]

        if layer_name == 'conv':
            kernels, bias = weights[index][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            # Similar to the deconvolution of TensorFlow.
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            inputs = _conv_ops(inputs, kernels, bias)

        elif layer_name == 'relu':
            # np.maximum(0, image)
            inputs = _relu_ops(inputs)

        elif layer_name == 'pool':
            inputs = _pool_ops(inputs)

        model[layer] = inputs

    assert len(model) == len(layers)
    return model


def _conv_ops(inputs, weights, bias):
    """ Neural convolution.

    Args:
        inputs: Input image tensor.
        weights: The weight of the neural layer.
        bias: The size of the deviation of the neural layer.

    Returns:
        The sum of the convolution and the deviation.
    """
    conv = tf.nn.conv2d(inputs, tf.constant(weights),
                        strides=(1, 1, 1, 1),
                        padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _relu_ops(inputs):
    """ Neural relu.

    Args:
        inputs: Input image tensor.

    Returns:
        Output a value greater than or equal to 0.
    """
    return np.maximum(0, inputs)


def _pool_ops(inputs):
    """ Neural max pool.

    Args:
        inputs: Input image tensor.

    Returns:
        Output a maximum pooled kernel value of 2.
    """
    return tf.nn.max_pool(inputs, (1, 2, 2, 1),
                          strides=(1, 2, 2, 1),
                          padding='SAME')


def preprocess(image):
    """ Preprocessing picture.

    Args:
        image: Input image tensor.

    Returns:
        The trichromatic value after normalization.
    """
    return image - MEAN_PIXEL


def unprocess(image):
    """ Reverse processing pictures.

    Args:
        image: Input image tensor.

    Returns:
        After the inverse normalization.
    """
    return image + MEAN_PIXEL
