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


def build(data_to_path, image):
    """

    Args:
        data_to_path: Data set file path.
        image: input image tensor.

    Returns:

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

    for i, layer in enumerate(layers):
        layer_name = layer[:4]
        if layer_name == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            # Similar to the deconvolution of TensorFlow.
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            image = _conv_layer(image, kernels, bias)
        elif layer_name == 'relu':
            image = tf.nn.relu(image)
        elif layer_name == 'pool':
            image = _pool_layer(image)
        model[layer] = image

    assert len(model) == len(layers)
    return model


def _conv_layer(inputs, weights, bias):
    conv = tf.nn.conv2d(inputs, tf.constant(weights), strides=(1, 1, 1, 1),
                        padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(inputs):
    return tf.nn.max_pool(inputs, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                          padding='SAME')


def preprocess(image):
    return image - MEAN_PIXEL


def unprocess(image):
    return image + MEAN_PIXEL
