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


# Import and configure modules
import time
import os
import tensorflow as tf

import matplotlib.pyplot as plt

if not os.path.exists('./result/'):
    os.mkdir('./result/')

# Download images and choose a style image and a content image:
content_path = './imgs/turtle.jpg'

style_path = './imgs/kandinsky.jpg'


# Visualize the input
def load_img(path_to_img):
    """ Define a function to load an image and limit its maximum dimension
        to 512 pixels.

    Args:
      path_to_img: The absolute path of the image.

    """
    max_dim = 512
    # step 1: process img (read->decode->convert).
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    # step 2: Get img size.
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    # step 3: Update img shape
    new_shape = tf.cast(shape * scale, tf.int32)

    # setp 4: Output a tensor stream data.
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


# Load the image to be processed
content_image = load_img(content_path)
style_image = load_img(style_path)


def imshow(img, title=None):
    """ Create a simple function to display an image.

    Args:
        img: Tensor for img.
        title: Displays the name of the image.

    """
    if len(img.shape) > 3:
        img = tf.squeeze(img, axis=0)

    plt.imshow(img)
    if title:
        plt.title(title)


def print_top_5():
    """ Print out the five possible object categories identified by the model.
    """

    # Define content and style representations
    # step 1: Processing of images to provide training.
    x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
    x = tf.image.resize(x, (224, 224))
    # step 2: Load the pretraining model.
    model = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    # step 3: Classify the possible categories of images.
    prediction_probabilities = model(x)

    # Predict the top five possible categories
    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(
        prediction_probabilities.numpy())[0]

    print('Score in the first five categories of the image:')
    print([(class_name, prob)
           for (number, class_name, prob) in predicted_top_5])


# Now load a VGG19 without the classification head, and list the layer names
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')


# Choose intermediate layers from the network to represent the
# style and content of the image:
# Content layer where will pull our features maps
content_layers = ['block5_conv2']

# Style layer of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# build the model
def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values.

    Args:
        layer_names: The name of each neuron layer.

    Returns:
        tf.keras.Model(**kawgs).

    """
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(names).output for names in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


# And to create the model.
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image * 255)


# Calculate style
def gram_matrix(input_tensor):
    """ The content of an image is represented by the values of the intermediate feature maps.

    Args:
        input_tensor: Input tensor stream.

    Returns:
        Take the average of the cross product.

    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


# Extract style and content
# Build a model that returns the style and content tensors.
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs, **kwargs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

style_results = results['style']

# Run gradient descent
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

image = tf.Variable(content_image)


def clip_0_1(img):
    """ Clips tensor values to a specified min and max.

    Args:
        img: input imgs tensor.

    Returns:
        A clipped Tensor or IndexedSlices.

    Raises:
        ValueError: If the clip tensors would trigger array broadcasting
                    that would make the returned tensor larger than the input.

    """
    return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)


opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

style_weight = 1e-2
content_weight = 1e4


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] -
                                           style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] -
                                             content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


# Total variation loss
def high_pass_x_y(img):
    x_var = img[:, :, 1:, :] - img[:, :, :-1, :]
    y_var = img[:, 1:, :, :] - img[:, :-1, :, :]

    return x_var, y_var


def total_variation_loss(img):
    x_deltas, y_deltas = high_pass_x_y(img)
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)


# Re-run the optimization
total_variation_weight = 1e8


@tf.function()
def train_step(img):
    with tf.GradientTape() as tape:
        outputs = extractor(img)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * total_variation_loss(img)

        grad = tape.gradient(loss, img)
        opt.apply_gradients([(grad, img)])
        img.assign(clip_0_1(img))


def train(epochs, steps_per_epoch):
    img = tf.Variable(content_image)
    start = time.time()

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(img)
        imshow(img.read_value())
        plt.title("Train step: {}".format(step))
        plt.show()

    end = time.time()
    print("Total time: {:.1f}".format(end - start))

    file_name = './result/kadinsky-turtle.png'
    plt.imsave(file_name, img[0])


if __name__ == '__main__':
    train(epochs=10, steps_per_epoch=100)
