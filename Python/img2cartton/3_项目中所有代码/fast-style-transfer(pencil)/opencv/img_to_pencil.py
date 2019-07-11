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


import cv2
import numpy as np

import time
import os


if not os.path.exists('./result/'):
    os.mkdir('./result/')


# plearse d'not use it, because its use very solw.
def dodge_naive(image, mask):
    """ Fusion algorithm of gray image and gaussian fuzzy film.

    Args:
      image: Input image algorithm data stream.
      mask: Image data stream after gauss processing.

    Returns:
      np.array(**kawgs).
    """
    # determine the shape of the input image
    width, height = image.shape[:2]

    # prepare output argument with same size as image
    blend = np.zeros((width, height), np.uint8)

    for col in range(width):
        for row in range(height):
            # do for every pixel
            if mask[col, row] == 255:
                # avoid division by zero
                blend[col, row] = 255
            else:
                # shift image pixel value by 8 bits
                # divide by the inverse of the mask
                tmp = (image[col, row] << 8) / (255 - mask)
                # print('tmp={}'.format(tmp.shape))
                # make sure resulting value stays within bounds
                if tmp.any() > 255:
                    tmp = 255
                    blend[col, row] = tmp

    return blend


def dodge(image, mask):
    """ Use a faster packaging algorithm to achieve dodge_naive(image, mask) function.

    Args:
      image: Input image algorithm data stream.
      mask: Image data stream after gauss processing.

    Returns:
      cv2.divide(**kawgs).
    """
    return cv2.divide(image, 255 - mask, scale=256)


def rgb_to_sketch(inputs, outputs):
    """ Convert color image to pixel image.

    Args:
      inputs: absolute path to the image to be processed.
      outputs: Absolute path after processing.

    Returns:
      cv2.imwrite(**kawgs).

    """
    # step 1: Read the image and convert it to grayscale.
    img_rgb = cv2.imread(inputs)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # option: The image is directly converted to grayscale when read.
    # img_gray = cv2.imread('*.jpg', cv2.IMREAD_GRAYSCALE)

    # step 2: Reverse colours.
    img_gray_inv = 255 - img_gray
    # step 3: Perform gaussian blur operation.
    # The parameter ksize represents the size of the gaussian kernel.
    # SigmaX and sigmaY represent the standard deviation of the gaussian
    # kernel in the X and Y directions respectively.
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
                                sigmaX=0, sigmaY=0)
    # step 5: Fusion of grayscale and gaussian blur film.
    img_blend = dodge(img_gray, img_blur)

    # save after processing imgs.
    cv2.imwrite(outputs, img_blend)


if __name__ == '__main__':
    src_img = './imgs/0.jpg'
    dst_img = './result/pencil_0.jpg'

    # cal time
    start = time.time()
    rgb_to_sketch(src_img, dst_img)
    end = time.time()
    print(f'Time: {end - start:.5f}s.')
