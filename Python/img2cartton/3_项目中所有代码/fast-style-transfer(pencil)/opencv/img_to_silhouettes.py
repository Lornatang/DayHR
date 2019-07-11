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


def cartoonise(picture_name):
    """ example neural style convert

    Args:
      picture_name: img name.

    """
    raw_img = picture_name
    out_img = './result/silhouetters_0.jpg'
    num_down = 2  # Reduce the number of pixel samples
    num_bilateral = 7  # Define the number of bilateral filters
    img_rgb = cv2.imread(raw_img)  # read img
    # Lower sampling with gaussian pyramid
    img_color = img_rgb
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    # Replace a large filter with a small bilateral filter
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(
            img_color, d=9, sigmaColor=9, sigmaSpace=7)
    # Lift sampled image to original size
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    # Convert to grayscale and give it a medium blur
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    # Detect edges and enhance them
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=9,
                                     C=2)
    # Convert back to color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    # Save the converted image
    cv2.imwrite(out_img, img_cartoon)


cartoonise('./imgs/0.jpg')
