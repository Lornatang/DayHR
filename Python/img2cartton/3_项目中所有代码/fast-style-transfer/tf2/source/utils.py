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


import scipy.misc
import numpy as np
import os

import cv2


def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, img)
    # scipy.misc.imsave(out_path, img)


def scale_img(style_path, style_scale):
    scale = float(style_scale)

    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = _get_img(style_path, img_size=new_shape)
    return style_target


def _get_img(src, img_size=False):
    img = cv2.imread(src)
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    if img_size:
        img = scipy.misc.imresize(img, img_size)
    return img


def exists(p, msg):
    assert os.path.exists(p), msg


def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files
