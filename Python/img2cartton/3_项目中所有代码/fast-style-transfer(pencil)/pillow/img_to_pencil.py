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


from PIL import Image, ImageFilter, ImageOps

img = Image.open('./imgs/0.jpg')


def dodge(a, b, alpha):
    return min(int(a * 255 / (256 - b * alpha)), 255)


def draw(img, blur=25, alpha=1.0):
    img1 = img.convert('L')  # The picture is gray
    img2 = img1.copy()
    img2 = ImageOps.invert(img2)
    for i in range(blur):  # The fuzzy degree of
        img2 = img2.filter(ImageFilter.BLUR)
    width, height = img1.size
    for x in range(width):
        for y in range(height):
            a = img1.getpixel((x, y))
            b = img2.getpixel((x, y))
            img1.putpixel((x, y), dodge(a, b, alpha))
    img1.show()
    img1.save('./result/pencil_0.jpg')


draw(img)
