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

import os

import cv2

if not os.path.exists('./extract_result/'):
    os.mkdir('./extract_result/')


def get_video_pic(path_to_videoname):
    # Read video state
    cap = cv2.VideoCapture(path_to_videoname)
    # Capture its intermediate frame
    cap.set(1, int(cap.get(7) / 2))
    # Checks if video interception is normal, returning True.
    # If normal and False if False.
    status, frame = cap.read()
    if status:
        cv2.imwrite('./extract_result/cover.jpg', frame)
    # Release video resources.
    cap.release()


get_video_pic('../pikachu.mp4')
