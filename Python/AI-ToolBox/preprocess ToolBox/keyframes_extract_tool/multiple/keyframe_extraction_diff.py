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

"""
Copy from https://blog.csdn.net/qq_21997625/article/details/81285096

keyframes extract tool

this key frame extract algorithm is based on interframe difference.

The principle is very simple
First, we load the video and compute the interframe difference between each frames

Then, we can choose one of these three methods to extract keyframes, which are
all based on the difference method:

1. use the difference order
    The first few frames with the largest average interframe difference
    are considered to be key frames.
2. use the difference threshold
    The frames which the average interframe difference are large than the
    threshold are considered to be key frames.
3. use local maximum
    The frames which the average interframe difference are local maximum are
    considered to be key frames.
    It should be noted that smoothing the average difference value before
    calculating the local maximum can effectively remove noise to avoid
    repeated extraction of frames of similar scenes.

After a lot of implementation, it is found that the method extracted from the
third method is the best.
"""

import os

import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

if not os.path.exists('./extract_result/'):
    os.mkdir('./extract_result/')

# Video path of the source file
video_to_path = '../video.mp4'
# Directory to store the processed frames
dirs = './extract_result/'


def smooth(x, window_len=13, window='hanning'):
    """ smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Args:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

    Returns:
        the smoothed signal

    Examples:
        import numpy as np
        t = np.linspace(-2,2,0.1)
        x = np.sin(t)+np.random.randn(len(t))*0.1
        y = smooth(x)

    see also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

    """
    print(len(x), window_len)

    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


class Frame:
    """class to hold information about each frame
    """

    def __init__(self, video_id, diff):
        self.id = video_id
        self.diff = diff

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def rel_change(exist, after):
    """ The difference between the largest and smallest frames

    Args:
        exist: Frame currently intercepted by video.
        after: Capture the next frame after video.

    Returns:
        The difference between frames.

    """
    diff = (after - exist) / max(exist, after)
    return diff


if __name__ == "__main__":
    # Setting fixed threshold criteria
    USE_THRESH = False
    # fixed threshold value
    THRESH = 0.6
    # Setting fixed threshold criteria
    USE_TOP_ORDER = False
    # Setting local maxima criteria
    USE_LOCAL_MAXIMA = True
    # Number of top sorted frames
    NUM_TOP_FRAMES = 50

    # smoothing window size
    len_window = int(50)

    print('target video: `' + video_to_path + '`')
    print('frame save directory: `' + dirs + '`')

    # load video and compute diff between frames
    video_capture = cv2.VideoCapture(str(video_to_path))
    current_frame = None
    preview_frame = None

    # The difference between frames
    frame_diffs = []
    # frames map
    frames = []

    # Start intercepting video
    success, frame = video_capture.read()

    i = 0
    # check Successful interception of video?
    while success:
        # cv2.COLOR_BGR_LUV function.
        # In the case of 8-bit and 16-bit images, R, G, and B are converted to
        # floating point format and scaled to fit the 0 to 1 range.
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        current_frame = luv

        if current_frame is not None and preview_frame is not None:
            # logic here
            diff = cv2.absdiff(current_frame, preview_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])

            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames.append(frame)
        # Sets the next frame to the current frame
        preview_frame = current_frame
        i = i + 1
        # Start intercepting video
        success, frame = video_capture.read()
    # release video.
    video_capture.release()

    # compute keyframe
    keyframe_id_set = set()

    # Caution:
    # This parameter corresponds to the first method, which is now set 'False'.
    # ======================================================================= #
    # Use the order of difference intensitiesWe sorted all the frames according
    # to the average inter-frame difference intensity, and selected
    # some pictures with the highest average inter-frame
    # difference intensity as the key frames of video.
    if USE_TOP_ORDER:
        # sort the list in descending order
        frames.sort(key=operator.attrgetter("diff"), reverse=True)
        for keyframe in frames[:NUM_TOP_FRAMES]:
            keyframe_id_set.add(keyframe.id)

    # Caution:
    # This parameter corresponds to the second method, which is now set 'False'.
    # ======================================================================= #
    # Use differential intensity thresholdsWe select the frame
    # whose average inter-frame difference intensity is higher than
    # the preset threshold as the key frame of video.
    if USE_THRESH:
        print("Using Threshold")
        for i in range(1, len(frames)):
            if (rel_change(np.float(frames[i - 1].diff),
                           np.float(frames[i].diff)) >= THRESH):
                keyframe_id_set.add(frames[i].id)

    # Caution:
    # This parameter corresponds to the thrid method, which is now set 'True'.
    if USE_LOCAL_MAXIMA:
        print("Using Local Maxima")
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        for i in frame_indexes:
            keyframe_id_set.add(frames[i - 1].id)

        plt.figure(figsize=(40, 20))
        plt.locator_params()
        plt.stem(sm_diff_array, use_line_collection=True)
        plt.savefig(dirs + 'plot.png')

    # save all keyframes as image
    cap = cv2.VideoCapture(str(video_to_path))

    current_frame = None
    keyframes = []

    # Start intercepting video
    success, frame = cap.read()
    idx = 0
    # check Successful interception of video?
    while success:
        if idx in keyframe_id_set:
            name = "keyframe_" + str(idx) + ".jpg"
            cv2.imwrite(dirs + name, frame)
            keyframe_id_set.remove(idx)
        idx = idx + 1
        success, frame = cap.read()
    # release video resource
    cap.release()
