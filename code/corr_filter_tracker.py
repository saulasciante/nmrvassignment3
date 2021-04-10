import numpy as np
import cv2
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mclr

sys.path.append('../toolkit-dir/utils')

from ex3_utils import get_patch, create_gauss_peak, create_cosine_window, show_image, Tracker
from numpy.fft import fft2, ifft2


def construct_filter(patch, gaussian, lmbd):
    patch_fft = fft2(patch)
    patch_fft_conj = np.conjugate(patch_fft)
    gaussian_fft = fft2(gaussian)

    filter_fft = np.divide(
        np.multiply(gaussian_fft, patch_fft_conj),
        lmbd + np.multiply(patch_fft, patch_fft_conj)
    )

    return filter_fft


class CorrFilterTracker(Tracker):

    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        self.gaussian = create_gauss_peak((int(self.window), int(self.window)), self.parameters.gaussian_sigma)
        self.patch_size = self.gaussian.shape

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])
        self.cosine_window = create_cosine_window(self.patch_size)

        patch, _ = get_patch(image, self.position, self.patch_size)

        # plt.imshow(patch)
        # plt.show()

        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = np.multiply(patch, self.cosine_window)

        self.filter_fft_conj = construct_filter(patch, self.gaussian, self.parameters.filter_lambda)

    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0],
                    self.size[1]], 0

        # patch = image[int(top):int(bottom), int(left):int(right)]
        patch, _ = get_patch(image, self.position, self.patch_size)
        patch = np.multiply(patch, self.cosine_window)

        # LOCALIZATION STEP
        patch_fft = fft2(patch)

        corr_response = ifft2(
            np.multiply(
                self.filter_fft_conj,
                patch_fft
            )
        )

        y_max, x_max = np.unravel_index(corr_response.argmax(), corr_response.shape)

        if x_max > patch.shape[0] / 2:
            x_max = x_max - patch.shape[0]
        if y_max > patch.shape[1] / 2:
            y_max = y_max - patch.shape[1]

        new_x = self.position[0] + x_max
        new_y = self.position[1] + y_max

        self.position = (new_x, new_y)

        # MODEL UPDATE
        patch, _ = get_patch(image, self.position, self.patch_size)
        patch = np.multiply(patch, self.cosine_window)
        new_filter_fft_conj = construct_filter(patch, self.gaussian, self.parameters.filter_lambda)
        self.filter_fft_conj = (1 - self.parameters.update_factor) * self.filter_fft_conj + self.parameters.update_factor * new_filter_fft_conj

        return [new_x, new_y, self.size[0], self.size[1]]


class CFParams():
    def __init__(self):
        self.enlarge_factor = 1
        self.gaussian_sigma = 4
        self.filter_lambda = 1
        self.update_factor = 0.3