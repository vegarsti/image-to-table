import base64

import cv2
import matplotlib.pyplot as plt
import numpy as np
import opencv_wrapper as cvw
import scipy.ndimage as snd



def find_columns(path):
    image = cv2.imread(path)
    gray = cvw.bgr2gray(image)

    otsu = cvw.threshold_otsu(gray, inverse=True)

    x_axis_sum = np.sum(otsu, axis=0).astype(np.float64)

    # Expand to 2-D for image operations
    sum_image = np.expand_dims(x_axis_sum, axis=0)
    sum_image = cvw.normalize(sum_image).astype(np.uint8)

    num_columns = find_number_of_columns(sum_image)
    column_placement = find_column_placement(sum_image, num_columns)

    return num_columns, column_placement


def overlay(image, column_placement):
    """Visualizing function"""
    new_image = image.copy()
    new_image[:, column_placement] = 0
    return new_image


def find_number_of_columns(sum_image):
    eroded = snd.grey_closing(sum_image, 11)
    otsu = cvw.threshold_otsu(eroded)
    dilated_otsu = cvw.dilate(otsu, 3)

    dilated = dilated_otsu[0]  # 1-D
    clipped, *_ = clip(dilated)
    diffed = np.diff(clipped)
    num_changes = np.count_nonzero(diffed)
    num_columns = (num_changes + 2) // 2

    return num_columns


def clip(image):
    first_white = np.argmax(image)
    last_white = image.shape[0] - np.argmax(image[::-1])
    image = image[first_white:last_white]

    return image, first_white, last_white


def find_column_placement(sum_image, num_columns):
    thresh_sum = cvw.threshold_binary(sum_image, 1)
    thresh_sum = thresh_sum[0]  # 1-D
    clipped, first_white, _ = clip(thresh_sum)

    is_zero = np.where(clipped == 0)[0]
    column_breaks = sorted(consecutive(is_zero), key=len)

    while len(column_breaks) + 1 > num_columns:
        clipped[column_breaks.pop(0)] = 255

    return list(map(lambda x: first_white + x[-1], column_breaks))


# https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
