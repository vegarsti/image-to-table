from typing import Tuple

import cv2
import numpy as np
import opencv_wrapper as cvw
import scipy.ndimage as snd


def find_columns(content: bytes) -> list:
    image = cv2.imdecode(np.frombuffer(content, np.uint8), 1)
    gray = cvw.bgr2gray(image)
    otsu = cvw.threshold_otsu(gray, inverse=True)
    x_axis_sum = np.sum(otsu, axis=0).astype(np.float64)
    # Expand to 2-D for image operations
    x_axis_sum_2d = np.expand_dims(x_axis_sum, axis=0)
    x_axis_sum_2d = cvw.normalize(x_axis_sum_2d).astype(np.uint8)
    num_columns = find_number_of_columns(x_axis_sum_2d)
    column_placement = find_column_placement(x_axis_sum_2d, num_columns)

    return column_placement


def overlay(image: np.ndarray, column_placement: int) -> np.ndarray:
    """Visualizing function"""
    new_image = image.copy()
    new_image[:, column_placement] = 0

    return new_image


def find_number_of_columns(sum_image: np.ndarray) -> int:
    eroded = snd.grey_closing(sum_image, 11)
    otsu = cvw.threshold_otsu(eroded)
    dilated_otsu = cvw.dilate(otsu, 3)
    dilated = dilated_otsu[0]  # 1-D
    clipped, *_ = clip(dilated)
    diffed = np.diff(clipped)
    num_changes = np.count_nonzero(diffed)
    num_columns = (num_changes + 2) // 2

    return num_columns


def clip(image) -> Tuple[np.ndarray, np.ndarray, int]:
    first_white = np.argmax(image)
    last_white = image.shape[0] - np.argmax(image[::-1])
    image = image[first_white:last_white]

    return image, first_white, last_white


def find_column_placement(sum_image, num_columns) -> list:
    thresh_sum = cvw.threshold_binary(sum_image, 1)
    thresh_sum = thresh_sum[0]  # 1-D
    clipped, first_white, _ = clip(thresh_sum)
    is_zero = np.where(clipped == 0)[0]
    column_breaks = sorted(consecutive(is_zero), key=len)
    merge_columns_until_limit(clipped, column_breaks, num_columns)

    return list(map(lambda x: first_white + x[-1], column_breaks))


def merge_columns_until_limit(image, column_breaks, limit) -> None:
    while len(column_breaks) + 1 > limit:
        image[column_breaks.pop(0)] = 255


# https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
