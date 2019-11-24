import base64

import cv2
import matplotlib.pyplot as plt
import numpy as np
import opencv_wrapper as cvw
import scipy.ndimage as snd


def image_as_b64(path):
    with open(path, "rb") as image:
        return base64.b64encode(image.read())


def find_number_of_columns(path, show=False, show1=False):
    image = cv2.imread(path)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray = image
    otsu = cvw.threshold_otsu(gray)
    x_axis_sum = np.sum(otsu, axis=0)
    sum_image_height = 50
    sum_image = np.zeros((sum_image_height, *x_axis_sum.shape))
    sum_image[:] = x_axis_sum
    cv2.normalize(sum_image, sum_image, 0, 255, cv2.NORM_MINMAX)
    sum_image = sum_image.astype(np.uint8)
    # sum_image = cvw.normalize(sum_image).astype(np.uint8)
    sum_image = cv2.resize(sum_image, (400, 50), interpolation=cv2.INTER_CUBIC)
    # sum_image = cvw.resize(sum_image, shape=(50, 400)) # Kommer snart :)

    if show:
        plt.figure()
        plt.imshow(sum_image)

    eroded = snd.grey_opening(sum_image, 11)

    if show:
        plt.figure()
        plt.imshow(eroded)

    _, otsu = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # otsu = cvw.threshold_otsu(eroded)
    dilated_otsu = otsu  # cv2.dilate(otsu, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    # otsu = cvw.dilate(otsu, 3)
    # eroded = cv.erode(otsu, cv.getStructuringElement(cv.MORPH_RECT, (35, 35)))
    if show1:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(gray)
        ax2.imshow(dilated_otsu)
        plt.show()
    dilated = dilated_otsu
    dilated = dilated[0]
    first_black = np.argmin(dilated)
    last_black = dilated.shape[0] - np.argmin(dilated[::-1])
    # clip
    dilated = dilated[first_black:last_black]
    diffed = np.diff(dilated)
    num_changes = np.count_nonzero(diffed)
    num_columns = (num_changes + 2) // 2
    return num_columns
