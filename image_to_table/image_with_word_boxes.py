import io
from typing import List
import itertools
import bisect

import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.cloud import vision

from image_to_table.models import TextBox
from opencv_wrapper import Rect


def create_text_box(text) -> TextBox:
    tl, tr, br, bl = text.bounding_poly.vertices
    rect = Rect(tl.x, tl.y, br.x - tl.x, br.y - tl.y)
    text_box = TextBox(text.description, rect)

    return text_box


def detect_text(filename: str) -> List[TextBox]:
    client = vision.ImageAnnotatorClient()
    with io.open(filename, "rb") as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    _summary, *texts = response.text_annotations
    boxes = map(create_text_box, texts)

    return boxes


def merge_sorted_text_boxes(boxes: List[TextBox]) -> TextBox:
    bounding_rect = boxes[0].rect or boxes[-1].rect
    description = " ".join(map(lambda x: x.text, boxes))

    return TextBox(description, bounding_rect)


def extract_table_from_image(filename: str, num_columns: int, placement: List[int]) -> List[List[str]]:
    image = cv2.imread(filename)
    height, width, _ = image.shape
    original_boxes = detect_text(filename)
    boxes = original_boxes

    # Sort boxes by row
    sorted_boxes = sorted(boxes, key=lambda box: box.rect.y)

    rows = merge_into_rows(sorted_boxes)
    table = separate_into_columns(rows, placement)

    for i in range(len(table)):
        table[i] = list(map(lambda x: x.text, (merge_sorted_text_boxes(col) for col in table[i])))

    return table


def merge_into_rows(boxes: List[TextBox], max_distance: int = 5) -> List[List[TextBox]]:
    """Merges text boxes with similar heights into rows

    :param max_distance: Maximum height difference between rows to be considered the same row.
    """
    ys = np.asarray(list(map(lambda x: x.rect.y, boxes)))
    diffs = np.diff(ys)
    indices = np.where(diffs > max_distance)[0] + 1

    rows = [boxes[: indices[0]]]
    for i in range(len(indices) - 1):
        rows.append(boxes[indices[i] : indices[i + 1]])

    return rows


def separate_into_columns(rows: List[List[TextBox]], placements: List[int]):
    bulks = []
    for row in rows:
        bulk = []
        row = sorted(row, key=lambda x: x.rect.tl.x)
        tr_xs = list(map(lambda x: x.rect.tr.x, row))

        last_index = 0
        for placement in placements:
            index = bisect.bisect_left(tr_xs, placement, lo=last_index)
            bulk.append(row[last_index:index])

            last_index = index

        bulk.append(row[last_index:])
        bulks.append(bulk)

    return bulks
