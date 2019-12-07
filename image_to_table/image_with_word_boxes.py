import io
from typing import List, Iterable
import bisect

import numpy as np
import cv2
from google.cloud import vision

from image_to_table.models import TextBox
from opencv_wrapper import Rect


def create_text_box(text) -> TextBox:
    tl, tr, br, bl = text.bounding_poly.vertices
    rect = Rect(tl.x, tl.y, br.x - tl.x, br.y - tl.y)
    text_box = TextBox(text.description, rect)

    return text_box


def detect_text(filename: str) -> Iterable[TextBox]:
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
    description = " ".join(x.text for x in boxes)

    return TextBox(description, bounding_rect)


def extract_table_from_image(filename: str, placement: List[int]) -> List[List[TextBox]]:
    image = cv2.imread(filename)
    height, width, _ = image.shape
    original_boxes = detect_text(filename)
    boxes = original_boxes

    boxes_sorted_by_height = sorted(boxes, key=lambda box: box.rect.y)

    rows = merge_into_rows(boxes_sorted_by_height)
    table = merge_into_columns(rows, placement)

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


def merge_into_columns(rows: List[List[TextBox]], placements: List[int]) -> List[List[TextBox]]:
    bulks = []
    for row in rows:
        bulk = []
        row = sorted(row, key=lambda x: x.rect.tl.x)
        top_right_xs = list(map(lambda x: x.rect.tr.x, row))

        last_index = 0
        for placement in placements:
            index = bisect.bisect_left(top_right_xs, placement, lo=last_index)
            bulk.append(merge_sorted_text_boxes(row[last_index:index]))
            last_index = index

        bulk.append(merge_sorted_text_boxes(row[last_index:]))
        bulks.append(bulk)

    return bulks
