import io
from typing import List
import itertools
import bisect

import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.cloud import vision

from image_to_table.models import Box, TextBox
from opencv_wrapper import Point, Rect


def create_text_box(text) -> TextBox:
    tl, tr, br, bl = (Point(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices)
    rect = Rect(tl.x, tl.y, br.x - tl.x, br.y - tl.y)
    text_box = TextBox(text.description, rect)

    return text_box


def detect_text(filename) -> List[TextBox]:
    client = vision.ImageAnnotatorClient()
    with io.open(filename, "rb") as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    boxes = map(create_text_box, texts)
    bounding_box, *word_boxes = boxes
    return word_boxes


def detect_text_old(filename):
    client = vision.ImageAnnotatorClient()
    with io.open(filename, "rb") as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    boxes = [Box.from_vision_object(text) for text in texts]
    bounding_box, *word_boxes = boxes
    return word_boxes


def show_image_with_word_boxes(filename, boxes=None):
    image = cv2.imread(filename)
    color = (0, 0, 0)
    thickness = 2
    if boxes is None:
        boxes = []
    for box in boxes:
        image = cv2.rectangle(
            image, box.lower_left_corner.as_tuple(), box.upper_right_corner.as_tuple(), color, thickness
        )
    cv2.imshow("table", image)
    while True:
        key = cv2.waitKey(0)
        key_is_esc = key == 27
        if key_is_esc:
            break
    cv2.destroyAllWindows()


def boxes_in_x(boxes, x):
    return [box for box in boxes if box.x_inside(x)]


def boxes_in_y(boxes, y):
    return [box for box in boxes if box.y_inside(y)]


def make_row_boxes(y_counts, max_x):
    inside = False
    starts = []
    stops = []

    for y, n in enumerate(y_counts):
        if n > 0 and not inside:
            starts.append(y - 1)
            inside = True
        if n == 0 and inside:
            stops.append(y)
            inside = False
    if inside:
        stops.append(y)

    box_ys = list(zip(starts, stops))
    row_boxes = [
        Box(
            lower_right_corner=Point(x=max_x, y=stop),
            lower_left_corner=Point(x=0, y=stop),
            upper_right_corner=Point(x=max_x, y=start),
            upper_left_corner=Point(x=0, y=start),
        )
        for (start, stop) in box_ys
    ]
    return row_boxes


def make_column_boxes(x_counts, max_y):
    inside = False
    starts = []
    stops = []

    for x, n in enumerate(x_counts):
        if n > 0 and not inside:
            starts.append(x - 1)
            inside = True
        if n == 0 and inside:
            stops.append(x)
            inside = False
    if inside:
        stops.append(x)

    box_xs = list(zip(starts, stops))
    column_boxes = [
        Box(
            lower_right_corner=Point(y=max_y, x=stop),
            lower_left_corner=Point(y=0, x=stop),
            upper_right_corner=Point(y=max_y, x=start),
            upper_left_corner=Point(y=0, x=start),
        )
        for (start, stop) in box_xs
    ]
    return column_boxes


def show_all_boxes_intersecting(filename, row_boxes, column_boxes):
    for row_box in row_boxes:
        for column_box in column_boxes:
            show_image_with_word_boxes(filename, [row_box, column_box])


def get_row(row_box, column_boxes, original_boxes):
    new_row = [get_cell_in_row(column_box, original_boxes, row_box) for column_box in column_boxes]
    return new_row


def get_cell_in_row(column_box, original_boxes, row_box):
    row_column = []
    for i, original_box in enumerate(original_boxes):
        if original_box.is_inside_box(column_box) and original_box.is_inside_box(row_box):
            row_column.append(original_box.text)
    return " ".join(row_column)


def merge_sorted_text_boxes(boxes: List[TextBox]):
    bounding_rect = boxes[0].rect or boxes[-1].rect
    description = " ".join(map(lambda x: x.text, boxes))

    return TextBox(description, bounding_rect)


def new_extract_table_from_image(filename, num_columns, placement):
    image = cv2.imread(filename)
    height, width, _ = image.shape
    original_boxes = detect_text(filename)
    boxes = original_boxes

    # Sort boxes by row
    sorted_boxes = sorted(boxes, key=lambda box: box.rect.y)

    rows = separate_into_rows(sorted_boxes)
    grid = separate_into_columns(rows, num_columns, placement)

    csv = []

    for row in grid:
        csv.append(",".join(map(lambda x: x.text, (merge_sorted_text_boxes(col) for col in row))))

    import pprint

    pprint.pprint(csv)


def separate_into_rows(sorted_boxes):
    boxes = sorted_boxes

    group_counts = []
    groups = []

    for group_count, group in itertools.groupby(boxes, key=lambda x: x.rect.y):
        group_counts.append(group_count)
        groups.append(list(group))

    diffs = np.diff(group_counts)
    rows = [groups[0]]
    row_idx = 0
    for i, diff in enumerate(diffs > 5, start=1):
        if diff:
            row_idx += 1
            rows.append([])
        rows[row_idx] += groups[i]

    return rows


def separate_into_columns(rows: List[List[TextBox]], num_columns: int, placements: List[int]):
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


def extract_table_from_image(filename):
    image = cv2.imread(filename)
    height, width, _ = image.shape
    original_boxes = detect_text_old(filename)
    boxes = original_boxes
    for box in original_boxes:
        box.fill("black")
    x_counts = [len(boxes_in_x(boxes, x)) for x in range(width)]
    boxes_in_ys = [boxes_in_y(boxes, y) for y in range(height)]
    y_counts = [len(l) for l in boxes_in_ys]
    row_boxes = make_row_boxes(y_counts, width)
    column_boxes = make_column_boxes(x_counts, height)
    """
    for box in row_boxes + column_boxes:
        box.plot(color="black")
    """
    plt.axis("off")
    plt.savefig("table_b.png")
    plt.show()

    table = [get_row(row_box, column_boxes, original_boxes) for row_box in row_boxes]

    return table
