import io

import cv2
from google.cloud import vision

from models import Box, Point


def detect_text(filename):
    client = vision.ImageAnnotatorClient()
    with io.open(filename, "rb") as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    boxes = [Box.from_vision_object(text) for text in texts]
    bounding_box, *word_boxes = boxes
    return word_boxes


def from_vision_object(text):
    lower_left_corner, lower_right_corner, upper_right_corner, upper_left_corner = [
        Point(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices
    ]
    box = Box(
        text=text.description,
        lower_left_corner=lower_left_corner,
        lower_right_corner=lower_right_corner,
        upper_left_corner=upper_left_corner,
        upper_right_corner=upper_right_corner,
    )
    return box


def show_image_with_word_boxes(filename, boxes=None):
    image = cv2.imread(filename)
    color = (0, 0, 0)
    thickness = 2
    # skip first, large box with everything
    if boxes is None:
        boxes = []
    for box in boxes:
        image = cv2.rectangle(
            image, box.lower_left_corner.as_tuple(), box.upper_right_corner.as_tuple(), color, thickness
        )
    cv2.imshow("table", image)
    while True:
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to break
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


def extract_table_from_image(filename):
    image = cv2.imread(filename)
    height, width, _ = image.shape
    original_boxes = detect_text(filename)
    boxes = original_boxes
    x_counts = [len(boxes_in_x(boxes, x)) for x in range(width)]
    boxes_in_ys = [boxes_in_y(boxes, y) for y in range(height)]
    y_counts = [len(l) for l in boxes_in_ys]
    row_boxes = make_row_boxes(y_counts, width)
    column_boxes = make_column_boxes(x_counts, height)
    table = []
    for row_box in row_boxes:
        new_row = []
        for column_box in column_boxes:
            row_column = []
            for original_box in original_boxes:
                if original_box.is_inside_box(column_box) and original_box.is_inside_box(row_box):
                    row_column.append(original_box.text)
            new_row.append(" ".join(row_column))
        table.append(new_row)
    return table


if __name__ == "__main__":
    filename = "example.png"
    table = extract_table_from_image(filename)

    for row in table:
        print(",".join(row))
