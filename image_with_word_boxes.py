import io
from dataclasses import dataclass

import cv2
from google.cloud import vision


@dataclass
class Corner:
    x: int
    y: int

    def as_tuple(self):
        return self.x, self.y


@dataclass
class Box:
    lower_left_corner: Corner
    lower_right_corner: Corner
    upper_right_corner: Corner
    upper_left_corner: Corner

    def point_inside(self, x, y):
        return (
            self.lower_left_corner.x <= x <= self.upper_right_corner.x
            and self.lower_left_corner.y <= y <= self.upper_right_corner.y
        )

    def x_inside(self, x):
        return self.lower_left_corner.x <= x <= self.upper_right_corner.x

    def y_inside(self, y):
        return self.lower_left_corner.y <= y <= self.upper_right_corner.y

    def max_x(self):
        return max(
            corner.x
            for corner in (
                self.lower_left_corner,
                self.lower_right_corner,
                self.upper_right_corner,
                self.lower_right_corner,
            )
        )

    def max_y(self):
        return max(
            corner.y
            for corner in (
                self.lower_left_corner,
                self.lower_right_corner,
                self.upper_right_corner,
                self.lower_right_corner,
            )
        )


def detect_text(image_path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    boxes = []
    for text in texts:
        corners = [Corner(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        box = Box(*corners)
        boxes.append(box)
    return boxes


def show_image_with_word_boxes(filename, boxes):
    image = cv2.imread(filename)
    height, width, channels = image.shape
    color = (0, 0, 0)
    thickness = 2
    # skip first, large box with everything
    for box in boxes:
        image = cv2.rectangle(
            image, box.upper_left_corner.as_tuple(), box.lower_right_corner.as_tuple(), color, thickness
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


filename = "example.png"

image = cv2.imread(filename)
height, width, _ = image.shape

boxes = detect_text(filename)[1:]

max_x = max(box.max_x() for box in boxes)
x_counts = [len(boxes_in_x(boxes, x)) for x in range(max_x)]

max_y = max(box.max_y() for box in boxes)
boxes_in_ys = [boxes_in_y(boxes, y) for y in range(max_y)]
y_counts = [len(l) for l in boxes_in_ys]

print(f"height {height}, width {width}, max_x {max_x}, max_y {max_y}")

max_x = width
max_y = height


def make_row_boxes():
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
            lower_right_corner=Corner(x=max_x, y=stop),
            lower_left_corner=Corner(x=0, y=stop),
            upper_right_corner=Corner(x=max_x, y=start),
            upper_left_corner=Corner(x=0, y=start),
        )
        for (start, stop) in box_ys
    ]
    return row_boxes


def make_column_boxes():
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
            lower_right_corner=Corner(y=max_y, x=stop),
            lower_left_corner=Corner(y=0, x=stop),
            upper_right_corner=Corner(y=max_y, x=start),
            upper_left_corner=Corner(y=0, x=start),
        )
        for (start, stop) in box_xs
    ]
    return column_boxes


row_boxes = make_row_boxes()
column_boxes = make_column_boxes()

number_of_rows = len(row_boxes)
number_of_columns = len(column_boxes)

print(f"{number_of_rows} rows, {number_of_columns} columns")


def show_all_boxes_intersecting(filename, row_boxes, column_boxes):
    for row_box in row_boxes:
        for column_box in column_boxes:
            show_image_with_word_boxes(filename, [row_box, column_box])


show_all_boxes_intersecting(filename, row_boxes, column_boxes)

"""
import itertools

def intertwine_lists(list1, list2):
    return list(itertools.chain.from_iterable(zip(list1, list2)))
"""
