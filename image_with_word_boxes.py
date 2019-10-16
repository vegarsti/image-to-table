import io
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
from google.cloud import vision
from shapely import geometry


@dataclass
class Point:
    x: int
    y: int

    def as_tuple(self):
        return self.x, self.y


@dataclass
class Box:
    lower_left_corner: Point
    lower_right_corner: Point
    upper_right_corner: Point
    upper_left_corner: Point
    text: str = None

    def is_inside_box(self, outer_box):  # x \in X
        return outer_box.as_polygon().contains(self.as_polygon()) or outer_box.as_polygon().overlaps(self.as_polygon())

    def point_inside(self, x, y):
        return (
            self.lower_left_corner.x <= x <= self.upper_right_corner.x
            and self.lower_left_corner.y <= y <= self.upper_right_corner.y
        )

    def x_inside(self, x):
        return self.lower_left_corner.x <= x <= self.upper_right_corner.x

    def y_inside(self, y):
        return self.lower_left_corner.y <= y <= self.upper_right_corner.y

    def min_x(self):
        return min(
            corner.x
            for corner in (
                self.lower_left_corner,
                self.lower_right_corner,
                self.upper_right_corner,
                self.lower_right_corner,
            )
        )

    def min_y(self):
        return min(
            corner.y
            for corner in (
                self.lower_left_corner,
                self.lower_right_corner,
                self.upper_right_corner,
                self.lower_right_corner,
            )
        )

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

    def as_polygon(self):
        polygon = geometry.Polygon(
            (p.x, p.y)
            for p in (self.lower_left_corner, self.lower_right_corner, self.upper_right_corner, self.upper_left_corner)
        )
        assert polygon.is_valid
        return polygon

    def plot(self, color):
        x, y = self.as_polygon().exterior.xy
        plt.plot(x, y, color=color)


def detect_text(image_path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    boxes = []
    for i, text in enumerate(texts):
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
        boxes.append(box)
    return boxes


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


if __name__ == "__main__":
    filename = "example.png"
    image = cv2.imread(filename)
    height, width, _ = image.shape
    original_boxes = detect_text(filename)[1:]
    boxes = original_boxes
    max_x = max(box.max_x() for box in boxes)
    x_counts = [len(boxes_in_x(boxes, x)) for x in range(max_x)]
    max_y = max(box.max_y() for box in boxes)
    boxes_in_ys = [boxes_in_y(boxes, y) for y in range(max_y)]
    y_counts = [len(l) for l in boxes_in_ys]
    max_x = width
    max_y = height

    row_boxes = make_row_boxes(y_counts, max_x)
    column_boxes = make_column_boxes(x_counts, max_y)

    number_of_rows = len(row_boxes)
    number_of_columns = len(column_boxes)

    rows = []
    for i, row_box in enumerate(row_boxes):
        new_row = []
        for j, column_box in enumerate(column_boxes):
            row_column = []
            row_column_boxes = []
            for k, original_box in enumerate(original_boxes):
                """
                if i == 0 and j == 2 and k == 6:
                    show_image_with_word_boxes(filename, [row_box, column_box, original_box])
                    print("original inside column")
                    print(original_box.is_inside_box(column_box))
                    print("original box")
                    print(original_box)
                    print("column box")
                    print(column_box)
                    print()
                    print("original inside row")
                    print(original_box.is_inside_box(row_box))
                    print("original box")
                    print(original_box)
                    print("row box")
                    print(row_box)

                    row_box.plot("red")
                    column_box.plot("blue")
                    original_box.plot("black")
                    plt.show()
                """
                if original_box.is_inside_box(column_box) and original_box.is_inside_box(row_box):
                    row_column.append(original_box.text)
                    row_column_boxes.append(original_box)
            new_row.append(" ".join(row_column))
        rows.append(new_row)

    for row in rows:
        print(",".join(row))
    # show_image_with_word_boxes(filename)

"""
import itertools

def intertwine_lists(list1, list2):
    return list(itertools.chain.from_iterable(zip(list1, list2)))
"""
