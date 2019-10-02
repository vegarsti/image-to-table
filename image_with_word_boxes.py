import io
from dataclasses import dataclass
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
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
    color = (0, 0, 0)
    thickness = 2
    # skip first, large box
    for box in boxes[1:]:
        image = cv2.rectangle(
            image, box.upper_left_corner.as_tuple(), box.lower_right_corner.as_tuple(), color, thickness
        )
    cv2.imshow("table", image)
    while True:
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to break
            break
    cv2.destroyAllWindows()


def count_boxes_in_x(boxes, x):
    return sum(box.x_inside(x) for box in boxes)


def count_boxes_in_y(boxes, y):
    return sum(box.y_inside(y) for box in boxes)


filename = "example.png"
boxes = detect_text(filename)
show_image_with_word_boxes(filename, boxes)


max_x = max(box.max_x() for box in boxes)
x_counts = [count_boxes_in_x(boxes, x) for x in range(max_x)]
plt.plot(x_counts)
plt.show()

max_y = max(box.max_y() for box in boxes)
y_counts = [count_boxes_in_y(boxes, y) for y in range(max_y)]
plt.plot(y_counts)
plt.show()
