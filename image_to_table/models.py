from dataclasses import dataclass

from matplotlib import pyplot as plt
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

    def fill(self, color=None):
        x, y = self.as_polygon().exterior.xy
        if color is not None:
            plt.fill(x, y, color=color)
        else:
            plt.fill(x, y)

    @staticmethod
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
