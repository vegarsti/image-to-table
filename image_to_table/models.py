from dataclasses import dataclass

from opencv_wrapper import Rect


@dataclass
class TextBox:
    text: str
    rect: Rect
