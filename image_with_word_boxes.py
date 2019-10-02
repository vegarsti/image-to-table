import io

import cv2
from google.cloud import vision


def detect_text(image_path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    word_boxes = []
    for text in texts:
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        word_boxes.append(vertices)
    return word_boxes


def show_image_with_word_boxes(filename, boxes):
    image = cv2.imread(filename)
    color = (0, 0, 0)
    thickness = 2
    # skip first, large box
    for upper_left_corner, _, lower_right_corner, _ in boxes[1:]:
        image = cv2.rectangle(image, upper_left_corner, lower_right_corner, color, thickness)
    cv2.imshow("table", image)
    while True:
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to break
            break
    cv2.destroyAllWindows()


filename = "example.png"
all_vertices = detect_text(filename)
show_image_with_word_boxes(filename, all_vertices)
