from typing import List

from image_to_table.extractor import image_with_word_boxes, number_of_columns


def get_table(image: bytes) -> List[List[str]]:
    placement = number_of_columns.find_columns(content=image)
    table = image_with_word_boxes.extract_table_from_image(image, placement)
    rows = [[cell.text for cell in row] for row in table]
    return rows
