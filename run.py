from image_to_table import image_with_word_boxes

if __name__ == "__main__":
    filename = "example.png"
    table = image_with_word_boxes.extract_table_from_image(filename)

    for row in table:
        print(",".join(row))
