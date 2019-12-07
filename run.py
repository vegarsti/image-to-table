from image_to_table import image_with_word_boxes, number_of_columns

if __name__ == "__main__":

    filename = "images/example.png"
    placement = number_of_columns.find_columns(filename)
    print(placement)

    table = image_with_word_boxes.extract_table_from_image(filename, placement)

    for row in table:
        print(",".join(map(lambda x: x.text, row)))
