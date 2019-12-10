from image_to_table.extractor.image_with_word_boxes import extract_table_from_image

if __name__ == "__main__":
    filename = "images/example.png"
    with open(filename, "rb") as f:
        image = f.read()
    rows = extract_table_from_image(content=image)
    for row in rows:
        print(",".join(row))
