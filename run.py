from image_to_table.extractor.main import get_table

if __name__ == "__main__":
    filename = "images/example.png"
    with open(filename, "rb") as f:
        image = f.read()
    rows = get_table(image)
    for row in rows:
        print(",".join(row))
