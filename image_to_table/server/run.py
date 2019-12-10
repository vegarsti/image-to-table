from image_to_table.server.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
