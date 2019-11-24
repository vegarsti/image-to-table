from flask import Flask

from image_to_table.server.app.routes import register_routes


def create_app() -> Flask:
    app = Flask(import_name=__name__)
    register_routes(app)
    return app
