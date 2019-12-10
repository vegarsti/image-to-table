from flask import Flask

from image_to_table.server.app.error_handling import register_error_handlers
from image_to_table.server.app.routes import register_routes


def create_app() -> Flask:
    app = Flask(import_name="image_to_table")
    register_routes(app)
    register_error_handlers(app)
    return app
