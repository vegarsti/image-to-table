from http import HTTPStatus

from flask import Flask, Response, make_response, request

from image_to_table.server.app.exceptions import InvalidData


def api() -> Response:
    if request.content_type != "image/png":
        raise InvalidData
    size = request.content_length
    return make_response({"image_size": size}, HTTPStatus.OK)


def register_routes(app: Flask) -> None:
    app.add_url_rule("/api", view_func=api, methods=["POST"])
