from http import HTTPStatus

from flask import Flask, make_response, request


def api():
    x = request.json
    y = request.files
    z = request.data
    return make_response({}, HTTPStatus.OK)


def register_routes(app: Flask) -> None:
    app.add_url_rule("/", view_func=api, methods=["POST"])
