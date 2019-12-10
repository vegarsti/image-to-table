from http import HTTPStatus
from logging import log

import werkzeug.exceptions
from flask import Flask, Response, make_response

from image_to_table.server.app.exceptions import InvalidData


def register_error_handlers(app: Flask) -> None:
    app.register_error_handler(
        Exception, lambda e: _make_error(f"Internal server error", code=HTTPStatus.INTERNAL_SERVER_ERROR)
    )
    app.register_error_handler(werkzeug.exceptions.HTTPException, lambda e: _make_error(e.description, e.code))
    app.register_error_handler(InvalidData, lambda e: _make_error(str(e), HTTPStatus.BAD_REQUEST))


def _make_error(message: str, code: int) -> Response:
    log(level=1, msg=message)
    error = {"error": {"message": message, "code": code}}
    return make_response(error, code)
