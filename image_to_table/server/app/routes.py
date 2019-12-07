from http import HTTPStatus

from flask import Flask, Response, make_response, request

from image_to_table.extractor.image_with_word_boxes import extract_table_from_image
from image_to_table.server.app.exceptions import InvalidData


def api() -> Response:
    if request.content_type != "image/png":
        raise InvalidData
    rows = extract_table_from_image(content=request.data)
    return make_response(
        {
            "image_size": request.content_length,
            "table": rows,
            "dimensions": {"rows": len(rows), "columns": len(rows[0])},
        },
        HTTPStatus.OK,
    )


def register_routes(app: Flask) -> None:
    app.add_url_rule("/api", view_func=api, methods=["POST"])
