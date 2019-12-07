from http import HTTPStatus
from pathlib import Path

import pytest
from flask.testing import FlaskClient


@pytest.fixture
def image() -> bytes:
    image_path = Path(__file__).parent.parent.parent.parent / "images/example.png"
    with open(image_path, "rb") as f:
        yield f.read()


class TestServer:
    def test_ok(self, test_client: FlaskClient, image: bytes) -> None:
        x = type(test_client)
        response = test_client.post("/api", content_type="image/png", data=image)
        expected_response = {"image_size": len(image)}
        assert response.status_code == HTTPStatus.OK
        assert response.get_json() == expected_response
