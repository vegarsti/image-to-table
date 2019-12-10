import pytest
from flask import Flask
from flask.testing import FlaskClient

from image_to_table.server.app import create_app


@pytest.fixture
def app() -> Flask:
    app = create_app()
    app.testing = True

    with app.app_context():
        yield app


@pytest.fixture
def test_client(app: Flask) -> FlaskClient:
    with app.test_client() as test_client:
        yield test_client
