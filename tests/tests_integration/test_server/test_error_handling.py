from http import HTTPStatus

import pytest


@pytest.fixture
def raise_exc_during_request(app):
    def _register_before_request(e):
        def _raise_exception():
            raise e

        app.before_request(_raise_exception)

    return _register_before_request


class TestErrorHandling:
    def test_500(self, raise_exc_during_request, test_client):
        expected_response = {
            "error": {"code": HTTPStatus.INTERNAL_SERVER_ERROR.value, "message": "Internal server error"}
        }
        raise_exc_during_request(Exception)
        response = test_client.post("/api")
        assert response.get_json() == expected_response

    def test_404(self, raise_exc_during_request, test_client):
        expected_response = {
            "error": {
                "code": HTTPStatus.NOT_FOUND.value,
                "message": "The requested URL was not found on the server. "
                + "If you entered the URL manually please check your spelling and try again.",
            }
        }
        response = test_client.post("/nonexistent")
        assert response.get_json() == expected_response

    def test_bad_request_data(self, test_client):
        expected_response = {
            "error": {"code": HTTPStatus.BAD_REQUEST.value, "message": "Request data needs to be a png image"}
        }
        response = test_client.post("/api")
        assert response.get_json() == expected_response
