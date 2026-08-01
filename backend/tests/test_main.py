import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app instance from main.py
from main import app

# Initialize the test client
client = TestClient(app)

def test_query_requires_authentication():
    """Ensure that auth.py blocks unauthenticated requests to protected endpoints."""
    # Attempting to call /query without a Bearer JWT header
    response = client.post("/query", json={"prompt": "Tailor my resume"})

    # Should be rejected with HTTP 401 Unauthorized
    assert response.status_code == 401
    assert "detail" in response.json()


def test_invalid_jwt_token_rejected():
    """Ensure malformed or tampered JWT tokens fail authentication."""
    bad_headers = {"Authorization": "Bearer invalid_mock_token_12345"}
    response = client.post("/query", json={"prompt": "Find missing skills"}, headers=bad_headers)

    # Should still reject invalid token signatures
    assert response.status_code == 401