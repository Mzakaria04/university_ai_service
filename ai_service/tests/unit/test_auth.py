import pytest
import time
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from jose import jwt

from ai_service.config.settings import settings
from ai_service.middleware.auth import JWTAuthMiddleware
from ai_service.middleware.request_id import RequestIdMiddleware

# Set up a test application specifically for isolating middleware behavior
app_for_testing = FastAPI()
app_for_testing.add_middleware(JWTAuthMiddleware)
app_for_testing.add_middleware(RequestIdMiddleware)

@app_for_testing.get("/test-protected")
async def protected_route(request: Request):
    user = request.state.user_context
    return {
        "user_id": user.user_id,
        "university_id": user.university_id,
        "full_name": user.full_name,
        "role": user.role
    }

@app_for_testing.get("/health")
async def public_route():
    return {"status": "ok"}


def create_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Helper to generate JWT tokens for testing."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm="HS256")


@pytest.fixture
def client():
    return TestClient(app_for_testing)


def test_public_route_bypasses_auth(client):
    """Verify that public routes like /health do not require authentication."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    # Request ID middleware should still run and inject request ID
    assert "X-Request-ID" in response.headers


def test_missing_token_returns_401(client):
    """Verify that accessing protected route without token returns 401."""
    response = client.get("/test-protected")
    assert response.status_code == 401
    assert "Missing authorization token" in response.json()["detail"]


def test_invalid_header_format_returns_401(client):
    """Verify that malformed Authorization headers return 401."""
    response = client.get("/test-protected", headers={"Authorization": "NotBearer some-token-string"})
    assert response.status_code == 401
    assert "Missing authorization token" in response.json()["detail"]


def test_expired_token_returns_401(client):
    """Verify that expired tokens are rejected with a 401."""
    user_payload = {
        "id": "student-uuid-123",
        "universityId": "20260001",
        "fullName": "Alice Smith",
        "role": "STUDENT"
    }
    expired_token = create_token({"user": user_payload}, expires_delta=timedelta(minutes=-5))
    response = client.get("/test-protected", headers={"Authorization": f"Bearer {expired_token}"})
    assert response.status_code == 401
    assert "Token has expired" in response.json()["detail"]


def test_valid_flat_payload_jwt_success(client):
    """Verify that valid flat payload decodes correctly and injects UserContext."""
    flat_payload = {
        "id": "student-uuid-123",
        "universityId": "20260001",
        "fullName": "Alice Smith",
        "role": "STUDENT"
    }
    token = create_token(flat_payload)
    response = client.get("/test-protected", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["user_id"] == "student-uuid-123"
    assert res_data["university_id"] == "20260001"
    assert res_data["full_name"] == "Alice Smith"
    assert res_data["role"] == "STUDENT"
    assert "X-Request-ID" in response.headers


def test_valid_nested_payload_jwt_success(client):
    """Verify that valid nested payload ('user' claim) decodes correctly and injects UserContext."""
    nested_user = {
        "id": "instructor-uuid-456",
        "universityId": "20269999",
        "fullName": "Dr. Bob Jones",
        "role": "INSTRUCTOR"
    }
    token = create_token({"user": nested_user})
    response = client.get("/test-protected", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["user_id"] == "instructor-uuid-456"
    assert res_data["university_id"] == "20269999"
    assert res_data["full_name"] == "Dr. Bob Jones"
    assert res_data["role"] == "INSTRUCTOR"


def test_missing_required_claims_returns_401(client):
    """Verify that tokens missing required claims are rejected with a 401."""
    # Missing 'role'
    incomplete_payload = {
        "id": "admin-uuid-789",
        "universityId": "20268888",
        "fullName": "Admin User"
    }
    token = create_token(incomplete_payload)
    response = client.get("/test-protected", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert "Missing required claim" in response.json()["detail"]


def test_invalid_signature_returns_401(client):
    """Verify that tokens signed with a different key are rejected."""
    wrong_token = jwt.encode(
        {"id": "uuid", "universityId": "123", "fullName": "Name", "role": "STUDENT"},
        "wrong-secret-key",
        algorithm="HS256"
    )
    response = client.get("/test-protected", headers={"Authorization": f"Bearer {wrong_token}"})
    assert response.status_code == 401
    assert "Signature verification failed" in response.json()["detail"]
