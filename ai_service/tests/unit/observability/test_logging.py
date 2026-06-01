import json
import logging
import sys
import pytest
import structlog
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from ai_service.middleware.request_id import RequestIdMiddleware
from ai_service.middleware.auth import JWTAuthMiddleware
from ai_service.observability.logging import setup_logging
from ai_service.tests.unit.test_auth import create_token

# We will create a test app to verify request logging middleware behavior
logging_test_app = FastAPI()
logging_test_app.add_middleware(JWTAuthMiddleware)
logging_test_app.add_middleware(RequestIdMiddleware)

@logging_test_app.get("/test-log")
async def log_route(request: Request):
    logger = structlog.get_logger("test_request_logger")
    logger.info("handling test request")
    return {"status": "ok"}

@pytest.fixture
def logging_client():
    return TestClient(logging_test_app)

def test_context_vars_binding():
    """Verify that contextvars are bound and merged into the logs."""
    # Ensure structlog is configured
    setup_logging("production")
    
    # We will capture sys.stdout during logging
    from io import StringIO
    captured_stdout = StringIO()
    
    # Get standard root logger handler
    root_logger = logging.getLogger()
    # Create custom stream handler
    test_handler = logging.StreamHandler(captured_stdout)
    # Get current formatter
    if root_logger.handlers:
        test_handler.setFormatter(root_logger.handlers[0].formatter)
    root_logger.addHandler(test_handler)
    
    try:
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id="custom-req-id",
            user_id="custom-user-id",
            role="INSTRUCTOR"
        )
        
        logger = structlog.get_logger("test_bindings")
        logger.info("Hello World", extra_key="extra_val")
        
        output = captured_stdout.getvalue().strip()
        assert output != ""
        
        # Parse output as JSON
        log_json = json.loads(output)
        assert log_json["request_id"] == "custom-req-id"
        assert log_json["user_id"] == "custom-user-id"
        assert log_json["role"] == "INSTRUCTOR"
        assert log_json["event"] == "Hello World"
        assert log_json["extra_key"] == "extra_val"
        assert "timestamp" in log_json
        
    finally:
        root_logger.removeHandler(test_handler)
        # Restore development logging to prevent side-effects in other tests
        setup_logging("development")

def test_middleware_logging_integration(logging_client):
    """Verify that JWTAuthMiddleware and RequestIdMiddleware populate structlog contextvars."""
    setup_logging("production")
    
    from io import StringIO
    captured_stdout = StringIO()
    root_logger = logging.getLogger()
    test_handler = logging.StreamHandler(captured_stdout)
    if root_logger.handlers:
        test_handler.setFormatter(root_logger.handlers[0].formatter)
    root_logger.addHandler(test_handler)
    
    try:
        user_payload = {
            "id": "user-uuid-111",
            "universityId": "20261111",
            "fullName": "Charlie Brown",
            "role": "STUDENT"
        }
        token = create_token(user_payload)
        
        response = logging_client.get(
            "/test-log",
            headers={
                "Authorization": f"Bearer {token}",
                "X-Request-ID": "external-req-999"
            }
        )
        
        assert response.status_code == 200
        output = captured_stdout.getvalue().strip()
        assert output != ""
        
        # In a multi-line output, find the line containing our log event
        lines = [line for line in output.split("\n") if "handling test request" in line]
        assert len(lines) > 0
        
        log_json = json.loads(lines[0])
        assert log_json["request_id"] == "external-req-999"
        assert log_json["user_id"] == "user-uuid-111"
        assert log_json["role"] == "STUDENT"
        assert log_json["event"] == "handling test request"
        
    finally:
        root_logger.removeHandler(test_handler)
        setup_logging("development")
