import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Extract X-Request-ID from request headers, or generate a new UUID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
            
        # Store in request state for downstream handlers
        request.state.request_id = request_id
        
        # Process the request
        response = await call_next(request)
        
        # Inject the request ID into the response headers
        response.headers["X-Request-ID"] = request_id
        return response
