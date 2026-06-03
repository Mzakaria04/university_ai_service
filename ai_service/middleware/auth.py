from fastapi import Request
from fastapi.responses import JSONResponse
from jose import jwt, JWTError
from starlette.middleware.base import BaseHTTPMiddleware

from ai_service.config.settings import settings
from ai_service.models.user_context import UserContext, UserRole
from ai_service.errors import InvalidTokenError, TokenExpiredError

class JWTAuthMiddleware(BaseHTTPMiddleware):
    # Endpoints that bypass authentication
    PUBLIC_ROUTES = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        # Bypass auth for public routes and internal debug routes
        if request.url.path in self.PUBLIC_ROUTES or request.url.path.startswith("/internal/debug"):
            return await call_next(request)

        # Extract authorization token
        token = self._extract_token(request)
        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing authorization token"}
            )

        try:
            # Decode and verify token using HS256
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=["HS256"]
            )
            
            # The user claims might be nested under "user" or flat
            user_data = payload.get("user", payload)
            
            # Validate required fields
            required_fields = ["id", "universityId", "fullName", "role"]
            for field in required_fields:
                if field not in user_data:
                    raise InvalidTokenError(f"Missing required claim: {field}")
            
            # Construct and inject the UserContext
            user_context = UserContext(
                user_id=user_data["id"],
                university_id=user_data["universityId"],
                full_name=user_data["fullName"],
                role=UserRole(user_data["role"])
            )
            request.state.user_context = user_context
            
            # Bind to structlog contextvars
            import structlog
            structlog.contextvars.bind_contextvars(
                user_id=user_context.user_id,
                role=user_context.role.name if hasattr(user_context.role, "name") else str(user_context.role)
            )
            
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Token has expired"}
            )
        except (JWTError, InvalidTokenError, ValueError) as e:
            return JSONResponse(
                status_code=401,
                content={"detail": f"Invalid token: {str(e)}"}
            )

        return await call_next(request)

    def _extract_token(self, request: Request) -> str | None:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return None
