import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

from ai_service.config.settings import settings
from ai_service.db.session import engine, get_db
from ai_service.middleware.request_id import RequestIdMiddleware
from ai_service.middleware.auth import JWTAuthMiddleware

# Configure standard logging
logging.basicConfig(
    level=logging.INFO if settings.ENVIRONMENT == "production" else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_service")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events manager. Handles resource startup and shutdown.
    """
    logger.info("Initializing AI service...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Verify database connection on startup
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection successfully verified on startup.")
    except Exception as e:
        logger.critical(f"Database connection verification failed: {e}")
        # In a real microservice, we might raise or log. We proceed to allow app to start.

    # Register tools in the registry
    from ai_service.tools.registry import ToolRegistry
    from ai_service.tools.student.gpa import gpa_tool_definition
    from ai_service.tools.rag.faculty_bylaw_search import bylaw_tool_definition
    from ai_service.tools.student.schedule import schedule_tool_definition
    from ai_service.tools.student.transcript import transcript_tool_definition
    from ai_service.tools.student.attendance import attendance_tool_definition
    from ai_service.tools.instructor.course_students import course_students_tool_definition
    from ai_service.tools.instructor.student_progress import student_progress_tool_definition
    from ai_service.tools.instructor.course_attendance import course_attendance_tool_definition
    from ai_service.tools.admin.registration_statistics import registration_statistics_tool_definition
    from ai_service.tools.admin.all_students import all_students_tool_definition
    
    ToolRegistry.register(gpa_tool_definition)
    ToolRegistry.register(bylaw_tool_definition)
    ToolRegistry.register(schedule_tool_definition)
    ToolRegistry.register(transcript_tool_definition)
    ToolRegistry.register(attendance_tool_definition)
    ToolRegistry.register(course_students_tool_definition)
    ToolRegistry.register(student_progress_tool_definition)
    ToolRegistry.register(course_attendance_tool_definition)
    ToolRegistry.register(registration_statistics_tool_definition)
    ToolRegistry.register(all_students_tool_definition)
    logger.info("Successfully registered core, instructor, and admin tools.")
        
    yield
    
    # Shutdown resources
    logger.info("Shutting down AI service resources...")
    await engine.dispose()
    logger.info("AI service shutdown complete.")

app = FastAPI(
    title="University AI Assistant Microservice",
    description="Intelligence layer for the University Management System",
    version="1.0.0",
    lifespan=lifespan,
)

# Register Middlewares (Note: Starlette executes middlewares in reverse order)
app.add_middleware(JWTAuthMiddleware)
app.add_middleware(RequestIdMiddleware)

# Mount API Routers
from ai_service.api.v1.sessions import router as sessions_router
from ai_service.api.v1.chat import router as chat_router
from ai_service.api.v1.feedback import router as feedback_router

app.include_router(sessions_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")

# Exception Handlers
from fastapi.responses import JSONResponse
from ai_service.errors import (
    AIServiceError,
    AuthError,
    SessionOwnershipError,
    ToolAuthorizationError,
    SessionNotFoundError,
    ProviderRateLimitError,
)

@app.exception_handler(AuthError)
async def auth_error_handler(request, exc: AuthError):
    return JSONResponse(status_code=401, content={"detail": str(exc)})

@app.exception_handler(SessionOwnershipError)
async def session_ownership_error_handler(request, exc: SessionOwnershipError):
    return JSONResponse(status_code=403, content={"detail": str(exc)})

@app.exception_handler(ToolAuthorizationError)
async def tool_authorization_error_handler(request, exc: ToolAuthorizationError):
    return JSONResponse(status_code=403, content={"detail": str(exc)})

@app.exception_handler(SessionNotFoundError)
async def session_not_found_error_handler(request, exc: SessionNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})

@app.exception_handler(ProviderRateLimitError)
async def provider_rate_limit_error_handler(request, exc: ProviderRateLimitError):
    return JSONResponse(status_code=429, content={"detail": str(exc)})

@app.exception_handler(AIServiceError)
async def ai_service_error_handler(request, exc: AIServiceError):
    return JSONResponse(status_code=500, content={"detail": f"AI Service error: {str(exc)}"})

@app.get("/health", tags=["monitoring"])
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint.
    Verifies that the microservice is running and has active database connectivity.
    """
    try:
        # Run a simple query to verify database health
        await db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "environment": settings.ENVIRONMENT,
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: Database connection error"
        )
