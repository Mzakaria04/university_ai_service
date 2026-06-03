import asyncio
import time
import logging
import json
import hashlib
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

from ai_service.tools.base import ToolResult, ToolDefinition
from ai_service.tools.registry import ToolRegistry
from ai_service.models.user_context import UserContext
from ai_service.errors import ToolAuthorizationError, ToolTimeoutError, ToolExecutionError

logger = logging.getLogger("ai_service.tools")

CACHEABLE_TOOLS = {
    "get_my_gpa": 300,
    "get_my_transcript": 600,
    "get_my_schedule": 60,
    "faculty_bylaw_search": 3600,
}

class ToolExecutor:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def execute(
        self,
        tool_name: str,
        arguments: dict,
        user_context: UserContext,
        session_id: str | None = None,
        message_event_id: str | None = None
    ) -> ToolResult:
        """
        Executes a registered tool. Enforces role authorization, validates arguments,
        injects the authenticated user's context parameters securely, and handles timeouts/retries.
        """
        from ai_service.observability.tracing import get_tracer
        tracer = get_tracer()
        
        with tracer.start_as_current_span("tool_execution") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("user.id", user_context.user_id)
            span.set_attribute("user.role", user_context.role.name if hasattr(user_context.role, "name") else str(user_context.role))
            
            start_time = time.monotonic()
            logger.info(f"Preparing to execute tool {tool_name} for user {user_context.user_id}")

            # Step 1: Authorization check
            if not ToolRegistry.is_authorized(tool_name, user_context.role):
                logger.warning(f"Unauthorized tool call: User {user_context.user_id} ({user_context.role}) tried to access {tool_name}")
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.error", "Unauthorized")
                
                from ai_service.observability.metrics import ai_tool_calls_total, ai_tool_latency_seconds
                ai_tool_calls_total.labels(tool_name=tool_name, success="false").inc()
                ai_tool_latency_seconds.labels(tool_name=tool_name).observe(time.monotonic() - start_time)
                
                # Write log to DB
                await self._save_execution_log(
                    session_id=session_id,
                    message_event_id=message_event_id,
                    tool_name=tool_name,
                    user_context=user_context,
                    arguments=arguments,
                    success=False,
                    error_message=f"Role {user_context.role} is not authorized to use tool {tool_name}",
                    attempts=0,
                    latency_ms=(time.monotonic() - start_time) * 1000,
                    data=None
                )
                
                raise ToolAuthorizationError(f"Role {user_context.role} is not authorized to use tool {tool_name}")

            # Step 2: Retrieve tool definition
            tool_def = ToolRegistry.get(tool_name)
            if not tool_def:
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.error", "Tool not found")
                
                from ai_service.observability.metrics import ai_tool_calls_total, ai_tool_latency_seconds
                ai_tool_calls_total.labels(tool_name=tool_name, success="false").inc()
                ai_tool_latency_seconds.labels(tool_name=tool_name).observe(time.monotonic() - start_time)
                
                # Write log to DB
                await self._save_execution_log(
                    session_id=session_id,
                    message_event_id=message_event_id,
                    tool_name=tool_name,
                    user_context=user_context,
                    arguments=arguments,
                    success=False,
                    error_message=f"Tool {tool_name} not found in registry",
                    attempts=0,
                    latency_ms=(time.monotonic() - start_time) * 1000,
                    data=None
                )
                
                raise ToolExecutionError(f"Tool {tool_name} not found in registry")

            # Step 2.5: Check Redis cache for cacheable tools
            ttl = CACHEABLE_TOOLS.get(tool_name)
            cache_key = None
            if ttl is not None:
                try:
                    import redis.asyncio as aioredis
                    from ai_service.config.settings import settings
                    
                    # Sort keys to ensure deterministic hashing of arguments
                    serialized_args = json.dumps(arguments, sort_keys=True)
                    args_hash = hashlib.sha256(serialized_args.encode("utf-8")).hexdigest()
                    cache_key = f"tool:{user_context.user_id}:{tool_name}:{args_hash}"
                    
                    async with aioredis.from_url(settings.REDIS_URL, decode_responses=True) as redis_client:
                        cached_val = await redis_client.get(cache_key)
                        
                    if cached_val:
                        logger.info(f"Cache hit for tool {tool_name} with key {cache_key}")
                        deserialized = json.loads(cached_val)
                        
                        latency_s = time.monotonic() - start_time
                        latency_ms = latency_s * 1000
                        
                        result = ToolResult(
                            success=True,
                            data=deserialized,
                            metadata={"cached": True, "attempts": 1, "latency_ms": latency_ms}
                        )
                        
                        from ai_service.observability.metrics import ai_tool_calls_total, ai_tool_latency_seconds
                        ai_tool_calls_total.labels(tool_name=tool_name, success="true").inc()
                        ai_tool_latency_seconds.labels(tool_name=tool_name).observe(latency_s)
                        
                        span.set_attribute("tool.success", True)
                        span.set_attribute("tool.cached", True)
                        
                        await self._save_execution_log(
                            session_id=session_id,
                            message_event_id=message_event_id,
                            tool_name=tool_name,
                            user_context=user_context,
                            arguments=arguments,
                            success=True,
                            error_message=None,
                            attempts=1,
                            latency_ms=latency_ms,
                            data=deserialized
                        )
                        return result
                except Exception as e:
                    logger.warning(f"Redis cache lookup failed for tool {tool_name}: {e}. Falling back to DB execution.")

            # Step 3: Clone arguments and validate/inject context
            safe_args = arguments.copy()
            
            # Inject DB session and user_id securely
            safe_args["db"] = self.db
            
            # Check if the tool def requires user_id parameter, or inject it by default
            safe_args["user_id"] = user_context.user_id

            # Step 4: Execute with retry and timeout
            from ai_service.errors import AuthorizationError
            try:
                result = await self._execute_with_retry(tool_def, safe_args)
            except AuthorizationError as e:
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.error", str(e))
                
                from ai_service.observability.metrics import ai_tool_calls_total, ai_tool_latency_seconds
                ai_tool_calls_total.labels(tool_name=tool_name, success="false").inc()
                ai_tool_latency_seconds.labels(tool_name=tool_name).observe(time.monotonic() - start_time)
                
                # Write log to DB
                await self._save_execution_log(
                    session_id=session_id,
                    message_event_id=message_event_id,
                    tool_name=tool_name,
                    user_context=user_context,
                    arguments=arguments,
                    success=False,
                    error_message=str(e),
                    attempts=1,
                    latency_ms=(time.monotonic() - start_time) * 1000,
                    data=None
                )
                
                raise e
            except Exception as e:
                latency_s = time.monotonic() - start_time
                latency_ms = latency_s * 1000
                logger.error(f"Tool {tool_name} execution failed in executor: {e}")
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.error", str(e))
                
                from ai_service.observability.metrics import ai_tool_calls_total, ai_tool_latency_seconds
                ai_tool_calls_total.labels(tool_name=tool_name, success="false").inc()
                ai_tool_latency_seconds.labels(tool_name=tool_name).observe(latency_s)
                
                # Write log to DB
                await self._save_execution_log(
                    session_id=session_id,
                    message_event_id=message_event_id,
                    tool_name=tool_name,
                    user_context=user_context,
                    arguments=arguments,
                    success=False,
                    error_message=str(e),
                    attempts=1,
                    latency_ms=latency_ms,
                    data=None
                )
                
                return ToolResult(
                    success=False,
                    data=None,
                    error_message=f"Tool failed: {str(e)}",
                    metadata={"latency_ms": latency_ms}
                )

            latency_s = time.monotonic() - start_time
            latency_ms = latency_s * 1000
            result.metadata["latency_ms"] = latency_ms
            logger.info(f"Tool {tool_name} executed in {latency_ms:.2f}ms. Success: {result.success}")
            
            span.set_attribute("tool.success", result.success)
            if not result.success:
                span.set_attribute("tool.error", result.error_message or "Execution failed")
                
            from ai_service.observability.metrics import ai_tool_calls_total, ai_tool_latency_seconds
            success_str = "true" if result.success else "false"
            ai_tool_calls_total.labels(tool_name=tool_name, success=success_str).inc()
            ai_tool_latency_seconds.labels(tool_name=tool_name).observe(latency_s)
            
            # Write log to DB
            attempts = result.metadata.get("attempts", 1)
            await self._save_execution_log(
                session_id=session_id,
                message_event_id=message_event_id,
                tool_name=tool_name,
                user_context=user_context,
                arguments=arguments,
                success=result.success,
                error_message=result.error_message,
                attempts=attempts,
                latency_ms=latency_ms,
                data=result.data if result.success else None
            )
            
            # After successful tool execution, if cacheable, write to Redis
            if result.success and cache_key and ttl:
                try:
                    import redis.asyncio as aioredis
                    from ai_service.config.settings import settings
                    async with aioredis.from_url(settings.REDIS_URL, decode_responses=True) as redis_client:
                        await redis_client.set(cache_key, json.dumps(result.data), ex=ttl)
                    logger.info(f"Cached result for tool {tool_name} with key {cache_key} (TTL: {ttl}s)")
                except Exception as e:
                    logger.warning(f"Failed to write to Redis cache for tool {tool_name}: {e}")

            return result

    async def _save_execution_log(
        self,
        session_id: str | None,
        message_event_id: str | None,
        tool_name: str,
        user_context: UserContext,
        arguments: dict,
        success: bool,
        error_message: str | None,
        attempts: int,
        latency_ms: float,
        data: Any
    ):
        if self.db is not None and session_id is not None:
            try:
                from ai_service.db.models import AIToolExecutionLog
                log_entry = AIToolExecutionLog(
                    session_id=session_id,
                    message_event_id=message_event_id,
                    tool_name=tool_name,
                    user_id=user_context.user_id,
                    user_role=user_context.role.name if hasattr(user_context.role, "name") else str(user_context.role),
                    arguments_json=arguments,
                    result_json=data,
                    success=success,
                    error_message=error_message,
                    attempt_number=attempts,
                    latency_ms=int(latency_ms)
                )
                self.db.add(log_entry)
                await self.db.commit()
            except Exception as db_err:
                logger.error(f"Failed to write tool execution log to database: {db_err}")

    async def _execute_with_retry(
        self,
        tool_def: ToolDefinition,
        args: dict
    ) -> ToolResult:
        from ai_service.errors import AuthorizationError, ToolTimeoutError
        from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception

        attempt_count = 0
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(tool_def.max_retries + 1),
                # multiplier=0.5, min=0.5 matches (0.5 * attempt) behavior
                wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
                retry=retry_if_exception(lambda e: not isinstance(e, AuthorizationError)),
                reraise=True
            ):
                with attempt:
                    attempt_count += 1
                    try:
                        res = await asyncio.wait_for(
                            tool_def.handler(**args),
                            timeout=tool_def.timeout_seconds
                        )
                        if res.metadata is None:
                            res.metadata = {}
                        res.metadata["attempts"] = attempt_count
                        return res
                    except asyncio.TimeoutError as e:
                        logger.warning(f"Timeout executing tool {tool_def.name}")
                        raise ToolTimeoutError(
                            f"Tool {tool_def.name} execution timed out after {tool_def.timeout_seconds}s"
                        ) from e
                    except Exception as e:
                        if not isinstance(e, AuthorizationError):
                            logger.error(f"Error executing tool {tool_def.name}: {e}")
                        raise e
        except AuthorizationError as e:
            raise e
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=str(e),
                metadata={"attempts": attempt_count}
            )
