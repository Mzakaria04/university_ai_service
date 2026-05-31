import asyncio
import time
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from ai_service.tools.base import ToolResult, ToolDefinition
from ai_service.tools.registry import ToolRegistry
from ai_service.models.user_context import UserContext
from ai_service.errors import ToolAuthorizationError, ToolTimeoutError, ToolExecutionError

logger = logging.getLogger("ai_service.tools")

class ToolExecutor:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def execute(
        self,
        tool_name: str,
        arguments: dict,
        user_context: UserContext
    ) -> ToolResult:
        """
        Executes a registered tool. Enforces role authorization, validates arguments,
        injects the authenticated user's context parameters securely, and handles timeouts/retries.
        """
        start_time = time.monotonic()
        logger.info(f"Preparing to execute tool {tool_name} for user {user_context.user_id}")

        # Step 1: Authorization check
        if not ToolRegistry.is_authorized(tool_name, user_context.role):
            logger.warning(f"Unauthorized tool call: User {user_context.user_id} ({user_context.role}) tried to access {tool_name}")
            raise ToolAuthorizationError(f"Role {user_context.role} is not authorized to use tool {tool_name}")

        # Step 2: Retrieve tool definition
        tool_def = ToolRegistry.get(tool_name)
        if not tool_def:
            raise ToolExecutionError(f"Tool {tool_name} not found in registry")

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
            raise e
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.error(f"Tool {tool_name} execution failed in executor: {e}")
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Tool failed: {str(e)}",
                metadata={"latency_ms": latency_ms}
            )

        latency_ms = (time.monotonic() - start_time) * 1000
        result.metadata["latency_ms"] = latency_ms
        logger.info(f"Tool {tool_name} executed in {latency_ms:.2f}ms. Success: {result.success}")
        return result

    async def _execute_with_retry(
        self,
        tool_def: ToolDefinition,
        args: dict
    ) -> ToolResult:
        from ai_service.errors import AuthorizationError
        last_error = None
        for attempt in range(tool_def.max_retries + 1):
            try:
                # Wrap handler execution in asyncio timeout
                return await asyncio.wait_for(
                    tool_def.handler(**args),
                    timeout=tool_def.timeout_seconds
                )
            except asyncio.TimeoutError:
                last_error = ToolTimeoutError(
                    f"Tool {tool_def.name} execution timed out after {tool_def.timeout_seconds}s"
                )
                logger.warning(f"Attempt {attempt + 1} for tool {tool_def.name} timed out.")
            except AuthorizationError as e:
                # Propagate authorization error immediately without retry
                raise e
            except Exception as e:
                last_error = e
                logger.error(f"Attempt {attempt + 1} for tool {tool_def.name} raised exception: {e}")
                
            # Wait for exponential backoff if not the last attempt
            if attempt < tool_def.max_retries:
                backoff_time = 0.5 * (attempt + 1)
                await asyncio.sleep(backoff_time)
                
        # Return failing ToolResult wrapping the final error
        return ToolResult(
            success=False,
            data=None,
            error_message=str(last_error)
        )
