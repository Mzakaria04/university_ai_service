import pytest
import uuid
import json
from unittest.mock import patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ai_service.models.user_context import UserContext, UserRole
from ai_service.tools.base import ToolResult, ToolDefinition, ToolDomain
from ai_service.tools.executor import ToolExecutor
from ai_service.tools.registry import ToolRegistry, ROLE_TOOL_PERMISSIONS

pytestmark = pytest.mark.anyio

@pytest.fixture
def anyio_backend():
    return "asyncio"

class FakeRedis:
    def __init__(self):
        self.store = {}

    def from_url(self, url, **kwargs):
        return self

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        return True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

async def test_redis_cache_hits_and_misses():
    """Verify cache miss on first call, cache hit on second call, and correct Redis key mapping."""
    fake_redis = FakeRedis()
    
    execution_counter = 0

    async def mock_handler(db, user_id, param):
        nonlocal execution_counter
        execution_counter += 1
        return ToolResult(success=True, data={"result": f"hello-{param}-{user_id}"})

    # Reg test tool
    tool_def = ToolDefinition(
        name="get_my_gpa", # Use get_my_gpa as it is listed in CACHEABLE_TOOLS
        description="GPA tool mock",
        domain=ToolDomain.DATABASE,
        allowed_roles={UserRole.STUDENT},
        parameters=[],
        handler=mock_handler
    )
    gpa_was_authorized = "get_my_gpa" in ROLE_TOOL_PERMISSIONS[UserRole.STUDENT]
    ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].add("get_my_gpa")
    
    # Overwrite if exists, restore later
    old_tool = ToolRegistry._tools.get("get_my_gpa")
    ToolRegistry._tools["get_my_gpa"] = tool_def

    try:
        user_ctx = UserContext(
            user_id="alice-123",
            university_id="20261111",
            full_name="Alice",
            role=UserRole.STUDENT
        )
        
        db_mock = AsyncMock(spec=AsyncSession)
        executor = ToolExecutor(db_mock)

        with patch("redis.asyncio.from_url", side_effect=fake_redis.from_url):
            # 1. First execution: Cache Miss
            res1 = await executor.execute("get_my_gpa", {"param": "val"}, user_ctx)
            assert res1.success is True
            assert res1.data == {"result": "hello-val-alice-123"}
            assert res1.metadata.get("cached") is not True
            assert execution_counter == 1

            # Verify cached key exists in FakeRedis
            import hashlib
            serialized_args = json.dumps({"param": "val"}, sort_keys=True)
            args_hash = hashlib.sha256(serialized_args.encode("utf-8")).hexdigest()
            expected_key = f"tool:alice-123:get_my_gpa:{args_hash}"
            assert expected_key in fake_redis.store
            assert json.loads(fake_redis.store[expected_key]) == {"result": "hello-val-alice-123"}

            # 2. Second execution: Cache Hit
            res2 = await executor.execute("get_my_gpa", {"param": "val"}, user_ctx)
            assert res2.success is True
            assert res2.data == {"result": "hello-val-alice-123"}
            assert res2.metadata.get("cached") is True
            # Execution counter should still be 1 (bypassed execution)
            assert execution_counter == 1

    finally:
        if old_tool:
            ToolRegistry._tools["get_my_gpa"] = old_tool
        else:
            del ToolRegistry._tools["get_my_gpa"]
        if not gpa_was_authorized:
            ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].discard("get_my_gpa")


async def test_redis_cache_user_isolation():
    """Verify that different users do not share cache hits (user-isolation)."""
    fake_redis = FakeRedis()
    
    execution_counter = 0

    async def mock_handler(db, user_id, param):
        nonlocal execution_counter
        execution_counter += 1
        return ToolResult(success=True, data={"result": f"hello-{param}-{user_id}"})

    tool_def = ToolDefinition(
        name="get_my_gpa",
        description="GPA tool mock",
        domain=ToolDomain.DATABASE,
        allowed_roles={UserRole.STUDENT},
        parameters=[],
        handler=mock_handler
    )
    gpa_was_authorized = "get_my_gpa" in ROLE_TOOL_PERMISSIONS[UserRole.STUDENT]
    ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].add("get_my_gpa")
    old_tool = ToolRegistry._tools.get("get_my_gpa")
    ToolRegistry._tools["get_my_gpa"] = tool_def

    try:
        alice_ctx = UserContext(
            user_id="alice-123",
            university_id="20261111",
            full_name="Alice",
            role=UserRole.STUDENT
        )
        bob_ctx = UserContext(
            user_id="bob-456",
            university_id="20262222",
            full_name="Bob",
            role=UserRole.STUDENT
        )
        
        db_mock = AsyncMock(spec=AsyncSession)
        executor = ToolExecutor(db_mock)

        with patch("redis.asyncio.from_url", side_effect=fake_redis.from_url):
            # Run Alice first
            res1 = await executor.execute("get_my_gpa", {"param": "val"}, alice_ctx)
            assert res1.data == {"result": "hello-val-alice-123"}
            assert execution_counter == 1

            # Run Bob with same arguments (should be cache miss because of user isolation)
            res2 = await executor.execute("get_my_gpa", {"param": "val"}, bob_ctx)
            assert res2.data == {"result": "hello-val-bob-456"}
            assert res2.metadata.get("cached") is not True
            assert execution_counter == 2

    finally:
        if old_tool:
            ToolRegistry._tools["get_my_gpa"] = old_tool
        else:
            del ToolRegistry._tools["get_my_gpa"]
        if not gpa_was_authorized:
            ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].discard("get_my_gpa")


async def test_redis_resilience_on_failure():
    """Verify that if Redis throws an exception, the tool executes successfully via DB fallback."""
    execution_counter = 0

    async def mock_handler(db, user_id, param):
        nonlocal execution_counter
        execution_counter += 1
        return ToolResult(success=True, data={"result": f"hello-{param}"})

    tool_def = ToolDefinition(
        name="get_my_gpa",
        description="GPA tool mock",
        domain=ToolDomain.DATABASE,
        allowed_roles={UserRole.STUDENT},
        parameters=[],
        handler=mock_handler
    )
    gpa_was_authorized = "get_my_gpa" in ROLE_TOOL_PERMISSIONS[UserRole.STUDENT]
    ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].add("get_my_gpa")
    old_tool = ToolRegistry._tools.get("get_my_gpa")
    ToolRegistry._tools["get_my_gpa"] = tool_def

    try:
        user_ctx = UserContext(
            user_id="alice-123",
            university_id="20261111",
            full_name="Alice",
            role=UserRole.STUDENT
        )
        
        db_mock = AsyncMock(spec=AsyncSession)
        executor = ToolExecutor(db_mock)

        # Mock from_url to raise a ConnectionError when trying to connect/get
        with patch("redis.asyncio.from_url", side_effect=Exception("Redis connection refused")):
            res = await executor.execute("get_my_gpa", {"param": "val"}, user_ctx)
            assert res.success is True
            assert res.data == {"result": "hello-val"}
            assert execution_counter == 1
            assert res.metadata.get("cached") is not True

    finally:
        if old_tool:
            ToolRegistry._tools["get_my_gpa"] = old_tool
        else:
            del ToolRegistry._tools["get_my_gpa"]
        if not gpa_was_authorized:
            ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].discard("get_my_gpa")
