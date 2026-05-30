# University AI Assistant — Production Architecture Design

> **Version:** 1.0 | **Status:** Architecture Blueprint | **Audience:** Engineering Teams

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Service Boundaries & Integration Model](#2-service-boundaries--integration-model)
3. [Authentication & Authorization Flow](#3-authentication--authorization-flow)
4. [Request & Conversation Lifecycle](#4-request--conversation-lifecycle)
5. [Tool Architecture](#5-tool-architecture)
6. [Conversational Memory Architecture](#6-conversational-memory-architecture)
7. [LLM Provider Abstraction](#7-llm-provider-abstraction)
8. [RAG Integration Strategy](#8-rag-integration-strategy)
9. [Database Interaction Design](#9-database-interaction-design)
10. [Schema Redesign Recommendations](#10-schema-redesign-recommendations)
11. [Observability & Debugging Architecture](#11-observability--debugging-architecture)
12. [Error Handling & Reliability](#12-error-handling--reliability)
13. [Security Model](#13-security-model)
14. [Suggested Folder Structure](#14-suggested-folder-structure)
15. [Production Deployment Considerations](#15-production-deployment-considerations)
16. [Scalability Recommendations](#16-scalability-recommendations)
17. [Developer Experience Guide](#17-developer-experience-guide)

---

## 1. System Overview

### 1.1 What This Service Is

The AI service is an **independent FastAPI microservice** that acts as the intelligence layer of the university management platform. It does not own authentication, business logic, or the primary database schema. It is a composable, tool-driven conversational assistant that integrates into an existing ecosystem.

It is not a graph orchestration engine, a DAG planner, or a state machine executor. It is a **prompt-orchestrated, tool-enabled, context-aware chat service** built on standard LLM tool-calling primitives.

### 1.2 Core Responsibilities

- Accept authenticated chat requests from the frontend
- Load and restore conversational session state
- Expose all role-authorized tools to the LLM and let it decide which to call
- Execute tool calls securely against controlled data sources
- Integrate with the existing RAG pipeline for faculty bylaw queries via a single controlled tool
- Stream responses back to the client
- Persist all conversation turns, tool executions, and metadata for observability and audit

### 1.3 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Client                          │
│              (WebSocket / HTTP + JWT Bearer Token)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      AI FastAPI Service                          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  JWT Auth    │  │  Session     │  │  Conversation         │  │
│  │  Middleware  │  │  Manager     │  │  Orchestrator         │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬───────────┘  │
│         │                 │                       │              │
│  ┌──────▼─────────────────▼───────────────────────▼───────────┐  │
│  │                  Execution Pipeline                         │  │
│  │  Memory Load → LLM + All Authorized Tools → Tool Executor  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Tool Layer  │  │  RAG Client  │  │  Memory Manager       │  │
│  │  (Authorized)│  │  (Existing)  │  │  (Session + Summary)  │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬───────────┘  │
│         │                 │                       │              │
└─────────┼─────────────────┼───────────────────────┼─────────────┘
          │                 │                       │
┌─────────▼──────┐  ┌───────▼──────────┐  ┌────────▼─────────────┐
│  PostgreSQL DB │  │  RAG Pipeline    │  │  AI Persistence      │
│  (Shared, RO)  │  │  (Python Import) │  │  (AI-owned tables)   │
└────────────────┘  └──────────────────┘  └──────────────────────┘
```

---

## 2. Service Boundaries & Integration Model

### 2.1 What the AI Service Owns

| Concern | AI Service Owns? | Notes |
|---|---|---|
| JWT verification | ✅ Yes | Validates tokens from existing backend |
| Conversational sessions | ✅ Yes | ai_session, ai_message, ai_tool_log tables |
| Tool registry & authorization | ✅ Yes | Tool definitions, permission maps, execution |
| LLM orchestration | ✅ Yes | Provider abstraction, prompt construction |
| Memory management | ✅ Yes | Windowing, summarization, persistence |
| RAG query dispatching | ✅ Yes | Calls existing RAG pipeline as a client |
| User entity schema | ❌ No | Reads only, through controlled tools |
| Database migrations (core) | ❌ No | Prisma-owned; AI service runs its own Alembic migrations |
| Authentication issuance | ❌ No | Handled by the existing backend |
| Business logic (grades, enrollment) | ❌ No | AI reads data, does not mutate it |

### 2.2 Inter-Service Communication

The AI service interacts with two external systems:

**Existing Backend:**
- Trust boundary: JWT signature verification using the shared public key (or secret, depending on algorithm)
- The AI service never calls backend endpoints to authorize requests — it verifies the JWT itself

**Existing RAG Pipeline:**
- Integrated as a **direct Python import** — `from app.rag import ask`. The RAG project (which exposes `app/` and `data/`) is co-located or installed as a package in the same Python environment as the AI service.
- The AI service wraps the imported function as a single tool: `faculty_bylaw_search`
- RAG is never called directly by the LLM — only through the tool executor
- The pipeline's internal implementation (embeddings, retrieval, generation) is opaque to the AI service; only the function signature matters: `async ask(question: str) -> str`

**PostgreSQL (Shared Database):**
- The AI service uses a read-only database role for all student/instructor/admin data queries
- The AI service has a dedicated schema (`ai_schema`) for its own persistence tables
- Prisma manages the primary schema; the AI service uses SQLAlchemy (async) + Alembic for its own tables only

---

## 3. Authentication & Authorization Flow

### 3.1 JWT Verification Middleware

Every request to the AI service must carry a valid JWT issued by the existing backend. The AI service does not issue, refresh, or invalidate tokens.

```python
# ai_service/middleware/auth.py

from fastapi import Request, HTTPException
from jose import jwt, JWTError
from ai_service.config import settings

class JWTAuthMiddleware:
    async def __call__(self, request: Request, call_next):
        token = self._extract_token(request)
        if not token:
            raise HTTPException(status_code=401, detail="Missing authorization token")
        try:
            # Algorithm: HS256. Secret loaded from JWT_SECRET env var.
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=["HS256"],
            )
        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

        # The existing backend embeds the user object as a top-level "user" claim.
        # The AI service only consumes the four fields it needs.
        user_data = payload.get("user", payload)  # support both nested and flat layouts
        request.state.user_context = UserContext(
            user_id=user_data["id"],
            university_id=user_data["universityId"],
            full_name=user_data["fullName"],
            role=UserRole(user_data["role"]),
        )
        return await call_next(request)

    def _extract_token(self, request: Request) -> str | None:
        # Frontend sends: Authorization: Bearer <access_token>
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return None
```

### 3.2 UserContext Object

The `UserContext` is the single source of truth for identity and permissions throughout a request. It is immutable after construction and propagated explicitly — never through global state or thread locals.

```python
# ai_service/models/user_context.py

from dataclasses import dataclass
from enum import Enum

class UserRole(str, Enum):
    STUDENT    = "STUDENT"
    INSTRUCTOR = "INSTRUCTOR"
    ADMIN      = "ADMIN"

@dataclass(frozen=True)
class UserContext:
    user_id:       str        # maps to user.id (UUID)
    university_id: str        # maps to user.universityId (e.g. "20260001")
    full_name:     str        # maps to user.fullName
    role:          UserRole

    def can_use_tool(self, tool_name: str) -> bool:
        from ai_service.tools.registry import ToolRegistry
        return ToolRegistry.is_authorized(tool_name, self.role)
```

### 3.3 Role-to-Tool Permission Map

Permissions are enforced at the tool registry level, not at the LLM prompt level. The LLM is never shown tools it cannot use.

```python
ROLE_TOOL_PERMISSIONS: dict[UserRole, set[str]] = {
    UserRole.STUDENT: {
        "get_my_gpa", "get_my_schedule", "get_my_transcript",
        "get_my_attendance", "faculty_bylaw_search",
    },
    UserRole.INSTRUCTOR: {
        "get_course_students", "get_student_progress",
        "get_course_attendance", "get_my_schedule",
        "faculty_bylaw_search",
    },
    UserRole.ADMIN: {
        "get_registration_statistics", "get_all_students",
        "get_course_students", "get_student_progress",
        "get_course_attendance", "faculty_bylaw_search",
    },
}
```

---

## 4. Request & Conversation Lifecycle

### 4.1 Full Lifecycle Walkthrough

There is no intent classification step. The LLM receives all tools it is authorized to use and decides on its own whether to call one, call several in sequence, or respond directly without any tool. This is standard LLM tool-calling behavior — the model's training already makes it good at this decision.

```
1. HTTP POST /api/v1/chat/{session_id}
   │
2. JWTAuthMiddleware → extract UserContext
   │
3. SessionManager.load_or_create(session_id, user_context)
   │  → Fetch ai_session record
   │  → Validate session ownership (user_id must match)
   │
4. MemoryManager.load(session_id)
   │  → Load last N messages (short-term window)
   │  → Load compressed summary if exists (long-term)
   │  → Compose memory context block
   │
5. ToolRegistry.get_authorized_tools(user_context.role)
   │  → Returns full list of tools this role can use
   │  → Serialized as OpenAI-compatible tool schemas for the LLM
   │
6. ConversationOrchestrator.run(
   │    user_message, memory_context, authorized_tools, user_context
   │  )
   │  → Build system prompt (role-aware instructions + memory block)
   │  → Call LLMProvider with messages + all authorized tools
   │  → LLM decides: call a tool, call multiple tools, or respond directly
   │  → If tool_calls in response:
   │       → ToolExecutor.execute(tool_call, user_context)
   │       → Append tool_result message
   │       → Call LLMProvider again with updated messages
   │       → Repeat until LLM produces a final text response (max 5 iterations)
   │  → Stream final text response
   │
7. MessagePersistence.save(session_id, turn)
   │  → Persist user message, assistant message
   │  → Persist tool_call and tool_result events
   │  → Persist execution metadata (tokens, latency, provider, tools used)
   │
8. MemoryManager.maybe_compress(session_id)  [background task]
   │  → If message count > threshold → trigger summarization
   │
9. Stream response chunks to client
```

### 4.2 Why No Routing — And What Replaces It

The Intent Router was a pre-classification step that used a second LLM call to decide which tool *category* to expose. This introduced:

- **Latency** — an extra round trip before the real call
- **Fragility** — misclassification silently hid valid tools from the LLM
- **Redundancy** — the main LLM call would make the same decision anyway

The replacement is simpler and more robust: **tool description quality**. The LLM selects tools based on their `description` field. Well-written descriptions are the entire "routing" mechanism.

```python
# The description IS the routing logic.
# Be explicit about when to use each tool.

ToolDefinition(
    name="faculty_bylaw_search",
    description=(
        "Answer questions about the faculty bylaw and official university academic regulations. "
        "Use this when the user asks about rules, policies, attendance requirements, "
        "GPA thresholds, warning systems, academic probation, registration deadlines, "
        "faculty conduct rules, or anything governed by the faculty bylaw document. "
        "Do NOT use this for live personal data like the user's own GPA or schedule — "
        "use the appropriate data tools for those."
    ),
    ...
)

ToolDefinition(
    name="get_my_gpa",
    description=(
        "Retrieve the authenticated student's current cumulative GPA, total earned credits, "
        "and number of completed courses from the live database. "
        "Use this when the student asks about their own GPA, academic standing, or credit progress. "
        "Do NOT use this for GPA rules or policy questions — use faculty_bylaw_search instead."
    ),
    ...
)
```

This pattern — clear, mutually exclusive descriptions with explicit negative guidance — is the idiomatic and reliable way to guide tool selection in production LLM applications.

### 4.3 Agentic Loop Design

The orchestrator supports a bounded agentic loop. The LLM may call tools iteratively, and this is desirable: a user asking "show me my schedule and flag any attendance issues" may result in two sequential tool calls before the final answer. The loop is strictly capped (default: 5 iterations) to prevent runaway execution.

```python
# ai_service/orchestration/conversation_orchestrator.py

MAX_TOOL_ITERATIONS = 5

async def run_turn(
    self,
    messages: list[Message],
    authorized_tools: list[ToolDefinition],
    user_context: UserContext,
) -> AsyncIterator[str]:
    iteration = 0
    while iteration < MAX_TOOL_ITERATIONS:
        response = await self.provider.chat(
            messages=messages,
            tools=authorized_tools,  # Full authorized set — LLM picks what it needs
        )

        if not response.tool_calls:
            # LLM decided no tool is needed, or has finished using tools — stream the answer
            async for chunk in response.stream():
                yield chunk
            break

        # Execute all tool calls the LLM requested in this iteration
        for tool_call in response.tool_calls:
            result = await self.tool_executor.execute(tool_call, user_context)
            messages.append(Message(role="tool", content=result, tool_call_id=tool_call.id))

        messages.append(response.as_assistant_message())
        iteration += 1
    else:
        yield "I was unable to complete this request within the allowed number of steps."
```

### 4.4 System Prompt Design (Role-Aware)

The system prompt is the primary mechanism for behavioral control. It tells the LLM who the user is, what it is allowed to do, and how to behave when tools return data or fail. There is no routing prompt — just a clean, role-specific instruction set.

```python
# ai_service/orchestration/prompt_builder.py

SYSTEM_PROMPT_TEMPLATE = """
You are an AI assistant for {institution_name} university management system.

You are speaking with a {role} named {user_display_name}.

Your capabilities:
{tool_capability_summary}

Guidelines:
- Always answer in the user's language.
- Use tools to retrieve live data whenever the question involves personal or institutional records.
- Use regulation/policy tools for questions about rules, policies, and handbook content.
- For general questions or greetings, respond conversationally without tools.
- Never reveal raw database values beyond what was asked.
- If a tool fails, tell the user the data is temporarily unavailable and suggest they try again.
- Never fabricate data. If you don't have the information, say so.

{memory_context_block}
"""

def build_system_prompt(
    user_context: UserContext,
    authorized_tools: list[ToolDefinition],
    memory_context: str,
) -> str:
    tool_summary = "\n".join(
        f"- {t.name}: {t.description.split('.')[0]}"  # First sentence only
        for t in authorized_tools
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        institution_name="Your University",
        role=user_context.role.value.title(),
        user_display_name=user_context.full_name,
        tool_capability_summary=tool_summary,
        memory_context_block=f"Conversation history summary:\n{memory_context}" if memory_context else "",
    )
```

---

## 5. Tool Architecture

### 6.1 Tool Definition Structure

Every tool is a self-describing unit. Its definition is separate from its implementation. This separation allows the registry to inspect, filter, and serialize tools without executing them.

```python
# ai_service/tools/base.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

class ToolDomain(str, Enum):
    DATABASE = "database"
    RAG = "rag"
    UTILITY = "utility"

@dataclass
class ToolParameter:
    name: str
    type: str  # "string", "integer", "boolean"
    description: str
    required: bool = True
    enum_values: list[str] | None = None

@dataclass
class ToolDefinition:
    name: str
    description: str
    domain: ToolDomain
    allowed_roles: set[UserRole]
    parameters: list[ToolParameter]
    handler: Callable[..., Coroutine[Any, Any, ToolResult]]
    timeout_seconds: float = 10.0
    max_retries: int = 2
    tags: list[str] = field(default_factory=list)

    def to_llm_schema(self) -> dict:
        """Serialize to OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        p.name: {
                            "type": p.type,
                            "description": p.description,
                            **({"enum": p.enum_values} if p.enum_values else {}),
                        }
                        for p in self.parameters
                    },
                    "required": [p.name for p in self.parameters if p.required],
                },
            },
        }

@dataclass
class ToolResult:
    success: bool
    data: Any
    error_message: str | None = None
    metadata: dict = field(default_factory=dict)
```

### 6.2 Tool Registry

```python
# ai_service/tools/registry.py

class ToolRegistry:
    _tools: dict[str, ToolDefinition] = {}

    @classmethod
    def register(cls, tool: ToolDefinition) -> None:
        cls._tools[tool.name] = tool

    @classmethod
    def get_authorized_tools(cls, role: UserRole) -> list[ToolDefinition]:
        allowed_names = ROLE_TOOL_PERMISSIONS.get(role, set())
        return [t for name, t in cls._tools.items() if name in allowed_names]

    @classmethod
    def is_authorized(cls, tool_name: str, role: UserRole) -> bool:
        return tool_name in ROLE_TOOL_PERMISSIONS.get(role, set())

    @classmethod
    def get(cls, tool_name: str) -> ToolDefinition | None:
        return cls._tools.get(tool_name)
```

### 6.3 Tool Executor

The executor is the single point of control for all tool invocations. It enforces authorization, validates arguments, handles retries, logs execution, and captures tracing metadata.

```python
# ai_service/tools/executor.py

class ToolExecutor:
    def __init__(self, db: AsyncSession, tracer: Tracer, logger: ToolLogger):
        self.db = db
        self.tracer = tracer
        self.logger = logger

    async def execute(
        self,
        tool_call: ToolCall,
        user_context: UserContext,
    ) -> ToolResult:
        start_time = time.monotonic()
        span = self.tracer.start_span(f"tool.{tool_call.name}")

        # Step 1: Authorization check
        if not ToolRegistry.is_authorized(tool_call.name, user_context.role):
            await self._log_unauthorized(tool_call, user_context)
            raise ToolAuthorizationError(f"Role {user_context.role} cannot use {tool_call.name}")

        # Step 2: Argument validation
        tool_def = ToolRegistry.get(tool_call.name)
        validated_args = self._validate_args(tool_call.arguments, tool_def)

        # Step 3: Context injection (user_id is never passed by LLM)
        safe_args = self._inject_user_context(validated_args, tool_def, user_context)

        # Step 4: Execution with retry
        result = await self._execute_with_retry(tool_def, safe_args)

        # Step 5: Logging & tracing
        latency_ms = (time.monotonic() - start_time) * 1000
        await self._log_execution(tool_call, result, user_context, latency_ms)
        span.finish(success=result.success, latency_ms=latency_ms)

        return result

    def _inject_user_context(
        self,
        args: dict,
        tool_def: ToolDefinition,
        user_context: UserContext,
    ) -> dict:
        """
        For user-scoped tools, the user_id is ALWAYS injected from the
        verified JWT context, never from LLM-generated arguments.
        This prevents prompt injection attacks that might try to access
        another user's data.
        """
        if "user_id" in [p.name for p in tool_def.parameters]:
            args["user_id"] = user_context.user_id
        return args

    async def _execute_with_retry(
        self,
        tool_def: ToolDefinition,
        args: dict,
    ) -> ToolResult:
        last_error = None
        for attempt in range(tool_def.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    tool_def.handler(**args),
                    timeout=tool_def.timeout_seconds,
                )
            except asyncio.TimeoutError:
                last_error = ToolTimeoutError(f"{tool_def.name} timed out after {tool_def.timeout_seconds}s")
            except Exception as e:
                last_error = e
                if attempt < tool_def.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
        return ToolResult(success=False, data=None, error_message=str(last_error))
```

### 6.4 Tool Implementations by Domain

#### Student Tools

```python
# ai_service/tools/student/gpa.py

@dataclass
class GetMyGpaTool:
    db: AsyncSession

    async def __call__(self, user_id: str) -> ToolResult:
        query = """
            SELECT
                SUM(gr.grade_points * c.credit_hours) / NULLIF(SUM(c.credit_hours), 0) AS gpa,
                SUM(c.credit_hours) AS total_credits,
                COUNT(gr.id) AS courses_completed
            FROM grade_records gr
            JOIN course_offerings co ON gr.offering_id = co.id
            JOIN courses c ON co.course_id = c.id
            JOIN enrollments e ON gr.enrollment_id = e.id
            WHERE e.student_id = :user_id
              AND gr.is_final = true
        """
        row = await self.db.execute(text(query), {"user_id": user_id})
        result = row.mappings().first()
        if not result:
            return ToolResult(success=True, data={"gpa": None, "message": "No completed courses found"})
        return ToolResult(
            success=True,
            data={
                "cumulative_gpa": round(float(result["gpa"] or 0), 2),
                "total_credits": result["total_credits"],
                "courses_completed": result["courses_completed"],
            }
        )

# Registration with metadata
ToolRegistry.register(ToolDefinition(
    name="get_my_gpa",
    description="Retrieve the authenticated student's cumulative GPA, total credits, and number of completed courses.",
    domain=ToolDomain.DATABASE,
    allowed_roles={UserRole.STUDENT},
    parameters=[],  # user_id injected from context
    handler=GetMyGpaTool(db=...).execute,
    timeout_seconds=5.0,
    tags=["student", "academic", "gpa"],
))
```

#### RAG Tool

There is a single RAG tool: `faculty_bylaw_search`. It wraps the existing RAG pipeline, which is already a function that accepts a question string and returns a string answer directly. The AI service does not manage retrieval, chunk scoring, or re-ranking — that is entirely handled inside the existing pipeline. The tool's job is simply to call it and wrap the result in a `ToolResult`.

```python
# ai_service/tools/rag/faculty_bylaw_search.py

from ai_service.tools.base import ToolDefinition, ToolDomain, ToolParameter, ToolResult
from ai_service.tools.registry import ToolRegistry, ROLE_TOOL_PERMISSIONS
from ai_service.models.user_context import UserRole

class FacultyBylawSearchTool:
    """
    Wraps the existing RAG pipeline function.
    The pipeline accepts a natural-language question and returns
    a complete string answer — no chunk management needed here.
    """

    def __init__(self, rag_pipeline_fn):
        # rag_pipeline_fn is the existing callable:
        # async def rag_pipeline(question: str) -> str
        self.rag_pipeline_fn = rag_pipeline_fn

    async def __call__(self, question: str) -> ToolResult:
        try:
            answer: str = await self.rag_pipeline_fn(question)

            if not answer or not answer.strip():
                return ToolResult(
                    success=True,
                    data="I could not find relevant information about that in the faculty bylaw.",
                )

            return ToolResult(success=True, data=answer)

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error_message=str(e),
            )


ToolRegistry.register(ToolDefinition(
    name="faculty_bylaw_search",
    description=(
        "Answer questions about the faculty bylaw and official university academic regulations. "
        "Use this when the user asks about rules, policies, attendance requirements, "
        "GPA thresholds, warning systems, academic probation, registration deadlines, "
        "faculty conduct rules, or anything governed by the faculty bylaw document. "
        "Do NOT use this for live personal data like the user's own GPA or schedule."
    ),
    domain=ToolDomain.RAG,
    allowed_roles=set(UserRole),   # STUDENT, INSTRUCTOR, and ADMIN can all access bylaw information
    parameters=[
        ToolParameter(
            name="question",
            type="string",
            description="The specific bylaw or policy question to look up.",
            required=True,
        )
    ],
    handler=FacultyBylawSearchTool(rag_pipeline_fn=...).execute,
    timeout_seconds=15.0,  # RAG pipelines can be slower than DB queries
    max_retries=1,
    tags=["rag", "bylaw", "policy"],
))
```

**Key design notes:**

- The `question` parameter is the only argument. The LLM formulates it naturally from the user's message.
- `timeout_seconds` is set to 15 seconds, higher than database tools, because RAG pipelines involve embedding + retrieval + generation internally.
- `max_retries` is 1 (one retry on failure), not 2, since RAG failures are usually caused by the pipeline being unavailable rather than transient errors that benefit from immediate retry.
- The tool returns `ToolResult.data` as a plain string. The LLM receives this string as a `tool` role message and uses it to compose its final answer. There is no chunk list, no score parsing, and no citation extraction needed on the AI service side.

---

## 6. Conversational Memory Architecture

### 7.1 Memory Layers

The memory system operates in two layers. They are independent in storage but composed together when building the prompt context block.

```
┌─────────────────────────────────────────────────────────┐
│                    Prompt Context Block                  │
│                                                          │
│  [System Prompt: role-aware instructions]                │
│  [Long-Term Summary: compressed history, if any]         │
│  [Short-Term Window: last N messages, verbatim]          │
│  [Current User Message]                                  │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Short-Term Memory (Message Window)

The short-term window holds the last N conversation turns in full fidelity. This supports follow-up understanding within an active session.

```python
# ai_service/memory/short_term.py

SHORT_TERM_WINDOW_SIZE = 12  # Last 12 messages (6 turns)
SHORT_TERM_TOKEN_BUDGET = 3000  # Soft cap in tokens

class ShortTermMemory:
    async def load(self, session_id: str, db: AsyncSession) -> list[Message]:
        rows = await db.execute(
            select(AIMessageEvent)
            .where(
                AIMessageEvent.session_id == session_id,
                AIMessageEvent.role.in_(["user", "assistant"]),
            )
            .order_by(AIMessageEvent.created_at.desc())
            .limit(SHORT_TERM_WINDOW_SIZE)
        )
        messages = list(reversed(rows.scalars().all()))
        return self._trim_to_token_budget(messages, SHORT_TERM_TOKEN_BUDGET)

    def _trim_to_token_budget(self, messages: list, budget: int) -> list:
        total = 0
        trimmed = []
        for msg in reversed(messages):
            tokens = estimate_tokens(msg.content)
            if total + tokens > budget:
                break
            trimmed.insert(0, msg)
            total += tokens
        return trimmed
```

### 7.3 Long-Term Memory (Summary Persistence)

When a session grows beyond the compression threshold, the memory manager triggers an asynchronous summarization job. The summary is persisted as a special message event and loaded at the beginning of subsequent sessions.

```python
# ai_service/memory/long_term.py

COMPRESSION_TRIGGER_COUNT = 20  # messages
COMPRESSION_TRIGGER_TOKENS = 6000  # tokens

class LongTermMemory:
    async def maybe_compress(self, session_id: str, db: AsyncSession) -> None:
        count = await self._count_unsummarized_messages(session_id, db)
        if count < COMPRESSION_TRIGGER_COUNT:
            return

        messages = await self._load_unsummarized_messages(session_id, db)
        summary_text = await self._generate_summary(messages)

        async with db.begin():
            # Persist the summary as a special message event
            summary_event = AIMessageEvent(
                session_id=session_id,
                role="system",
                message_type=MessageType.SUMMARY,
                content=summary_text,
            )
            db.add(summary_event)

            # Mark summarized messages
            await db.execute(
                update(AIMessageEvent)
                .where(AIMessageEvent.id.in_([m.id for m in messages]))
                .values(is_summarized=True)
            )

    async def load_summary(self, session_id: str, db: AsyncSession) -> str | None:
        row = await db.execute(
            select(AIMessageEvent)
            .where(
                AIMessageEvent.session_id == session_id,
                AIMessageEvent.message_type == MessageType.SUMMARY,
            )
            .order_by(AIMessageEvent.created_at.desc())
            .limit(1)
        )
        event = row.scalars().first()
        return event.content if event else None

    async def _generate_summary(self, messages: list[AIMessageEvent]) -> str:
        conversation_text = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in messages
        )
        prompt = f"""
Summarize the following university assistant conversation.
Focus on: what data was retrieved, what policies were discussed, 
any unresolved questions, and the student/instructor's apparent goals.
Be concise. Max 300 words.

Conversation:
{conversation_text}
"""
        return await self.provider.complete(prompt, max_tokens=400)
```

### 7.4 Memory Composition

```python
# ai_service/memory/composer.py

class MemoryComposer:
    async def compose(self, session_id: str, db: AsyncSession) -> MemoryContext:
        summary = await self.long_term.load_summary(session_id, db)
        recent_messages = await self.short_term.load(session_id, db)

        context_parts = []
        if summary:
            context_parts.append(f"[Conversation History Summary]\n{summary}")

        return MemoryContext(
            summary=summary,
            recent_messages=recent_messages,
            context_block="\n\n".join(context_parts),
        )
```

---

## 7. LLM Provider Abstraction

### 8.1 Provider Interface

All providers implement a single interface. New providers require only a new adapter class — no changes to the orchestrator.

```python
# ai_service/providers/base.py

from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = True,
    ) -> LLMResponse:
        ...

    @abstractmethod
    async def complete(self, prompt: str, max_tokens: int = 512) -> str:
        ...

    @property
    @abstractmethod
    def supports_tool_calling(self) -> bool:
        ...

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        ...
```

### 8.2 OpenRouter Adapter (Primary)

```python
# ai_service/providers/openrouter.py

class OpenRouterProvider(LLMProvider):
    MODEL = "thudm/glm-4.5-air"
    BASE_URL = "https://openrouter.ai/api/v1"

    async def chat(self, messages, tools=None, stream=True) -> LLMResponse:
        payload = {
            "model": self.MODEL,
            "messages": [m.to_openai_dict() for m in messages],
            "stream": stream,
        }
        if tools:
            payload["tools"] = [t.to_llm_schema() for t in tools]
            payload["tool_choice"] = "auto"

        async with self._client.post("/chat/completions", json=payload) as resp:
            if stream:
                return StreamingLLMResponse(resp)
            data = await resp.json()
            return self._parse_response(data)

    @property
    def supports_tool_calling(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True
```

### 8.3 Groq Fallback Adapter

```python
# ai_service/providers/groq.py

class GroqProvider(LLMProvider):
    MODEL = "llama-3.3-70b-versatile"
    BASE_URL = "https://api.groq.com/openai/v1"

    # Implementation mirrors OpenRouter; Groq is OpenAI-compatible
    # All methods follow the same interface
```

### 8.4 Failover Orchestrator

```python
# ai_service/providers/failover.py

class FailoverProviderOrchestrator(LLMProvider):
    """
    Wraps a primary provider with automatic failover to a secondary.
    Handles rate limits, timeouts, and API failures transparently.
    """

    def __init__(self, primary: LLMProvider, fallback: LLMProvider):
        self.primary = primary
        self.fallback = fallback
        self._primary_healthy = True
        self._circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

    async def chat(self, messages, tools=None, stream=True) -> LLMResponse:
        provider = self._select_provider()
        try:
            result = await asyncio.wait_for(
                provider.chat(messages, tools, stream),
                timeout=settings.LLM_TIMEOUT_SECONDS,
            )
            self._circuit_breaker.record_success()
            return result
        except (RateLimitError, ProviderUnavailableError, asyncio.TimeoutError) as e:
            self._circuit_breaker.record_failure()
            await self._log_failover_event(provider, e)
            if provider is self.primary:
                return await self.fallback.chat(messages, tools, stream)
            raise ProviderExhaustedError("All providers failed") from e

    def _select_provider(self) -> LLMProvider:
        if self._circuit_breaker.is_open():
            return self.fallback
        return self.primary
```

### 8.5 Provider Configuration

```python
# ai_service/config/providers.py

PROVIDER_CONFIG = {
    "openrouter": {
        "model": "thudm/glm-4.5-air",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "timeout": 30,
        "max_retries": 2,
        "supports_tools": True,
        "supports_streaming": True,
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "timeout": 20,
        "max_retries": 2,
        "supports_tools": True,
        "supports_streaming": True,
    },
    # Future providers — adding these requires only a new adapter file
    # "openai": { "model": "gpt-4o", ... },
    # "anthropic": { "model": "claude-sonnet-4-6", ... },
    # "gemini": { "model": "gemini-2.0-flash", ... },
    # "ollama": { "model": "llama3", "base_url": "http://localhost:11434/v1", ... },
}
```

---

## 8. RAG Integration Strategy

### 8.1 Integration Model

The existing RAG pipeline is integrated as a **single controlled tool** named `faculty_bylaw_search` via a **direct Python import** (`from app.rag import ask`). The pipeline is treated as a black box: it accepts a natural-language question string and returns a natural-language answer string. The AI service does not manage embeddings, chunk retrieval, scoring, or re-ranking — all of that happens inside the existing pipeline.

```
LLM selects tool: faculty_bylaw_search(question="What is the max absence limit?")
    ↓
ToolExecutor validates role + args
    ↓
from app.rag import ask  →  ask(question)   ← direct Python call, no HTTP
    ↓
Returns: "Students may not exceed 25% unexcused absences per course..."  (plain string)
    ↓
ToolResult(success=True, data=answer_string)
    ↓
Appended to conversation as a "tool" role message
    ↓
LLM uses the answer string to compose its final response
```

### 8.2 Integration Interface

The RAG function is imported directly from the RAG project at startup. No HTTP client, no service URL, no network overhead. The RAG project exposes the following package layout:

```
# RAG project layout (co-located or installed in same environment)
app/
data/
```

The AI service imports the single public entry point:

```python
from app.rag import ask
# Signature: async ask(question: str) -> str
```

A thin adapter wraps it at the boundary for observability (logging, timeout enforcement) without modifying the original function:

```python
# ai_service/rag/pipeline_adapter.py

from typing import Callable, Awaitable

RAGPipelineFn = Callable[[str], Awaitable[str]]

def wrap_rag_pipeline(fn: RAGPipelineFn) -> RAGPipelineFn:
    """
    Thin wrapper around the imported RAG function.
    Adds a logging boundary without touching the original implementation.
    """
    async def wrapped(question: str) -> str:
        return await fn(question)
    return wrapped
```

Wired up at application startup:

```python
# ai_service/main.py  (inside lifespan)

from app.rag import ask                                          # direct Python import
from ai_service.rag.pipeline_adapter import wrap_rag_pipeline
from ai_service.tools.rag.faculty_bylaw_search import FacultyBylawSearchTool, faculty_bylaw_tool_def
from ai_service.tools.registry import ToolRegistry

rag_fn = wrap_rag_pipeline(ask)
faculty_bylaw_tool_def.handler = FacultyBylawSearchTool(rag_pipeline_fn=rag_fn).__call__
ToolRegistry.register(faculty_bylaw_tool_def)
```

### 8.3 What the Tool Result Looks Like in the Conversation

Because the RAG pipeline returns a plain string, the `tool` role message the LLM sees is simple:

```
role: "tool"
tool_call_id: "call_abc123"
content: "According to Article 14 of the faculty bylaw, students may not exceed
          25% unexcused absences in any course. Exceeding this limit results in
          an automatic failing grade regardless of academic performance."
```

The LLM then uses this content to formulate its final conversational response. It may paraphrase, add context from the session history, or combine it with data from another tool call in the same turn.

### 8.4 Persistence

Since the tool returns a plain string (not structured chunks), persistence is straightforward. The string answer is stored directly as the `content` of a `tool_result` message event. No citation extraction, no chunk scoring, no metadata parsing.

```python
# In MessagePersistence.save() — the tool_result event for a RAG call:
AIMessageEvent(
    session_id=session_id,
    role="tool",
    message_type=MessageType.TOOL_RESULT,
    content=tool_result.data,          # the plain string answer from the RAG pipeline
    tool_call_id=tool_call.id,
    tool_name="faculty_bylaw_search",
    metadata_json={
        "question": tool_call.arguments["question"],
        "success": True,
        "latency_ms": latency_ms,
    },
)
```

The existing `citationsJson` field on `AIMessage` is not applicable here — the RAG pipeline owns its own sourcing internally and the AI service does not receive or re-expose chunk-level metadata.

### 8.5 Error Handling

If the RAG pipeline is unavailable or returns an empty answer, the tool returns a graceful `ToolResult` and the LLM tells the user the information is temporarily unavailable. No special error path is needed beyond the standard `ToolExecutor` retry and failure handling.

```python
# From FacultyBylawSearchTool.__call__:
if not answer or not answer.strip():
    return ToolResult(
        success=True,
        data="I could not find relevant information about that in the faculty bylaw.",
    )
# The LLM receives this as the tool result and relays it conversationally.
# success=True is intentional: an empty result is a valid outcome, not a failure.
```

---

## 9. Database Interaction Design

### 10.1 Read-Only Access Principle

The AI service **never writes** to the core university schema. It only reads. This is enforced at the database connection level:

```python
# ai_service/db/connections.py

# Core university data: read-only role
CORE_DB_URL = settings.DATABASE_URL.replace(
    "postgresql://", f"postgresql://{settings.DB_READONLY_USER}:{settings.DB_READONLY_PASSWORD}@"
)

# AI-owned tables: read-write role (only for ai_schema tables)
AI_DB_URL = settings.DATABASE_URL  # Uses standard RW role, restricted to ai_schema via row-level security
```

### 10.2 Query Safety in Tools

All tool queries use:
- **Named bind parameters** (never f-strings or string concatenation)
- **Pre-written SQL** defined in the tool implementation
- **Schema-qualified table names** to prevent ambiguity
- **Result set limits** to prevent accidental large data extraction

```python
# Example: safe parameterized query in a tool
TRANSCRIPT_QUERY = """
    SELECT
        c.code, c.name, co.semester, co.year,
        gr.letter_grade, gr.grade_points, c.credit_hours
    FROM grade_records gr
    JOIN enrollments e ON gr.enrollment_id = e.id
    JOIN course_offerings co ON gr.offering_id = co.id
    JOIN courses c ON co.course_id = c.id
    WHERE e.student_id = :student_id
      AND gr.is_final = true
    ORDER BY co.year DESC, co.semester DESC
    LIMIT 100
"""
# user_id is ALWAYS from UserContext, never from LLM arguments
```

### 10.3 Async Database Sessions

```python
# ai_service/db/session.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_db() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
```

---

## 10. Schema Redesign Recommendations

### 11.1 Critical Evaluation of Existing Schema

The current schema (`AISession`, `AIMessage`, `AIFeedback`) has three fundamental limitations:

1. **`AIMessage` conflates all message roles** into a single `(userPrompt, aiAnswer)` pair. This prevents tool call persistence, system message storage, and streaming metadata.
2. **There are no tool execution logs** — no way to audit what tools were called, with what arguments, and what they returned.
3. **There is no observability table** — no token usage, no latency, no provider tracking.

### 11.2 Proposed Schema

```sql
-- ───────────────────────────────────────────────
-- AI Sessions (minimal changes to existing table)
-- ───────────────────────────────────────────────
ALTER TABLE "AISession" ADD COLUMN IF NOT EXISTS summary_text TEXT;
ALTER TABLE "AISession" ADD COLUMN IF NOT EXISTS last_active_at TIMESTAMPTZ;
ALTER TABLE "AISession" ADD COLUMN IF NOT EXISTS message_count INTEGER DEFAULT 0;

-- ─────────────────────────────────────────────────────────────
-- AI Message Events (replaces / extends existing AIMessage)
-- Event-based: one row per message role, not per request pair
-- ─────────────────────────────────────────────────────────────
CREATE TABLE ai_message_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES "AISession"(id) ON DELETE CASCADE,
    role            VARCHAR(20) NOT NULL, -- user | assistant | tool | system
    message_type    VARCHAR(30) NOT NULL, -- text | tool_call | tool_result | summary | system_instruction
    content         TEXT NOT NULL,
    tool_call_id    VARCHAR(100),         -- for role=tool, links to the tool_call that spawned it
    tool_name       VARCHAR(100),         -- populated for tool_call and tool_result events
    metadata_json   JSONB,               -- citations, scores, streaming chunks, prompt version
    is_summarized   BOOLEAN DEFAULT false,
    sequence_number INTEGER NOT NULL,     -- ordering within session
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ai_message_session ON ai_message_events(session_id, created_at);
CREATE INDEX idx_ai_message_session_role ON ai_message_events(session_id, role);

-- ──────────────────────────────────────────────
-- Tool Execution Logs
-- One row per tool invocation
-- ──────────────────────────────────────────────
CREATE TABLE ai_tool_execution_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES "AISession"(id),
    message_event_id UUID REFERENCES ai_message_events(id),
    tool_name       VARCHAR(100) NOT NULL,
    user_id         VARCHAR(100) NOT NULL,
    user_role       VARCHAR(50) NOT NULL,
    arguments_json  JSONB NOT NULL,
    result_json     JSONB,
    success         BOOLEAN NOT NULL,
    error_message   TEXT,
    attempt_number  SMALLINT DEFAULT 1,
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tool_log_session ON ai_tool_execution_logs(session_id);
CREATE INDEX idx_tool_log_tool ON ai_tool_execution_logs(tool_name, created_at);
CREATE INDEX idx_tool_log_user ON ai_tool_execution_logs(user_id, created_at);

-- ──────────────────────────────────────────────
-- Execution Traces (per-request observability)
-- One row per conversation turn
-- ──────────────────────────────────────────────
CREATE TABLE ai_execution_traces (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID NOT NULL REFERENCES "AISession"(id),
    request_id          VARCHAR(100) UNIQUE NOT NULL,
    user_id             VARCHAR(100) NOT NULL,
    user_role           VARCHAR(50) NOT NULL,
    provider_used       VARCHAR(50),       -- openrouter | groq
    model_used          VARCHAR(100),
    provider_fallback   BOOLEAN DEFAULT false,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    total_tokens        INTEGER,
    tool_calls_count    SMALLINT DEFAULT 0,
    tools_used          JSONB,             -- list of tool names called in this turn
    latency_ms          INTEGER,
    rag_chunks_retrieved SMALLINT DEFAULT 0,
    success             BOOLEAN NOT NULL DEFAULT true,
    error_type          VARCHAR(50),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trace_session ON ai_execution_traces(session_id);
CREATE INDEX idx_trace_user ON ai_execution_traces(user_id, created_at);
CREATE INDEX idx_trace_provider ON ai_execution_traces(provider_used, created_at);

-- ──────────────────────────────────────────────
-- AI Feedback (preserve existing, minor extension)
-- ──────────────────────────────────────────────
ALTER TABLE "AIFeedback" ADD COLUMN IF NOT EXISTS message_event_id UUID REFERENCES ai_message_events(id);
```

### 11.3 Migration Strategy

The existing `AIMessage` table can be migrated to the new event model via a one-time data migration script that converts each `(userPrompt, aiAnswer)` pair into two `ai_message_events` rows (one `user` + one `assistant`). The original table can be archived post-migration.

---

## 11. Observability & Debugging Architecture

### 12.1 Structured Logging

Every significant event emits a structured JSON log. The logging context is propagated via Python's `contextvars` to avoid threading issues in async code.

```python
# ai_service/observability/logging.py

import structlog
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="unknown")
session_id_var: ContextVar[str] = ContextVar("session_id", default="unknown")
user_id_var: ContextVar[str] = ContextVar("user_id", default="unknown")

logger = structlog.get_logger()

def bind_request_context(request_id: str, session_id: str, user_id: str) -> None:
    request_id_var.set(request_id)
    session_id_var.set(session_id)
    user_id_var.set(user_id)
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        session_id=session_id,
        user_id=user_id,
    )
```

**Log Events Emitted:**

| Event | Fields |
|---|---|
| `request.received` | request_id, session_id, user_role, message_length |
| `session.loaded` | session_id, message_count, has_summary |
| `memory.composed` | window_size, has_summary, token_estimate |
| `tools.exposed` | role, tool_count, tool_names |
| `tool.called` | tool_name, user_role, args_keys |
| `tool.completed` | tool_name, success, latency_ms, result_size |
| `tool.failed` | tool_name, error_type, attempt, latency_ms |
| `provider.called` | provider, model, prompt_tokens |
| `provider.responded` | provider, model, completion_tokens, latency_ms |
| `provider.fallback` | from_provider, to_provider, reason |
| `rag.retrieved` | collection, query, chunks_found, latency_ms |
| `response.streamed` | total_tokens, latency_ms, tools_used_count |

### 12.2 Request Tracing

Every request receives a `request_id` (UUID). This ID propagates through all log events and is persisted in `ai_execution_traces`. Developers can reconstruct the entire lifecycle of any conversation turn by querying on `request_id`.

```python
# The full trace of a single conversation turn is reconstructable:
# SELECT * FROM ai_execution_traces WHERE request_id = 'xxx';
# SELECT * FROM ai_message_events WHERE session_id = '...' AND created_at BETWEEN ...;
# SELECT * FROM ai_tool_execution_logs WHERE session_id = '...' AND created_at BETWEEN ...;
```

### 12.3 Debug Endpoints (Internal Only)

```
GET  /internal/debug/session/{session_id}/memory
     → Returns current memory state: window, summary, token counts

GET  /internal/debug/session/{session_id}/trace?last_n=5
     → Returns last N execution traces with tool call details

GET  /internal/debug/tools
     → Lists all registered tools with their schemas and role permissions

GET  /internal/debug/providers/health
     → Returns circuit breaker state and recent provider health

GET  /internal/debug/session/{session_id}/messages?include_tools=true
     → Full message event history including tool calls and results
```

These endpoints are gated behind an internal network policy and require an `X-Internal-Key` header. They are never exposed to the public internet.

---

## 12. Error Handling & Reliability

### 13.1 Error Taxonomy

```python
# ai_service/errors.py

class AIServiceError(Exception): ...

# Authentication
class TokenExpiredError(AIServiceError): ...
class InvalidTokenError(AIServiceError): ...

# Authorization
class ToolAuthorizationError(AIServiceError): ...
class SessionOwnershipError(AIServiceError): ...

# Tool Execution
class ToolTimeoutError(AIServiceError): ...
class ToolArgumentError(AIServiceError): ...
class ToolExecutionError(AIServiceError): ...

# Provider
class ProviderRateLimitError(AIServiceError): ...
class ProviderUnavailableError(AIServiceError): ...
class ProviderTimeoutError(AIServiceError): ...
class ProviderExhaustedError(AIServiceError): ...

# Memory
class MemoryLoadError(AIServiceError): ...
class SessionNotFoundError(AIServiceError): ...

# RAG
class RAGRetrievalError(AIServiceError): ...
class RAGUnavailableError(AIServiceError): ...
```

### 13.2 Graceful Degradation Strategy

| Failure | Behavior |
|---|---|
| RAG pipeline down | Tool returns a safe error result; LLM informs user retrieval is unavailable; conversation continues |
| Tool DB timeout | Retry up to 2 times; if all fail, return structured error result to LLM; LLM tells user the data is temporarily unavailable |
| Primary provider rate limited | Transparent failover to Groq; logged as a `provider.fallback` event |
| Both providers fail | Return a graceful HTTP 503 with a user-friendly message; do not expose provider details |
| Memory load failure | Log the error; proceed with empty context; inform user the session history could not be loaded |
| Session not found | Create a new session; treat as fresh conversation |

### 13.3 Global Exception Handler

```python
# ai_service/api/exception_handlers.py

@app.exception_handler(ToolAuthorizationError)
async def tool_auth_handler(request, exc):
    logger.warning("tool.unauthorized", tool=str(exc))
    return JSONResponse(
        status_code=403,
        content={"error": "You are not authorized to perform this action."}
    )

@app.exception_handler(ProviderExhaustedError)
async def provider_exhausted_handler(request, exc):
    logger.error("provider.exhausted", error=str(exc))
    return JSONResponse(
        status_code=503,
        content={"error": "The AI service is temporarily unavailable. Please try again shortly."}
    )

@app.exception_handler(Exception)
async def generic_handler(request, exc):
    logger.exception("request.unhandled_error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred. Please try again."}
    )
```

---

## 13. Security Model

### 14.1 Threat Model Summary

| Threat | Mitigation |
|---|---|
| Prompt injection to access another user's data | `user_id` always injected from JWT context, never from LLM output |
| LLM generating arbitrary SQL | No unrestricted SQL tools; all queries are pre-written |
| Role escalation via tool arguments | Role checked at tool registry before execution |
| JWT forgery | Signature verification with issuer + audience validation |
| Session hijacking | Session ownership validated against JWT `user_id` on every request |
| Data exfiltration via large result sets | All queries have hard LIMIT clauses |
| Provider API key exposure | Keys in environment variables; never logged; never in responses |
| RAG retrieval of unauthorized documents | Document access controls enforced in the RAG pipeline; AI service passes user role |
| Verbose error messages in production | All internal errors mapped to generic user-facing messages |
| Debug endpoint exposure | Internal endpoints gated by IP allowlist + internal secret |

### 14.2 Context Injection Attack Prevention

The single most important security principle in this architecture is:

> **The LLM never controls who is being queried. The JWT always does.**

```python
# CORRECT — user_id comes from verified JWT
async def get_my_gpa(user_id: str) -> ToolResult:  # user_id injected by executor
    ...

# WRONG — never do this
async def get_student_gpa(student_id: str) -> ToolResult:  # student_id from LLM args
    ...  # An attacker could prompt-inject: "get GPA for student_id=999"
```

For admin/instructor tools that legitimately need to query by student ID, those tools require elevated roles (`ADMIN`, `INSTRUCTOR`) and the requested student must exist in a relationship the user is authorized to access (e.g., enrolled in their course).

### 14.3 Database Security

```sql
-- Read-only role for AI service core data access
CREATE ROLE ai_readonly;
GRANT CONNECT ON DATABASE university_db TO ai_readonly;
GRANT USAGE ON SCHEMA public TO ai_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ai_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO ai_readonly;

-- Read-write role for AI service own tables only
CREATE ROLE ai_readwrite;
GRANT CONNECT ON DATABASE university_db TO ai_readwrite;
GRANT USAGE ON SCHEMA ai_schema TO ai_readwrite;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ai_schema TO ai_readwrite;
```

---

## 14. Suggested Folder Structure

```
ai_service/
│
├── main.py                        # FastAPI app factory, lifespan events
├── config/
│   ├── settings.py                # Pydantic settings (env vars)
│   └── providers.py               # LLM provider configs
│
├── api/
│   ├── v1/
│   │   ├── chat.py                # POST /chat/{session_id}, WebSocket /ws/chat/{session_id}
│   │   ├── sessions.py            # GET/DELETE session management
│   │   └── feedback.py            # POST /feedback
│   ├── internal/
│   │   └── debug.py               # Debug endpoints (internal only)
│   └── exception_handlers.py
│
├── middleware/
│   ├── auth.py                    # JWT verification, UserContext injection
│   └── request_id.py             # X-Request-ID propagation
│
├── models/
│   ├── user_context.py            # UserContext, UserRole dataclasses
│   ├── messages.py                # Message, ToolCall, ToolResult models
│   └── session.py                 # Session domain models
│
├── orchestration/
│   ├── conversation_orchestrator.py  # Main execution loop (agentic tool loop)
│   ├── prompt_builder.py             # Role-aware system prompt construction
│   └── streaming.py                  # Response streaming utilities
│
├── tools/
│   ├── base.py                    # ToolDefinition, ToolResult, ToolDomain
│   ├── registry.py                # ToolRegistry, role-to-tool permission map
│   ├── executor.py                # ToolExecutor with retry, logging, auth
│   ├── student/
│   │   ├── gpa.py
│   │   ├── schedule.py
│   │   ├── transcript.py
│   │   └── attendance.py
│   ├── instructor/
│   │   ├── course_students.py
│   │   ├── student_progress.py
│   │   └── course_attendance.py
│   ├── admin/
│   │   ├── registration_statistics.py
│   │   └── all_students.py
│   └── rag/
│       └── faculty_bylaw_search.py    # Single RAG tool wrapping the existing pipeline
│
├── memory/
│   ├── short_term.py              # Message windowing
│   ├── long_term.py               # Summarization + persistence
│   └── composer.py                # Memory composition for prompt
│
├── providers/
│   ├── base.py                    # LLMProvider abstract interface
│   ├── openrouter.py              # OpenRouter adapter (GLM 4.5 Air)
│   ├── groq.py                    # Groq adapter (LLaMA 3.3)
│   ├── failover.py                # Circuit breaker + failover orchestrator
│   └── circuit_breaker.py
│
├── rag/
│   └── pipeline_adapter.py        # Thin wrapper injecting the existing RAG function
│
├── sessions/
│   └── manager.py                 # Session load/create/validate
│
├── persistence/
│   └── message_writer.py          # Persist message events + traces + tool logs
│
├── db/
│   ├── session.py                 # SQLAlchemy async engine + session factory
│   └── models.py                  # SQLAlchemy ORM models for AI-owned tables
│
├── observability/
│   ├── logging.py                 # Structlog configuration + context binding
│   ├── tracing.py                 # OpenTelemetry span management
│   └── metrics.py                 # Prometheus metrics (token usage, latency)
│
├── errors.py                      # Full error taxonomy
│
├── migrations/
│   ├── env.py
│   └── versions/
│       └── 001_create_ai_schema.py
│
└── tests/
    ├── unit/
    │   ├── test_tools/
    │   ├── test_memory/
    │   ├── test_routing/
    │   └── test_providers/
    ├── integration/
    │   ├── test_conversation_flow.py
    │   └── test_tool_execution.py
    └── fixtures/
        ├── mock_provider.py
        └── mock_db.py
```

---

## 15. Production Deployment Considerations

### 16.1 Environment Configuration

```bash
# Required environment variables
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/university_db
DB_READONLY_USER=ai_readonly
DB_READONLY_PASSWORD=...
OPENROUTER_API_KEY=...
GROQ_API_KEY=...
JWT_SECRET=...            # HS256 shared secret from the existing backend
INTERNAL_API_KEY=...      # For debug endpoints
ENVIRONMENT=production    # production | staging | development
```

### 16.2 FastAPI App Factory with Lifespan

```python
# ai_service/main.py

from contextlib import asynccontextmanager
from app.rag import ask as rag_ask                          # direct Python import
from ai_service.rag.pipeline_adapter import wrap_rag_pipeline
from ai_service.tools.rag.faculty_bylaw_search import FacultyBylawSearchTool, faculty_bylaw_tool_def
from ai_service.tools.registry import ToolRegistry

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db_engine.connect()

    # Wire RAG: import the function, wrap it, register the tool
    rag_fn = wrap_rag_pipeline(rag_ask)
    faculty_bylaw_tool_def.handler = FacultyBylawSearchTool(rag_pipeline_fn=rag_fn).__call__
    ToolRegistry.initialize_all()  # Register all other tool definitions

    await provider_orchestrator.warm_up()
    yield
    # Shutdown
    await db_engine.dispose()

app = FastAPI(
    title="University AI Assistant",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if settings.ENVIRONMENT == "production" else "/docs",
)
```

### 16.3 Docker Configuration

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY ai_service/ ./ai_service/
COPY migrations/ ./migrations/

# Non-root user for security
RUN useradd -m aiuser && chown -R aiuser /app
USER aiuser

CMD ["uvicorn", "ai_service.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "4"]
```

### 16.4 Key Production Dependencies

```toml
# pyproject.toml
[project.dependencies]
fastapi = ">=0.115"
uvicorn = { extras = ["standard"] }
sqlalchemy = { extras = ["asyncio"] }
asyncpg = "*"
alembic = "*"
python-jose = { extras = ["cryptography"] }
httpx = "*"
pydantic = ">=2.0"
pydantic-settings = "*"
structlog = "*"
opentelemetry-sdk = "*"
opentelemetry-instrumentation-fastapi = "*"
prometheus-fastapi-instrumentator = "*"
tenacity = "*"
tiktoken = "*"  # Token counting
```

---

## 16. Scalability Recommendations

### 17.1 Horizontal Scaling

The AI service is designed to be stateless at the request level. All session state is in PostgreSQL. Multiple instances can run behind a load balancer without sticky sessions.

```
                    ┌────────────┐
                    │Load Balancer│
                    └─────┬──────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
   ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
   │ AI Service  │ │ AI Service  │ │ AI Service  │
   │ Instance 1  │ │ Instance 2  │ │ Instance 3  │
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          └───────────────┼───────────────┘
                          │
                   ┌──────▼──────┐
                   │ PostgreSQL  │
                   │ (with PgBouncer)│
                   └─────────────┘
```

### 17.2 Async Memory Compression

Memory summarization is triggered asynchronously — it does not block the response stream. Use a background task queue (e.g., ARQ or Celery) for compression jobs.

```python
# Triggered after response is sent, not during
background_tasks.add_task(memory_manager.maybe_compress, session_id, db)
```

### 17.3 Connection Pooling

Use PgBouncer in transaction pooling mode between the AI service and PostgreSQL. Each AI service instance should configure `pool_size=10, max_overflow=20`. With 3 instances this yields 30–90 connections through PgBouncer.

### 17.4 Tool Result Caching

For read-heavy tools with low volatility (e.g., transcript, GPA), add Redis caching with short TTLs. Cache keys are scoped to `user_id + tool_name + args_hash`.

```python
TOOL_CACHE_TTL = {
    "get_my_gpa": 300,        # 5 minutes
    "get_my_transcript": 600, # 10 minutes
    "get_my_schedule": 60,    # 1 minute
    "regulation_search": 3600, # 1 hour (static content)
}
```

---

## 17. Developer Experience Guide

### 17.1 Adding a New Tool

1. Create `ai_service/tools/{domain}/my_tool.py`
2. Implement the handler as an `async def __call__(self, ...) -> ToolResult`
3. Define a `ToolDefinition` and call `ToolRegistry.register(...)` in an `__init__.py`
4. Add the tool name to `ROLE_TOOL_PERMISSIONS` for appropriate roles
5. Write a unit test in `tests/unit/test_tools/`

```python
# Step 1-3: Complete tool in one file
class GetMyEnrollmentStatusTool:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def __call__(self, user_id: str) -> ToolResult:
        # Implementation here
        ...

enrollment_tool = ToolDefinition(
    name="get_my_enrollment_status",
    description="Get the student's current enrollment status and registered courses.",
    domain=ToolDomain.DATABASE,
    allowed_roles={UserRole.STUDENT},
    parameters=[],
    handler=GetMyEnrollmentStatusTool(db=...).execute,
    tags=["student", "enrollment"],
)

# Step 3: Register
ToolRegistry.register(enrollment_tool)

# Step 4: Permissions
ROLE_TOOL_PERMISSIONS[UserRole.STUDENT].add("get_my_enrollment_status")
```

### 17.2 Debugging a Conversation Turn

```bash
# 1. Get the request_id from logs or response headers (X-Request-ID)

# 2. Pull the full execution trace
SELECT * FROM ai_execution_traces WHERE request_id = 'your-request-id';

# 3. Pull all message events for that session around that time
SELECT role, message_type, content, tool_name, created_at
FROM ai_message_events
WHERE session_id = 'your-session-id'
ORDER BY sequence_number;

# 4. Pull tool execution details
SELECT tool_name, arguments_json, result_json, success, latency_ms, error_message
FROM ai_tool_execution_logs
WHERE session_id = 'your-session-id'
ORDER BY created_at;

# 5. Use the internal debug endpoint for memory state
curl -H "X-Internal-Key: $INTERNAL_KEY" \
  http://localhost:8002/internal/debug/session/{session_id}/memory
```

### 17.3 Inspecting Tool Selection Decisions

Since there is no router, what you debug is the LLM's tool selection — which tools it called, in what order, and what arguments it generated. All of this is captured in `ai_tool_execution_logs` and `ai_message_events`.

```bash
# See which tools were called in a session
SELECT tool_name, arguments_json, success, latency_ms
FROM ai_tool_execution_logs
WHERE session_id = 'your-session-id'
ORDER BY created_at;

# See the full message thread including tool calls and results
SELECT sequence_number, role, message_type, tool_name, LEFT(content, 200)
FROM ai_message_events
WHERE session_id = 'your-session-id'
ORDER BY sequence_number;

# Check what tools were used per turn
SELECT tools_used, total_tokens, latency_ms
FROM ai_execution_traces
WHERE session_id = 'your-session-id'
ORDER BY created_at DESC;
```

If the LLM is calling the wrong tool or not calling one when it should, the fix is almost always **improving the tool's description** — not adding routing logic. Revisit the description to make it more specific about when to use or not use the tool.

### 17.4 Monitoring Provider Health

```bash
# Internal health endpoint
curl -H "X-Internal-Key: $INTERNAL_KEY" \
  http://localhost:8002/internal/debug/providers/health

# Response:
{
  "openrouter": {
    "circuit_breaker_state": "closed",
    "consecutive_failures": 0,
    "last_failure_at": null,
    "last_latency_ms": 1240
  },
  "groq": {
    "circuit_breaker_state": "closed",
    "consecutive_failures": 0,
    "last_latency_ms": 890
  }
}
```

### 17.5 Replaying a Failed Execution

Since every tool execution stores its arguments in `ai_tool_execution_logs`, failed executions can be replayed:

```python
# ai_service/dev/replay.py

async def replay_tool_execution(log_id: str, db: AsyncSession) -> ToolResult:
    log = await db.get(AIToolExecutionLog, log_id)
    tool_def = ToolRegistry.get(log.tool_name)
    mock_context = UserContext(user_id=log.user_id, role=UserRole(log.user_role), permissions=[])
    return await ToolExecutor(db).execute(
        ToolCall(name=log.tool_name, arguments=log.arguments_json),
        mock_context,
    )
```

---

*This document represents the complete production architecture blueprint for the University AI Assistant service. All design decisions have been made with the constraints of an existing backend ecosystem, a shared PostgreSQL database under Prisma ownership, and a pre-existing RAG pipeline that must be reused and integrated — not replaced.*
