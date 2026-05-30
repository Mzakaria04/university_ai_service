Overview
--------

Phase 1 establishes the core FastAPI microservice with end-to-end request handling: JWT authentication, a single AI-owned persistence layer, one database tool, OpenRouter integration, and SSE streaming. The service is fully runnable locally via Docker with PostgreSQL. This phase intentionally implements the minimal viable surface area—only one tool and no memory, RAG, or observability—to prove the execution pipeline before expanding it.

Goals
-----

*   Create a working local FastAPI application with async lifespan management.
    
*   Implement HS256 JWT verification middleware that injects a UserContext into every request.
    
*   Establish async SQLAlchemy database sessions against the existing PostgreSQL instance.
    
*   Persist conversational data to the existing ai\_sessions and ai\_message\_events tables only.
    
*   Integrate OpenRouter as the sole LLM provider (no failover).
    
*   Build the minimal tool architecture: ToolDefinition, ToolRegistry, and ToolExecutor.
    
*   Implement the first database tool: get\_my\_gpa.
    
*   Deliver a single SSE chat endpoint that completes the full flow: User → JWT → LLM → Tool → Database → Response.
    

Architecture Components
-----------------------

*   **FastAPI Application (main.py)** — App factory with asynccontextmanager lifespan. Startup connects the async engine and initializes the tool registry.
    
*   **Configuration (config/settings.py, config/providers.py)** — Pydantic-settings for env vars. providers.py holds the OpenRouter config only.
    
*   **JWT Authentication Middleware (middleware/auth.py)** — Extracts Bearer tokens, verifies HS256 signatures against JWT\_SECRET, and constructs UserContext from the user claim.
    
*   **Request ID Middleware (middleware/request\_id.py)** — Injects X-Request-ID for request correlation.
    
*   **UserContext & UserRole (models/user\_context.py)** — Frozen dataclass with user\_id, university\_id, full\_name, role. Enum values: STUDENT, INSTRUCTOR, ADMIN.
    
*   **Message Models (models/messages.py)** — Message, ToolCall, ToolResult domain models for internal orchestration.
    
*   **Session Models (models/session.py)** — Domain representation for ai\_sessions records.
    
*   **Async Database (db/session.py, db/models.py)** — create\_async\_engine with async\_sessionmaker, pool\_pre\_ping=True, and get\_db() dependency. SQLAlchemy ORM models for ai\_sessions and ai\_message\_events.
    
*   **OpenRouter Provider (providers/base.py, providers/openrouter.py)** — LLMProvider abstract base. OpenRouterProvider implements chat() and complete() using thudm/glm-4.5-air with OpenAI-compatible payload serialization.
    
*   **Conversation Orchestrator (orchestration/conversation\_orchestrator.py)** — Bounded agentic loop (max 5 iterations). Handles LLM calls, tool execution, and final SSE streaming. In Phase 1, only single-tool turns are practically exercised.
    
*   **Prompt Builder (orchestration/prompt\_builder.py)** — Minimal system prompt template. Includes role, user display name, and a flat list of authorized tools.
    
*   **Streaming Utilities (orchestration/streaming.py)** — Async generator wrappers for SSE formatting.
    
*   **Tool Base (tools/base.py)** — ToolDefinition, ToolResult, ToolDomain, ToolParameter.
    
*   **Tool Registry (tools/registry.py)** — Global registry with register(), get\_authorized\_tools(role), and is\_authorized(). ROLE\_TOOL\_PERMISSIONS contains only get\_my\_gpa for STUDENT in this phase.
    
*   **Tool Executor (tools/executor.py)** — Single point of control: authorization check, argument validation, user\_id context injection from JWT (never from LLM), execution with timeout, and result capture.
    
*   **Student GPA Tool (tools/student/gpa.py)** — GetMyGpaTool queries grade\_records, enrollments, course\_offerings, and courses using named bind parameters and returns cumulative GPA, total credits, and courses completed.
    
*   **Session Manager (sessions/manager.py)** — load\_or\_create() and ownership validation against user\_id.
    
*   **Message Persistence (persistence/message\_writer.py)** — Writes user and assistant ai\_message\_events rows with sequence\_number ordering.
    

Folder Structure
----------------

**plain**

```text
ai_service/
├── main.py
├── config/
│   ├── settings.py
│   └── providers.py
├── api/
│   └── v1/
│       ├── chat.py
│       └── sessions.py
├── middleware/
│   ├── auth.py
│   └── request_id.py
├── models/
│   ├── user_context.py
│   ├── messages.py
│   └── session.py
├── orchestration/
│   ├── conversation_orchestrator.py
│   ├── prompt_builder.py
│   └── streaming.py
├── tools/
│   ├── base.py
│   ├── registry.py
│   ├── executor.py
│   └── student/
│       └── gpa.py
├── sessions/
│   └── manager.py
├── persistence/
│   └── message_writer.py
├── db/
│   ├── session.py
│   └── models.py
├── errors.py
└── tests/
    ├── unit/
    │   ├── test_auth.py
    │   ├── test_tools.py
    │   └── test_orchestrator.py
    └── integration/
        └── test_chat_e2e.py
```

Database Usage
--------------

The AI service consumes the existing finalized schema. No migrations, no new tables, no schema changes.

*   **ai\_sessions** — Inserted on first chat request (lazy creation). Validated on every subsequent request: the JWT user\_id must match the session's user\_id. Columns used: id, user\_id, title, created\_at, updated\_at.
    
*   **ai\_message\_events** — Inserted per turn. In Phase 1, only role values user and assistant are stored. Columns used: id, session\_id, role, message\_type (text), content, sequence\_number, created\_at.
    

APIs
----

* POST /api/v1/chat/{session\_id}
**Headers:** Authorization: Bearer 
**Body:** { "message": "What is my GPA?" }
**Response:** text/event-stream (SSE). Streams the assistant's final response after any internal tool execution.
**Flow:** JWT → load session → build minimal prompt → call OpenRouter → if tool call → execute get\_my\_gpa → append result → call OpenRouter again → stream final text.
    
*   POST /api/v1/sessions _(optional explicit creation)_
**Response:** { "session\_id": "uuid" }Creates a new ai\_sessions row for the authenticated user.

Services
--------
| Service                    | Responsibility                                                                                           |
| -------------------------- | -------------------------------------------------------------------------------------------------------- |
| `JWTAuthMiddleware`        | Validates HS256 tokens, raises 401 on missing/invalid, injects `UserContext` into `request.state`.       |
| `SessionManager`           | Loads `ai_sessions` by ID, verifies ownership, creates new sessions on demand.                           |
| `OpenRouterProvider`       | Serializes messages to OpenAI-compatible format, calls `/chat/completions`, handles streaming responses. |
| `ConversationOrchestrator` | Runs the bounded agentic loop: LLM call → tool execution → LLM recall → SSE streaming.                   |
| `ToolExecutor`             | Enforces role authorization, injects `user_id` from JWT, runs handler, returns `ToolResult`.             |
| `MessagePersistence`       | Persists user and assistant messages to `ai_message_events` with monotonic `sequence_number`.            |

Tools
-----

| Tool         | Domain   | Authorized Roles | Description                                                                                                                                                                                    |
| ------------ | -------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_my_gpa` | DATABASE | `STUDENT`        | Retrieves the authenticated student's cumulative GPA, total credits, and completed courses from the live database. `user_id` is injected from `UserContext`; no parameters exposed to the LLM. |


Dependencies
------------


```toml
fastapi = ">=0.115"
uvicorn = { extras = ["standard"] }
sqlalchemy = { extras = ["asyncio"] }
asyncpg = "*"
python-jose = { extras = ["cryptography"] }
httpx = "*"
pydantic = ">=2.0"
pydantic-settings = "*"
pydantic-settings = "*"
```  

Environment Variables
---------------------

| Variable             | Purpose                                   | Example                                                     |
| -------------------- | ----------------------------------------- | ----------------------------------------------------------- |
| `DATABASE_URL`       | PostgreSQL async connection               | `postgresql+asyncpg://postgres:1234@localhost:5432/edux_db` |
| `JWT_SECRET`         | HS256 shared secret from existing backend | `super-secret-key`                                          |
| `OPENROUTER_API_KEY` | Primary LLM provider key                  | `sk-or-...`                                                 |
| `ENVIRONMENT`        | Runtime environment                       | `development`                                               |


Deliverables
------------

1.  Working FastAPI service runnable via docker-compose up with a local PostgreSQL container.
    
2.  JWT middleware rejecting unauthenticated requests.
    
3.  End-to-end chat endpoint where a STUDENT user can ask "What is my GPA?" and receive a live database answer streamed via SSE.
    
4.  ai\_sessions and ai\_message\_events populated correctly after each turn.
    
5.  Unit tests for middleware, tool executor, and the orchestrator loop.
    
6.  Integration test covering the full request lifecycle with a mocked OpenRouter response.
    

Testing Strategy
----------------

*   **Unit:** Mock the OpenRouter provider to return a forced tool\_call for get\_my\_gpa, verify ToolExecutor invokes the handler with JWT-injected user\_id, and verify the final assistant message is persisted.
    
*   **Unit:** Test JWTAuthMiddleware with valid, expired, and malformed tokens.
    
*   **Integration:** Spin up PostgreSQL in Docker, create a session via HTTP, send a chat message, and assert ai\_message\_events contains exactly two rows (user, assistant).
    
*   **Security:** Attempt to access get\_my\_gpa with an INSTRUCTOR JWT and assert ToolAuthorizationError (403).
    

Success Criteria
----------------

*   \[ \] Authenticated SSE chat returns a coherent response.
    
*   \[ \] get\_my\_gpa correctly queries the existing university schema and returns live data.
    
*   \[ \] user\_id is never accepted from the LLM; it is always injected from the JWT context.
    
*   \[ \] Messages are persisted to ai\_message\_events with correct sequence\_number ordering.
    
*   \[ \] Service runs entirely in Docker with no external dependencies beyond PostgreSQL and OpenRouter.
    

Anti-Overengineering Notes
--------------------------

*   **No RAG.** The faculty\_bylaw\_search tool is explicitly excluded until Phase 2.
    
*   **No memory summarization.** Only the current user message and system prompt are sent to the LLM; no historical window is loaded.
    
*   **No observability stack.** No structlog, OpenTelemetry, Prometheus, or execution traces.
    
*   **No provider failover.** Only OpenRouter is wired; no Groq adapter, no circuit breaker.
    
*   **No Redis or caching.** Every tool call hits the database directly.
    
*   **No internal debug endpoints.** All /internal/\* routes are excluded.
    
*   **No background tasks.** All work is synchronous within the request/response cycle.
    
*   **One tool only.** get\_my\_gpa is the sole registered tool to keep the pipeline simple and verifiable.

