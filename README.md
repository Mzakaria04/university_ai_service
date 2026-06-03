# University AI Assistant Microservice

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-316192?style=for-the-badge&logo=postgresql)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=for-the-badge&logo=redis)](https://redis.io/)
[![Docker](https://img.shields.io/badge/Docker-Collab-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)

An independent, production-grade **FastAPI microservice** acting as the intelligence layer for a University Management System. This service implements a secure, role-aware, prompt-orchestrated conversational agent utilizing standard LLM tool-calling primitives. It leverages a dual-provider LLM failover stack, short-term and long-term memory management, database tool execution, Redis caching, co-located RAG search, and comprehensive production observability (OpenTelemetry, Prometheus, and structured logging).

---

## 📖 Table of Contents
1. [System Overview & Architecture](#1-system-overview--architecture)
2. [Key Capabilities & Features](#2-key-capabilities--features)
3. [Folder Structure](#3-folder-structure)
4. [Database Design & Schema Mappings](#4-database-design--schema-mappings)
5. [LLM Provider Stack & Circuit Breaker](#5-llm-provider-stack--circuit-breaker)
6. [Security & Role-Based Authorization](#6-security--role-based-authorization)
7. [API Specifications](#7-api-specifications)
8. [Configuration & Environment Variables](#8-configuration--environment-variables)
9. [Local Development & Docker Deployment](#9-local-development--docker-deployment)
10. [Testing & Verification](#10-testing--verification)

---

## 1. System Overview & Architecture

The AI service does not own user management, core business logic mutations, or primary tables. Instead, it interacts securely with the existing systems:
* **Existing Backend**: Integrates via stateless verification of HS256 JWT tokens.
* **Shared Database**: Uses a read-only PostgreSQL role for university data (grades, rosters, schedules) and a read-write role restricted to the dedicated AI schema (`ai_schema`).
* **Existing RAG Pipeline**: Calls a co-located FAISS + BM25 search pipeline directly as a local Python package import to execute official faculty bylaw queries.

### High-Level Architecture Diagram
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
│  │  │                  Execution Pipeline                         │  │
│  │  │  Memory Load → LLM + All Authorized Tools → Tool Executor  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │
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

## 2. Key Capabilities & Features

### ⚡ Bounded Agentic Loop & SSE Streaming
* Streams completions using standard **Server-Sent Events (SSE)** matching the OpenAI Choices delta format.
* Employs an agentic turn-handling loop capped at a maximum of **5 iterations** to prevent infinite execution loops while allowing sequential tool use (e.g., retrieving a schedule, then fetching course attendance).

### 🧠 Dual-Layer Conversational Memory
* **Short-Term Memory**: Automatically loads the last 12 message events (6 conversation turns) from the database and trims context window parameters chronologically to fit within a parameterized **3000-token budget** using `tiktoken`.
* **Long-Term Memory**: When unsummarized context exceeds **20 messages** or **6000 tokens**, a background summarization task (offloaded asynchronously to FastAPI `BackgroundTasks`) is triggered. The LLM produces a concise summary (max 300 words) stored in the database as a `summary` message event, which is injected on subsequent session loads.

### 🔌 Resilient LLM Failover Stack
* **Primary Provider**: OpenRouter calling `thudm/glm-4.5-air`.
* **Fallback Provider**: Groq calling `llama-3.3-70b-versatile`.
* **Circuit Breaker**: Tracks health states. If the primary provider experiences 3 consecutive timeouts or rate limits (HTTP 429 / 5xx), the circuit opens, automatically routing all requests directly to Groq for a recovery period of 60 seconds.

### 🛡️ Secure Tool Execution Layer
* Resolves tool authorization checks at the registry level before the LLM can see them.
* Validates arguments against parameter definitions.
* **Context Injection**: Security-critical variables like `user_id` are injected directly from the verified JWT context, never from LLM-generated arguments, preventing context-injection attacks.
* Wraps invocations in `tenacity` retry decorators (up to 2 retries with exponential backoff) and enforces strict timeouts.

### 🗄️ Database & Cache Optimization
* **Dual Engines**: Routes session and telemetry logs to a primary write-enabled database engine, while university-specific queries route to a read-only role (`DB_READONLY_USER`) engine.
* **Redis Caching**: Caches result sets of read-heavy databases and RAG tools (GPA, schedule, transcript, bylaw search).
* **Cross-User Data Isolation**: Cache keys are securely isolated using `tool:{user_id}:{tool_name}:{args_hash}` with role-appropriate Time-To-Live (TTL) values.

### 📊 Production Observability
* **Structured Logs**: Configures `structlog` for JSON-formatted logs in production and color-coded logging in development. Injects request correlation metadata (`request_id`, `user_id`, `role`) bound via contextvars middleware.
* **OpenTelemetry Tracing**: Instruments FastAPI routers, LLM provider calls, and tool executors, producing traceable spans.
* **Prometheus Metrics**: Exposes a standard `/metrics` endpoint measuring latency, token usage, provider failover rates, and active requests.
* **Audit Trails**: Automatically persists execution metrics to dedicated DB tables (`ai_tool_execution_logs` and `ai_execution_traces`).

---

## 3. Folder Structure

```text
ai_service/
├── Dockerfile                         # Production Docker configurations
├── docker-compose.yml                 # Local development compose
├── docker-compose.prod.yml            # Hardened production stack (PgBouncer + Redis)
├── requirements.txt                   # Project package dependencies
│
├── ai_service/                        # Main Application Package
│   ├── main.py                        # FastAPI Lifespan configuration & entrypoint
│   ├── errors.py                      # Core Exception Taxonomy
│   │
│   ├── api/
│   │   ├── v1/
│   │   │   ├── chat.py                # SSE /chat/{session_id} Endpoint
│   │   │   ├── sessions.py            # Session Retrieval & Deletion
│   │   │   └── feedback.py            # Feedback (rating + comments)
│   │   └── internal/
│   │       └── debug.py               # Administrative Debug Endpoints
│   │
│   ├── config/
│   │   ├── settings.py                # Pydantic Settings
│   │   └── providers.py               # LLM Provider Configuration Matrix
│   │
│   ├── db/
│   │   ├── session.py                 # Multi-engine SQLAlchemy connections
│   │   └── models.py                  # Database Models (ai_schema maps)
│   │
│   ├── memory/
│   │   ├── short_term.py              # sliding window tiktoken trimming
│   │   ├── long_term.py               # Asynchronous LLM memory summarization
│   │   └── composer.py                # Prompts context assembly
│   │
│   ├── middleware/
│   │   ├── auth.py                    # HS256 JWT validation & UserContext injection
│   │   └── request_id.py              # X-Request-ID propagation
│   │
│   ├── observability/
│   │   ├── logging.py                 # JSON structlog configuration
│   │   ├── metrics.py                 # Prometheus meters
│   │   └── tracing.py                 # OpenTelemetry Tracer setup
│   │
│   ├── orchestration/
│   │   ├── conversation_orchestrator.py # Bounded agentic loops
│   │   ├── prompt_builder.py          # Dynamic system prompts
│   │   └── streaming.py               # SSE Formatter
│   │
│   ├── providers/
│   │   ├── base.py                    # LLMProvider Contract
│   │   ├── openrouter.py              # OpenRouter Adapter (GLM 4.5 Air)
│   │   ├── groq.py                    # Groq Adapter (LLaMA 3.3)
│   │   ├── circuit_breaker.py         # State tracker
│   │   └── failover.py                # Provider Fallback Orchestrator
│   │
│   ├── persistence/
│   │   └── message_writer.py          # Saves message events, logs, and traces
│   │
│   └── tools/
│       ├── base.py                    # Tool definitions and parameters schemas
│       ├── registry.py                # Registry & role permissions
│       ├── executor.py                # Thread-safe executor, retry, caching
│       │
│       ├── student/                   # Student-scoped tools
│       │   ├── gpa.py                 # Cumulative GPA queries
│       │   ├── schedule.py            # Term timetables
│       │   ├── transcript.py          # Term course results
│       │   └── attendance.py          # Attendance percentages
│       │
│       ├── instructor/                # Instructor-scoped tools
│       │   ├── course_students.py     # Class rosters
│       │   ├── student_progress.py    # Assignment & exam grades
│       │   ├── course_attendance.py   # Roster presence statistics
│       │   └── get_my_schedule.py     # Classroom schedules
│       │
│       ├── admin/                     # Admin-scoped tools
│       │   ├── registration_statistics.py # Enrollment aggregations
│       │   └── all_students.py        # Paginated student rosters
│       │
│       └── rag/
│           └── faculty_bylaw_search.py # Co-located regulations search
│
└── docs/                              # Project Blueprints & Tasks lists
```

---

## 4. Database Design & Schema Mappings

The microservice maps directly to five tables in `db/models.py`, executing without database schema modification constraints:

```mermaid
erDiagram
    "AISession" ||--o{ "AIMessageEvent" : contains
    "AISession" ||--o{ "AIToolExecutionLog" : generates
    "AISession" ||--o{ "AIExecutionTrace" : traces
    "AIMessageEvent" ||--o{ "AIFeedback" : receives

    "AISession" {
        UUID id PK
        VARCHAR title
        VARCHAR user_id FK
        TEXT summary_text
        INTEGER message_count
        TIMESTAMPTZ last_active_at
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        JSONB metadata_json
    }

    "AIMessageEvent" {
        UUID id PK
        UUID session_id FK
        VARCHAR role "user | assistant | tool | system"
        VARCHAR message_type "text | tool_call | tool_result | summary"
        TEXT content
        VARCHAR tool_call_id
        VARCHAR tool_name
        JSONB metadata_json "citations, tool outputs"
        BOOLEAN is_summarized
        INTEGER sequence_number
        TIMESTAMPTZ created_at
    }

    "AIToolExecutionLog" {
        UUID id PK
        UUID session_id FK
        UUID message_event_id FK
        VARCHAR tool_name
        VARCHAR user_id
        VARCHAR user_role
        JSONB arguments_json
        JSONB result_json
        BOOLEAN success
        TEXT error_message
        SMALLINT attempt_number
        INTEGER latency_ms
        TIMESTAMPTZ created_at
    }

    "AIExecutionTrace" {
        UUID id PK
        UUID session_id FK
        VARCHAR request_id
        VARCHAR user_id
        VARCHAR user_role
        VARCHAR provider_used
        VARCHAR model_used
        BOOLEAN provider_fallback
        INTEGER prompt_tokens
        INTEGER completion_tokens
        INTEGER total_tokens
        SMALLINT tool_calls_count
        JSONB tools_used
        INTEGER latency_ms
        BOOLEAN success
        VARCHAR error_type
        TIMESTAMPTZ created_at
    }

    "AIFeedback" {
        UUID id PK
        UUID message_event_id FK
        VARCHAR user_id
        INTEGER rating "1 (Positive) | 0 (Negative)"
        TEXT comment
        TIMESTAMPTZ created_at
    }
```

---

## 5. LLM Provider Stack & Circuit Breaker

The application handles provider downtime or rate limits transparently. When client traffic arrives, the orchestrator invokes a shared provider workflow structured as follows:

```mermaid
graph TD
    A[Start Chat Request] --> B{Circuit Breaker State?}
    B -- CLOSED --> C[Execute Call via OpenRouter]
    B -- OPEN --> D[Fallback: Execute Call via Groq]
    
    C -- Success --> E[Return Response]
    C -- Failure / Timeout --> F[Increment CB Failures]
    F --> G{Failures >= 3?}
    G -- Yes --> H[Open Circuit + Route to Groq]
    G -- No --> I[Instant Failover to Groq for current turn]
    
    H --> J[Run on Groq for 60s cooldown]
    J --> K[After 60s: Attempt half-open reset call]
```

---

## 6. Security & Role-Based Authorization

### Multi-Tenant Data Isolation
The AI service secures personal records by evaluating permissions at the registry boundary:
* **The LLM never defines context IDs**: The client credentials parsed from the JWT header define the query parameters.
* **Role Permissions Map**: 
  - `STUDENT`: `get_my_gpa`, `get_my_schedule`, `get_my_transcript`, `get_my_attendance`, `faculty_bylaw_search`
  - `INSTRUCTOR`: `get_course_students`, `get_student_progress`, `get_course_attendance`, `get_my_schedule`, `faculty_bylaw_search`
  - `ADMIN`: `get_registration_statistics`, `get_all_students`, `get_course_students`, `get_student_progress`, `get_course_attendance`, `faculty_bylaw_search`

### Production Hardening
* **Swagger/API Docs Gated**: The Swagger `/docs` and ReDoc `/redoc` interfaces are disabled when `ENVIRONMENT=production`.
* **Sanitized Production Errors**: To prevent database structure leaks, all raw exceptions in production map to a generic payload: `"An unexpected error occurred. Please try again."`.
* **Database Isolation**: Split-engines read-only mapping for academic records restricts execution authorization to database `SELECT` statements.

---

## 7. API Specifications

### Client API Route Map
* `POST /api/v1/sessions` - Explicitly creates a new conversational session.
* `GET /api/v1/sessions/{session_id}` - Restores session context chronologically, returning the compressed summary alongside active short-term messages.
* `DELETE /api/v1/sessions/{session_id}` - Soft-deletes a session by marking the `is_deleted` key true in metadata.
* `POST /api/v1/chat/{session_id}` - Accept user messages, runs the agentic loop, and streams SSE chunks back.
* `POST /api/v1/feedback` - Rates assistant responses (1 for positive, 0 for negative) with optional comments.

### Internal Debug Router (Requires `X-Internal-Key`)
* `GET /internal/debug/session/{session_id}/memory` - Inspects active sliding window size, token counts, and summary text.
* `GET /internal/debug/session/{session_id}/trace?last_n=5` - Inspects performance, fallbacks, and token metrics.
* `GET /internal/debug/tools` - Lists registered tools and schemas.
* `GET /internal/debug/providers/health` - Inspects circuit breaker failure states and recent request latencies.
* `GET /internal/debug/session/{session_id}/messages?include_tools=true` - Returns full audit messages lists including intermediate tool requests.

---

## 8. Configuration & Environment Variables

| Variable | Description | Default / Example |
|---|---|---|
| `ENVIRONMENT` | Defines app behavior. Suppresses docs and errors in production. | `development` (or `production`) |
| `DATABASE_URL` | RW connection string for AI tables schema | `postgresql+asyncpg://postgres:1234@localhost:5432/edux_db` |
| `DB_READONLY_USER` | DB role username for reading academic data | `postgres` (or `ai_readonly`) |
| `DB_READONLY_PASSWORD` | DB role password | `1234` |
| `REDIS_URL` | Cache server endpoint URL | `redis://localhost:6379/0` |
| `JWT_SECRET` | HS256 JWT encryption key shared with core backend | `super-secret-key` |
| `OPENROUTER_API_KEY` | Primary LLM Provider API Key | `sk-or-...` |
| `GROQ_API_KEY` | Fallback LLM Provider API Key | `gsk_...` |
| `INTERNAL_API_KEY` | Gates `/internal/debug` API access header | `internal-debug-key` |

---

## 9. Local Development & Docker Deployment

### Prerequisites
* Python 3.12+
* PostgreSQL & Redis instances

### Standard Virtual Environment Setup
```powershell
# Clone the repository and navigate to root directory
cd ai_service

# Initialize python virtualenv
python -m venv venv
.\venv\Scripts\activate

# Install required project dependencies
pip install -r requirements.txt

# Run the local uvicorn server
uvicorn ai_service.main:app --host 127.0.0.1 --port 8002 --reload
```

### Local Dev Build via Docker Compose
Launches FastAPI app, dev PostgreSQL, and Redis containers:
```bash
docker-compose up --build
```

### Production Build via Hardened Docker Compose
Launches the service built on python-slim, running under a non-root `aiuser`, with connections pooled via PgBouncer (transaction mode) and Redis caching enabled:
```bash
docker-compose -f docker-compose.prod.yml up --build
```

---

## 10. Testing & Verification

The testing suite contains **93 unit and integration tests** checking JWT handling, database tool authorization, LLM streaming choice mocks, circuit breaker failure states, cache hits, and background memory compression.

Run tests locally:
```powershell
.\venv\Scripts\pytest.exe
```

Run test coverage report:
```powershell
.\venv\Scripts\pytest.exe --cov=ai_service
```
