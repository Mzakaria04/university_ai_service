Overview
--------

Phase 2 transforms the foundation into a functional university assistant by adding all role-authorized tools, RAG integration via the existing pipeline, short-term conversational memory, session restoration, and multi-tool conversation support. The tool registry is fully populated, descriptions are optimized to act as the sole routing mechanism, and the orchestrator's bounded agentic loop is exercised with multiple sequential tool calls.

Goals
-----

*   Integrate the existing RAG pipeline via from app.rag import ask as a single tool.
    
*   Implement all student tools: get\_my\_schedule, get\_my\_transcript, get\_my\_attendance.
    
*   Implement all instructor tools: get\_course\_students, get\_student\_progress, get\_course\_attendance, get\_my\_schedule.
    
*   Implement all admin tools: get\_registration\_statistics, get\_all\_students.
    
*   Complete ROLE\_TOOL\_PERMISSIONS for all three roles.
    
*   Optimize tool descriptions for LLM self-selection (clear, mutually exclusive guidance).
    
*   Add short-term memory: load the last N messages from ai\_message\_events into the prompt context.
    
*   Enable session restoration from existing ai\_sessions + ai\_message\_events.
    
*   Improve the system prompt with role-aware capabilities and memory context blocks.
    
*   Support multi-tool conversations (up to 5 iterations).
    

Architecture Components
-----------------------

*   **RAG Pipeline Adapter (rag/pipeline\_adapter.py)** — Thin boundary wrapper around the imported ask function. Adds logging boundaries without modifying the original implementation.
    
*   **Faculty Bylaw Search Tool (tools/rag/faculty\_bylaw\_search.py)** — FacultyBylawSearchTool calls the wrapped RAG function. timeout\_seconds=15.0, max\_retries=1. Returns plain string answers as ToolResult.data.
    
*   **Student Tools (tools/student/)** — schedule.py, transcript.py, attendance.py. All use named bind parameters and hard LIMIT clauses against the existing university schema.
    
*   **Instructor Tools (tools/instructor/)** — course\_students.py, student\_progress.py, course\_attendance.py, get\_my\_schedule.py. All validate that the instructor is authorized for the requested course context.
    
*   **Admin Tools (tools/admin/)** — registration\_statistics.py, all\_students.py. Broad read-only queries scoped to the university level.
    
*   **Complete Role-Based Authorization (tools/registry.py)** — ROLE\_TOOL\_PERMISSIONS fully populated:
    
    *   STUDENT: get\_my\_gpa, get\_my\_schedule, get\_my\_transcript, get\_my\_attendance, faculty\_bylaw\_search
        
    *   INSTRUCTOR: get\_course\_students, get\_student\_progress, get\_course\_attendance, get\_my\_schedule, faculty\_bylaw\_search
        
    *   ADMIN: get\_registration\_statistics, get\_all\_students, get\_course\_students, get\_student\_progress, get\_course\_attendance, faculty\_bylaw\_search
        
*   **Tool Description Optimization** — Every tool description explicitly states when to use it and when _not_ to use it. The LLM receives the full authorized set; descriptions are the only routing mechanism.
    
*   **Short-Term Memory (memory/short\_term.py)** — ShortTermMemory.load() fetches the last 12 messages (6 turns) from ai\_message\_events, then trims to a 3000-token soft budget using tiktoken estimation.
    
*   **Memory Composer (memory/composer.py)** — Composes the MemoryContext block from short-term recent messages only. No long-term summary is included yet.
    
*   **Session Manager (sessions/manager.py)** — Enhanced load() validates ownership and returns the session plus metadata. Supports restoration of active conversations.
    
*   **Prompt Builder (orchestration/prompt\_builder.py)** — Role-aware system prompt now includes:
    
    *   Institution name and user role
        
    *   full\_name personalization
        
    *   Summarized authorized tool capabilities
        
    *   Memory context block (recent conversation history)
        
    *   Behavioral guidelines (language, data privacy, failure handling)
        
*   **Conversation Orchestrator (orchestration/conversation\_orchestrator.py)** — The bounded agentic loop (max 5 iterations) is now fully utilized. The LLM may call multiple tools sequentially within a single turn (e.g., schedule then attendance).
    
*   **Message Persistence (persistence/message\_writer.py)** — Now handles tool\_call and tool\_result roles in ai\_message\_events. Populates tool\_call\_id, tool\_name, and metadata\_json for tool events.
    
*   **Feedback API (api/v1/feedback.py)** — POST /api/v1/feedback accepts { "message\_event\_id": "uuid", "rating": 1|0, "comment": "..." } and writes to the existing AIFeedback table.
    

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
│       ├── sessions.py
│       └── feedback.py
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
├── memory/
│   ├── short_term.py
│   └── composer.py
├── tools/
│   ├── base.py
│   ├── registry.py
│   ├── executor.py
│   ├── student/
│   │   ├── gpa.py
│   │   ├── schedule.py
│   │   ├── transcript.py
│   │   └── attendance.py
│   ├── instructor/
│   │   ├── course_students.py
│   │   ├── student_progress.py
│   │   ├── course_attendance.py
│   │   └── get_my_schedule.py
│   ├── admin/
│   │   ├── registration_statistics.py
│   │   └── all_students.py
│   └── rag/
│       └── faculty_bylaw_search.py
├── rag/
│   └── pipeline_adapter.py
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
    │   ├── test_tools/
    │   ├── test_memory/
    │   └── test_orchestrator.py
    ├── integration/
    │   ├── test_conversation_flow.py
    │   └── test_tool_execution.py
    └── fixtures/
        ├── mock_provider.py
        └── mock_db.py
```

Database Usage
--------------

*   **ai\_sessions** — Loaded for restoration. Ownership validated on every request. last\_active\_at updated after each turn.
    
*   **ai\_message\_events** — Now stores role values: user, assistant, tool. message\_type values: text, tool\_call, tool\_result. For tool events, tool\_call\_id, tool\_name, and metadata\_json are populated. sequence\_number guarantees ordering. is\_summarized remains false in this phase.
    

APIs
----

*   POST /api/v1/chat/{session\_id}**Enhancements:** Loads short-term memory window into the prompt. Supports multi-tool agentic loops. Persists tool\_call and tool\_result events.
    
*   GET /api/v1/sessions/{session\_id}**Response:** { "session\_id": "...", "messages": \[...\], "message\_count": N }Restores the full recent message history from ai\_message\_events ordered by sequence\_number.
    
*   DELETE /api/v1/sessions/{session\_id}Soft-deletes the session (or marks it archived per existing schema behavior).
    
*   POST /api/v1/feedback**Body:** { "message\_event\_id": "uuid", "rating": 1, "comment": "..." }Writes feedback linked to a specific ai\_message\_events row in the existing AIFeedback table.
    

Services
--------

| Service                    | Responsibility                                                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FacultyBylawSearchTool`   | Wraps `from app.rag import ask`. Accepts a natural-language question, returns the RAG pipeline's plain string answer.                             |
| `ShortTermMemory`          | Loads the last 12 messages from `ai_message_events`, trims to ~3000 tokens, and returns them as `Message` objects.                                |
| `MemoryComposer`           | Assembles the memory context block for the prompt builder from short-term recent messages only.                                                   |
| `PromptBuilder`            | Constructs the role-aware system prompt including tool capability summaries and the memory context block.                                         |
| `ConversationOrchestrator` | Runs the full agentic loop: LLM may call one or more tools iteratively; each result is appended before the next LLM call. Capped at 5 iterations. |
| `MessagePersistence`       | Persists `user`, `assistant`, `tool_call`, and `tool_result` events. Updates `ai_sessions.last_active_at`.                                        |


Tools
-----

| Tool                          | Domain   | Authorized Roles        | Description                                                                                                   |
| ----------------------------- | -------- | ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| `get_my_gpa`                  | DATABASE | `STUDENT`               | Cumulative GPA, credits, completed courses.                                                                   |
| `get_my_schedule`             | DATABASE | `STUDENT`, `INSTRUCTOR` | Current semester schedule with times and locations.                                                           |
| `get_my_transcript`           | DATABASE | `STUDENT`               | Final grades and credit hours per course, ordered by semester.                                                |
| `get_my_attendance`           | DATABASE | `STUDENT`               | Attendance record per course with absence percentages.                                                        |
| `faculty_bylaw_search`        | RAG      | All                     | Answers policy/regulation questions via the existing RAG pipeline. Explicitly excludes personal data queries. |
| `get_course_students`         | DATABASE | `INSTRUCTOR`, `ADMIN`   | Roster for a course the instructor teaches.                                                                   |
| `get_student_progress`        | DATABASE | `INSTRUCTOR`, `ADMIN`   | Academic progress metrics for a specific student.                                                             |
| `get_course_attendance`       | DATABASE | `INSTRUCTOR`, `ADMIN`   | Attendance summary for a specific course section.                                                             |
| `get_registration_statistics` | DATABASE | `ADMIN`                 | University-wide enrollment and registration metrics.                                                          |
| `get_all_students`            | DATABASE | `ADMIN`                 | Paginated list of all students (with hard limit).                                                             |


Dependencies
------------

Add to Phase 1 dependencies:

```toml
tiktoken = "*" # Token estimation for short-term memory windowing
```

Environment Variables
---------------------

Same as Phase 1. No additional variables required. The RAG pipeline is imported directly as Python code; no HTTP endpoint or API key is needed for it.

Deliverables
------------

1.  All student, instructor, and admin tools implemented and registered.
    
2.  faculty\_bylaw\_search tool calling the existing RAG pipeline via from app.rag import ask.
    
3.  Short-term memory loading recent messages into every prompt.
    
4.  Session restoration endpoint returning full conversation history.
    
5.  Multi-tool conversation support demonstrated (e.g., "Show me my schedule and attendance" triggers two sequential tool calls).
    
6.  Feedback submission endpoint wired to the existing AIFeedback table.
    

Testing Strategy
----------------

*   **Unit:** Test each tool handler in isolation with an async test database session. Verify SQL queries use named parameters and respect LIMIT clauses.
    
*   **Unit:** Test ShortTermMemory trimming logic. Verify it drops older messages when the token budget is exceeded.
    
*   **Integration:** Test a multi-tool conversation turn. Mock OpenRouter to first request get\_my\_schedule, then get\_my\_attendance, then produce a final answer. Assert both tool results are persisted and the SSE stream contains the final text.
    
*   **Integration:** Test RAG tool. Mock app.rag.ask to return a bylaw excerpt; assert the LLM receives it as a tool role message.
    
*   **Authorization:** For each role, attempt to call every tool and assert 403 for unauthorized tools.
    

Success Criteria
----------------

*   \[ \] A STUDENT can ask about GPA, schedule, transcript, and attendance in the same conversation.
    
*   \[ \] An INSTRUCTOR can query their course rosters and student progress.
    
*   \[ \] An ADMIN can query registration statistics and all students.
    
*   \[ \] faculty\_bylaw\_search returns RAG-generated answers for policy questions.
    
*   \[ \] The LLM correctly selects tools based on descriptions alone—no routing layer needed.
    
*   \[ \] Short-term memory provides conversational context across multiple turns.
    
*   \[ \] Session restoration returns the exact recent message history from ai\_message\_events.
    
*   \[ \] Multi-tool turns execute sequentially within the 5-iteration cap and stream a single final response.
    

Anti-Overengineering Notes
--------------------------

*   **No long-term memory summarization.** is\_summarized and message\_type=SUMMARY are not used. Compression is deferred to Phase 3.
    
*   **No Redis or caching.** All tool calls continue to hit PostgreSQL directly.
    
*   **No observability stack.** No structured logging, metrics, tracing, or execution trace tables are written to yet.
    
*   **No provider failover.** OpenRouter remains the sole provider.
    
*   **No internal debug endpoints.** No /internal/\* routes.
    
*   **No background tasks.** Memory loading and persistence remain synchronous within the request cycle.
    
*   **Tool descriptions are the router.** Do not add an intent classification model or routing prompt.