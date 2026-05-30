import json
from typing import Any

def format_sse_chunk(content: str) -> str:
    """
    Formats a raw text chunk into a standard SSE choice delta payload.
    Example output:
    data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n
    """
    payload = {
        "choices": [
            {
                "delta": {
                    "content": content
                }
            }
        ]
    }
    return f"data: {json.dumps(payload)}\n\n"

def format_sse_done() -> str:
    """
    Formats the termination marker for standard SSE.
    Output:
    data: [DONE]\n\n
    """
    return "data: [DONE]\n\n"

def format_sse_error(error_message: str) -> str:
    """
    Formats an error message into a structured SSE data payload.
    Output:
    data: {"error": {"message": "Error message", "type": "orchestration_error"}}\n\n
    """
    payload = {
        "error": {
            "message": error_message,
            "type": "orchestration_error"
        }
    }
    return f"data: {json.dumps(payload)}\n\n"
