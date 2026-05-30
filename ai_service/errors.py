class AIServiceError(Exception):
    """Base exception for all AI Service errors."""
    pass

# Authentication Errors
class AuthError(AIServiceError):
    """Base authentication error."""
    pass

class TokenExpiredError(AuthError):
    """Raised when a JWT token has expired."""
    pass

class InvalidTokenError(AuthError):
    """Raised when a JWT token is invalid, malformed, or signature verification fails."""
    pass

# Authorization Errors
class AuthorizationError(AIServiceError):
    """Base authorization error."""
    pass

class ToolAuthorizationError(AuthorizationError):
    """Raised when a user's role is not authorized to invoke a requested tool."""
    pass

class SessionOwnershipError(AuthorizationError):
    """Raised when a user attempts to access a session they do not own."""
    pass

# Tool Execution Errors
class ToolError(AIServiceError):
    """Base tool execution error."""
    pass

class ToolTimeoutError(ToolError):
    """Raised when a tool execution times out."""
    pass

class ToolArgumentError(ToolError):
    """Raised when invalid arguments are provided to a tool."""
    pass

class ToolExecutionError(ToolError):
    """Raised when a tool execution handler fails."""
    pass

# Provider Errors
class ProviderError(AIServiceError):
    """Base LLM provider error."""
    pass

class ProviderRateLimitError(ProviderError):
    """Raised when the LLM provider rate limits requests."""
    pass

class ProviderUnavailableError(ProviderError):
    """Raised when the LLM provider is down or unreachable."""
    pass

class ProviderTimeoutError(ProviderError):
    """Raised when a call to the LLM provider times out."""
    pass

class ProviderExhaustedError(ProviderError):
    """Raised when all configured LLM providers fail."""
    pass

# Memory & Session Errors
class SessionNotFoundError(AIServiceError):
    """Raised when a requested chat session is not found in the database."""
    pass

class MemoryLoadError(AIServiceError):
    """Raised when memory history cannot be loaded."""
    pass
