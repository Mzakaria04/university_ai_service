import sys
import os
import logging

logger = logging.getLogger("ai_service.rag.adapter")

# Compute absolute path to university_rag directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "university_rag"))

# Append RAG_DIR to sys.path if not present to enable absolute imports inside university_rag
if RAG_DIR not in sys.path:
    logger.debug(f"Appending RAG directory to sys.path: {RAG_DIR}")
    sys.path.insert(0, RAG_DIR)

try:
    from app.rag import ask as _ask
except ImportError as e:
    logger.critical(f"RAG import failed: {e}. Check RAG folder path and layout.")
    raise e

def ask(question: str) -> str:
    """
    Wraps the existing university RAG ask() pipeline function.
    Provides logging context without modifying the underlying RAG implementation.
    """
    logger.info(f"Invoking RAG pipeline for query: {question}")
    try:
        response = _ask(question)
        logger.info("RAG pipeline successfully retrieved answer.")
        return response
    except Exception as e:
        logger.error(f"Error while running RAG pipeline: {e}", exc_info=True)
        raise e
