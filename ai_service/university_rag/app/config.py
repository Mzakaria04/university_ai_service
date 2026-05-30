import os


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


BASE_DIR = os.path.dirname(CURRENT_DIR)


DATA_PATH = os.path.join(BASE_DIR, "data")


FAISS_PATH = os.path.join(DATA_PATH, "faiss_index")
DOCS_PATH = os.path.join(DATA_PATH, "documents_e5.pkl")


MODEL_NAME = "intfloat/multilingual-e5-base"
LLM_MODEL = "llama-3.3-70b-versatile"
