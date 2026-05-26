import logging
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings():
    logger.info("Initializing embeddings model: %s", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_vector_store(persist_directory=None):
    persist_path = Path(persist_directory) if persist_directory is not None else VECTOR_STORE_DIR

    if not persist_path.exists():
        return None

    index_files = [persist_path / "index.faiss", persist_path / "index.pkl"]
    if not all(path.exists() for path in index_files):
        return None

    logger.info("Loading FAISS index from %s", persist_path)

    return FAISS.load_local(
        str(persist_path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def save_vector_store(vectorstore, persist_directory=None):
    persist_path = Path(persist_directory) if persist_directory is not None else VECTOR_STORE_DIR
    persist_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving FAISS index to %s", persist_path)
    vectorstore.save_local(str(persist_path))
    return vectorstore


def create_vector_store(chunks, persist_directory=None):
    persist_path = Path(persist_directory) if persist_directory is not None else VECTOR_STORE_DIR

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=get_embeddings()
    )

    return save_vector_store(vectorstore, persist_path)