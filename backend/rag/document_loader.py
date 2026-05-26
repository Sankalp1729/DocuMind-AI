from backend.rag.multimodal_loader import load_supported_documents


def load_documents(data_folder=None):
    return load_supported_documents(data_folder)