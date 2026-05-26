from backend.rag.document_loader import load_documents
from backend.rag.text_splitter import split_documents
from backend.rag.vector_store import create_vector_store


def main():
    documents = load_documents()

    chunks = split_documents(documents)

    print(f"\nLoaded {len(chunks)} chunks\n")

    if not chunks:
        print("No chunks were generated.")
        return

    vectorstore = create_vector_store(chunks)

    print("Vector store created successfully!\n")

    query = "How does the pothole detection system work?"

    results = vectorstore.similarity_search(query, k=3)

    print("Top Retrieval Results:\n")

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("-" * 50)
        print(result.page_content[:500])
        print("\nMetadata:")
        print(result.metadata)


if __name__ == "__main__":
    main()