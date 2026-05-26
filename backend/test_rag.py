from backend.rag.document_loader import load_documents
from backend.rag.text_splitter import split_documents
from backend.rag.vector_store import create_vector_store
from backend.rag.rag_chain import generate_answer


def main():
    documents = load_documents()

    chunks = split_documents(documents)

    if not chunks:
        print("No chunks were generated.")
        return

    vectorstore = create_vector_store(chunks)

    query = "How does the pothole detection system work?"

    results = vectorstore.similarity_search(query, k=3)

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    answer = generate_answer(context, query)

    print("\nQUESTION:\n")
    print(query)

    print("\nANSWER:\n")
    print(answer)

    print("\nSOURCES:\n")

    for i, doc in enumerate(results, start=1):
        print(f"\nSource {i}")
        print(doc.metadata.get("source_file"))
        print(f"Page: {doc.metadata.get('page')}")


if __name__ == "__main__":
    main()