from backend.rag.document_loader import load_documents
from backend.rag.text_splitter import split_documents


def main():
	documents = load_documents()

	print(f"\nLoaded {len(documents)} pages\n")

	chunks = split_documents(documents)

	print(f"Generated {len(chunks)} chunks\n")

	if not chunks:
		print("No chunks were generated.")
		return

	print("Sample Chunk:\n")
	print(chunks[0].page_content[:1000])

	print("\nMetadata:\n")
	print(chunks[0].metadata)


if __name__ == "__main__":
	main()