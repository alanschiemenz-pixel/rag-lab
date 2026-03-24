import os
import glob
import fitz  # pymupdf
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField,
    SearchFieldDataType, VectorSearch,
    HnswAlgorithmConfiguration, VectorSearchProfile
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()

openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01"
)
index_client = SearchIndexClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)


def extract_and_chunk(pdf_path: str, chunk_size: int = 500, overlap: int = 100) -> list:
    chunks = []
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)

    full_text = ""
    page_map = []
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        page_map.append((len(full_text), len(full_text) + len(page_text), page_num + 1))
        full_text += page_text + " "
    doc.close()

    words = full_text.split()
    chunk_index = 0
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words).strip()

        if len(chunk_text) > 50:
            char_pos = len(" ".join(words[:i]))
            page_num = 1
            for start, end, pnum in page_map:
                if start <= char_pos < end:
                    page_num = pnum
                    break

            chunks.append({
                "text": chunk_text,
                "filename": filename,
                "page": page_num,
                "chunk_index": chunk_index
            })
            chunk_index += 1

        i += (chunk_size - overlap)

    return chunks


def create_index():
    try:
        index_client.delete_index(os.getenv("AZURE_SEARCH_INDEX"))
        print("Deleted existing index")
    except Exception:
        pass  # index didn't exist yet, that's fine

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile"
        )
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw"
        )]
    )
    index = SearchIndex(
        name=os.getenv("AZURE_SEARCH_INDEX"),
        fields=fields,
        vector_search=vector_search
    )
    index_client.create_or_update_index(index)
    print("Index created")


def index_documents():
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=os.getenv("AZURE_SEARCH_INDEX"),
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
    )

    pdf_files = glob.glob("docs/*.pdf")
    if not pdf_files:
        print("No PDF files found in docs/. Add PDFs and rerun.")
        return

    all_docs = []
    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path}")
        chunks = extract_and_chunk(pdf_path)
        print(f"    → {len(chunks)} chunks")

        for chunk in chunks:
            # Generate embedding for this chunk
            response = openai_client.embeddings.create(
                input=chunk["text"],
                model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            )
            embedding = response.data[0].embedding

            # Unique ID: filename + chunk index
            doc_id = f"{chunk['filename'].replace('.', '_')}_{chunk['chunk_index']}"

            all_docs.append({
                "id": doc_id,
                "content": chunk["text"],
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "embedding": embedding
            })

        # Upload in batches of 100 to avoid request size limits
        batch_size = 100
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i:i + batch_size]
            search_client.upload_documents(documents=batch)
            print(f"    Uploaded batch {i // batch_size + 1}")

    print(f"\nDone. Indexed {len(all_docs)} chunks from {len(pdf_files)} PDFs.")


if __name__ == "__main__":
    create_index()
    index_documents()