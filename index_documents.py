import os
import glob
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField,
    SearchField, SearchFieldDataType, VectorSearch,
    HnswAlgorithmConfiguration, VectorSearchProfile
)
from azure.core.credentials import AzureKeyCredential

from azure.identity import DefaultAzureCredential

load_dotenv()

# Clients
openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01"
)

if os.getenv("USE_MANAGED_IDENTITY") == "true":
    credential = DefaultAzureCredential()
else:
    credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))


index_client = SearchIndexClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    credential=credential
)

# Step 1: Create the search index with vector support
def create_index():
    try:
        index_client.delete_index(os.getenv("AZURE_SEARCH_INDEX"))
        print("Deleted existing index")
    except Exception:
        pass  # index didn't exist yet, that's fine

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="filename", type=SearchFieldDataType.String),
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
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
    )

    index = SearchIndex(
        name=os.getenv("AZURE_SEARCH_INDEX"),
        fields=fields,
        vector_search=vector_search
    )

    index_client.create_or_update_index(index)
    print("✅ Index created")

# Step 2: Generate embeddings and upload documents
def index_documents():
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=os.getenv("AZURE_SEARCH_INDEX"),
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
    )

    docs = []
    for i, filepath in enumerate(glob.glob("docs/*.txt")):
        with open(filepath, "r") as f:
            content = f.read().strip()

        # Generate embedding
        response = openai_client.embeddings.create(
            input=content,
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        )
        embedding = response.data[0].embedding

        docs.append({
            "id": str(i),
            "content": content,
            "filename": os.path.basename(filepath),
            "embedding": embedding
        })
        print(f"  Embedded: {filepath}")

    search_client.upload_documents(documents=docs)
    print(f"✅ Indexed {len(docs)} documents")


if __name__ == "__main__":
    create_index()
    index_documents()

