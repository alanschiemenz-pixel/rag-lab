import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

from azure.identity import DefaultAzureCredential

load_dotenv()

openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01"
)

if os.getenv("USE_MANAGED_IDENTITY") == "true":
    credential = DefaultAzureCredential()
else:
    credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=credential
)



def retrieve_context(question: str, top_k: int = 3) -> str:
    # Embed the question
    response = openai_client.embeddings.create(
        input=question,
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    query_vector = response.data[0].embedding

    # Vector search
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["content", "filename"]
    )

    context_parts = []
    for r in results:
        context_parts.append(f"[Source: {r['filename']}]\n{r['content']}")

    return "\n\n".join(context_parts)

energy_promp = """You are an AI assistant for energy sector consulting.
Answer questions using ONLY the context provided below.
If the context doesn't contain enough information, say so clearly.
Always cite which source document your answer comes from.

CONTEXT:
"""

race_prompt = """You are an AI assistant for a trail running company. Answer questions using ONLY the context provided below."""

def chat(question: str, history: list) -> str:
    context = retrieve_context(question)

        system_prompt = f"""{race_prompt}

CONTEXT:
""" + context

    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": question})

    response = openai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=messages,
        temperature=0.3,  # Lower = more factual, less hallucination
        max_tokens=500
    )

    return response.choices[0].message.content
def sanitize_input(text):
    forbidden = ['ignore previous instructions', 'forget your instructions', 'system:']
    if any(phrase in text.lower() for phrase in forbidden):
        return '[Input blocked: potential prompt injection detected]'
    return text[:1000]

def main():
    print("🤖 Energy AI Chatbot (RAG) — type 'quit' to exit\n")
    history = []

    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "exit"]:
            break
        if not question:
            continue
        
        question = sanitize_input(question)
        answer = chat(question, history)
        print(f"\nAssistant: {answer}\n")

        # Maintain conversation history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()

