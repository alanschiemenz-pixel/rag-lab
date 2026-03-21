import gradio as gr
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────
openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01"
)

# Security: Managed Identity in production, API key locally
if os.getenv("USE_MANAGED_IDENTITY") == "true":
    search_credential = DefaultAzureCredential()
else:
    search_credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=search_credential
)

# ── Security: Input Sanitization ──────────────────────────────────────────────
def sanitize_input(text: str) -> str:
    forbidden = [
        "ignore previous instructions",
        "forget your instructions",
        "system:",
        "disregard all prior",
        "you are now",
        "new instruction:",
    ]
    if any(phrase in text.lower() for phrase in forbidden):
        return "[Input blocked: potential prompt injection detected]"
    return text[:1000]  # cap input length to prevent token flooding

# ── RAG: Retrieve semantically relevant context ────────────────────────────────
def retrieve_context(question: str, top_k: int = 3) -> str:
    # Embed the question using the same model used to embed documents
    response = openai_client.embeddings.create(
        input=question,
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    vector_query = VectorizedQuery(
        vector=response.data[0].embedding,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )
    try:
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["content", "filename","page"]
        )
        chunks = [
        f"[Source: {r['filename']}, page {r['page']}]\n{r['content']}"
        for r in results
        ]
    except:
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["content", "filename"]
        )
        chunks = [f"[Source: {r['filename']}]\n{r['content']}" for r in results]
    return "\n\n".join(chunks)

# ── Chat: Ground GPT in retrieved context only ────────────────────────────────
def respond(message: str, history: list) -> str:
    safe_message = sanitize_input(message)
    context = retrieve_context(safe_message)

    industry = "trail running"
    #industry = "energy sector consulting"
    system_prompt = (
        f"You are an expert AI assistant for {industry}.\n"
        "Answer questions using ONLY the context documents provided below.\n"
        "Always cite the source document filename in your answer.\n"
        "If the context does not contain enough information to answer, "
        "say clearly: I do not have information on that topic in my knowledge base.\n"
        "Do not use any outside knowledge or make things up.\n\n"
        "CONTEXT DOCUMENTS:\n" + context
    )

    messages = [{"role": "system", "content": system_prompt}]
    for entry in history:
        if isinstance(entry, dict):
            messages.append({"role": entry["role"], "content": entry["content"]})
        else:
            human_msg, assistant_msg = entry
            messages.append({"role": "user", "content": human_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": safe_message})

    response = openai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=messages,
        temperature=0.3,   # low = factual, reduces hallucination
        max_tokens=600
    )
    return response.choices[0].message.content

# ── Gradio UI ─────────────────────────────────────────────────────────────────
energy_description = (
        "Ask questions about AI applications in the energy sector. "
        "Powered by Azure OpenAI (GPT-4o-mini) and Azure AI Search (vector RAG). "
        "Answers are grounded in the knowledge base only — no hallucination."
    ),
running_description = (
        "Ask questions about trail running. "
        "Powered by Azure OpenAI (GPT-4o-mini) and Azure AI Search (vector RAG). "
        "Answers are grounded in the knowledge base only — no hallucination."
    )
energy_examples = [
        "What cost savings does predictive maintenance deliver in energy?",
        "How is AI used in HSE safety monitoring on drilling rigs?",
        "What AI techniques are used to optimize oil and gas drilling?",
        "How does AI improve grid management with renewable energy?",
        "How is AI accelerating carbon capture and storage?",
        "What AI methods are used for pipeline asset integrity?",
    ]
running_examples = [
        "What races are near Houston, TX in the next 3 months?",
        "What races are good for beginners?",
        "Give me a list of all races that have a distance of at least 50 miles"]

energy_title = "Energy AI Assistant"  # Change to "Trail Running AI Assistant" for running context
running_title = "Race calendar AI Assistant"  
demo = gr.ChatInterface(
    fn=respond,
    title=running_title,
    description=running_description,
    examples=running_examples,
)

if __name__ == "__main__":
    demo.launch(share=True)