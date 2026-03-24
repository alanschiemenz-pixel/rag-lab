import gradio as gr
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from duckduckgo_search import DDGS

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────
openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01"
)

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
    return text[:1000]

# ── RAG: Retrieve from PDF index ──────────────────────────────────────────────
def retrieve_context(question: str, top_k: int = 3) -> str:
    response = openai_client.embeddings.create(
        input=question,
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    vector_query = VectorizedQuery(
        vector=response.data[0].embedding,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["content", "filename", "page"]
    )
    chunks = [
        f"[Source: {r['filename']}, page {r['page']}]\n{r['content']}"
        for r in results
    ]
    return "\n\n".join(chunks)

# ── Web Search: DuckDuckGo ────────────────────────────────────────────────────
def search_web(query: str, top_k: int = 3) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=top_k))
        if not results:
            return ""
        chunks = [
            f"[Web source: {r.get('href', '')}]\n{r.get('body', '')}"
            for r in results
        ]
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"Web search failed: {e}")
        return ""

# ── Chat ──────────────────────────────────────────────────────────────────────
def respond(message: str, history: list, use_web: bool) -> str:
    safe_message = sanitize_input(message)

    pdf_context = retrieve_context(safe_message)
    web_context = search_web(safe_message) if use_web else ""

    combined_context = ""
    if pdf_context:
        combined_context += "=== KNOWLEDGE BASE (PDFs) ===\n" + pdf_context
    if web_context:
        combined_context += "\n\n=== WEB SEARCH RESULTS ===\n" + web_context

    if use_web:
        # Looser prompt — web search is on, model can synthesise freely
        system_prompt = (
            "You are an expert AI assistant.\n"
            "You have been provided two types of context: internal knowledge base "
            "documents (PDFs) and live web search results.\n"
            "Prioritize the knowledge base for proprietary or detailed information. "
            "Use web results for current events, news, or topics not covered in "
            "the knowledge base.\n"
            "Always cite your source — either the PDF filename and page number, "
            "or the web URL.\n"
            "If neither source contains enough information, say so clearly.\n"
            "Do not make things up.\n\n"
            "CONTEXT:\n" + combined_context
        )
    else:
        # Strict prompt — web search is off, only PDF context is allowed
        if not combined_context.strip():
            return (
                "I cannot answer this — no relevant information was found "
                "in the knowledge base. Enable web search to broaden my sources."
            )
        system_prompt = (
            "You are an expert AI assistant.\n"
            "You have access ONLY to the context documents provided below.\n"
            "You MUST NOT use any knowledge from your training data.\n"
            "You MUST NOT answer any question that cannot be answered "
            "from the provided context alone.\n"
            "If the context does not contain enough information to fully "
            "answer the question, respond with exactly: "
            "'I cannot answer this from the available documents.'\n"
            "Always cite the exact source filename and page number.\n"
            "Do not speculate. Do not infer beyond what is written.\n\n"
            "CONTEXT:\n" + combined_context
        )

    messages = [{"role": "system", "content": system_prompt}]
    for entry in history:
        messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": safe_message})

    response = openai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=messages,
        temperature=0.3 if use_web else 0,
        max_tokens=600
    )
    return response.choices[0].message.content


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def handle_message(message, history, use_web):
    if not message.strip():
        return history, ""
    answer = respond(message, history, use_web)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]
    return history, ""


with gr.Blocks(title="Energy AI Assistant") as demo:
    gr.Markdown("## Energy AI Assistant")
    gr.Markdown(
        "Ask questions about your documents or the web. "
        "Click an example or type your own question below."
    )

    with gr.Row():
        web_toggle = gr.Checkbox(
            label="Enable web search",
            value=True
        )

    chatbot = gr.Chatbot(height=500)

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question...",
            label="",
            scale=9,
            lines=1
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    gr.Examples(
        examples=[
            ["What races are near Houston in the next 3 months?"],
            ["What is the hardest race you know about?"],
            ["Show me all races that are at least 50 miles or longer"],
            ["What are the best trail race shoes?"]
        ],
        inputs=[msg],
        label="Example questions"
    )

    clear_btn = gr.Button("Clear conversation")

    submit_btn.click(
        handle_message,
        inputs=[msg, chatbot, web_toggle],
        outputs=[chatbot, msg]
    )
    msg.submit(
        handle_message,
        inputs=[msg, chatbot, web_toggle],
        outputs=[chatbot, msg]
    )
    clear_btn.click(
        lambda: ([], ""),
        outputs=[chatbot, msg]
    )


if __name__ == "__main__":
    demo.launch()
    