import os
from dotenv import load_dotenv

load_dotenv()  # must run before LangChain imports — sets AZURESEARCH_FIELDS_CONTENT_VECTOR

import gradio as gr
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import AzureSearch
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from duckduckgo_search import DDGS

# ── LangChain clients ──────────────────────────────────────────────────────────
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    api_version="2024-02-01"
)

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version="2024-02-01",
    max_tokens=600
)

search_key = (
    None if os.getenv("USE_MANAGED_IDENTITY") == "true"
    else os.getenv("AZURE_SEARCH_KEY")
)

vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=search_key,
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    embedding_function=embeddings.embed_query
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
    docs = vector_store.similarity_search(question, k=top_k)
    chunks = [
        f"[Source: {doc.metadata.get('filename', 'unknown')}, page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
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

    messages = [SystemMessage(content=system_prompt)]
    for entry in history:
        if entry["role"] == "user":
            messages.append(HumanMessage(content=entry["content"]))
        else:
            messages.append(AIMessage(content=entry["content"]))
    messages.append(HumanMessage(content=safe_message))

    response = llm.invoke(messages, temperature=0.3 if use_web else 0)
    return response.content


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
            ["How is AI being utilized to improve energy transmission infrastructure and support the clean energy transition?"],
            ["What are the key risks and challenges associated with deploying generative AI in the energy and materials sectors, and how can they be mitigated?"],
            ["How is the growth of sustainable financing, such as green bonds, influencing investments in energy transition technologies?"],
            ["What are the power and compute demands of large AI models, and how are major companies scaling their data centers to meet these needs?"]
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
    demo.launch(share=True)
