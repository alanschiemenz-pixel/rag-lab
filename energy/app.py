import os
from dotenv import load_dotenv

load_dotenv()  # must run before LangChain imports — sets AZURESEARCH_FIELDS_CONTENT_VECTOR

import gradio as gr
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import AzureSearch
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

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


# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal PDF knowledge base for energy topics."""
    docs = vector_store.similarity_search(query, k=3)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(
        f"[Source: {doc.metadata.get('filename', 'unknown')}, page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )

@tool
def search_web(query: str) -> str:
    """Search the web via DuckDuckGo for current information."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No web results found."
        return "\n\n".join(
            f"[Web source: {r.get('href', '')}]\n{r.get('body', '')}"
            for r in results
        )
    except Exception as e:
        return f"Web search failed: {e}"


# ── Agents ─────────────────────────────────────────────────────────────────────
PROMPT_STRICT = (
    "You are an expert AI assistant. "
    "You MUST only use the search_knowledge_base tool to answer questions. "
    "Do NOT use any training knowledge. "
    "If the knowledge base lacks enough information, say: "
    "'I cannot answer this from the available documents.' "
    "Always cite the exact source filename and page number."
)

PROMPT_WEB = (
    "You are an expert AI assistant. "
    "Use search_knowledge_base for proprietary or detailed information, "
    "and search_web for current events or topics not in the knowledge base. "
    "Always cite your source — PDF filename and page, or web URL. "
    "If neither source has enough information, say so clearly."
)

agent_strict = create_react_agent(
    llm.bind(temperature=0), tools=[search_knowledge_base], prompt=PROMPT_STRICT
)
agent_web = create_react_agent(
    llm.bind(temperature=0.3), tools=[search_knowledge_base, search_web], prompt=PROMPT_WEB
)

# ── Chat ───────────────────────────────────────────────────────────────────────
def respond(message: str, history: list, use_web: bool) -> str:
    safe_message = sanitize_input(message)

    messages = []
    for entry in history:
        if entry["role"] == "user":
            messages.append(HumanMessage(content=entry["content"]))
        else:
            messages.append(AIMessage(content=entry["content"]))
    messages.append(HumanMessage(content=safe_message))

    agent = agent_web if use_web else agent_strict
    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content
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
