import os
from dotenv import load_dotenv
from loguru import logger
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv()

# --- Logging Setup ---
logger.add("logs/agent.log", rotation="1 MB", retention="7 days", level="INFO")

# --- State Definition ---
class AgentState(TypedDict):
    query: str
    docs: list
    context: str
    response: str
    confidence: str
    escalate: bool
    attempts: int

# --- Setup ---
def setup_vectorstore():
    docs = [
        "To reset Windows password: Boot to safe mode, run net user admin newpass.",
        "Restart service: systemctl restart nginx or use services.msc on Windows.",
        "Common error 404: Check Apache/Nginx config and syntax carefully.",
        "VPN not connecting: Check firewall rules and restart VPN client service.",
        "Printer offline: Restart print spooler via services.msc and reconnect."
    ]
    documents = [Document(page_content=d) for d in docs]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=splits, embedding=embeddings)

vectorstore = setup_vectorstore()
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

# --- Agent 1: Retriever ---
def retriever_agent(state: AgentState) -> AgentState:
    logger.info(f"🔍 Retriever Agent: query='{state['query']}'")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(state["query"])
    context = "\n".join([doc.page_content for doc in docs])
    logger.info(f"📄 Retrieved {len(docs)} docs")
    return {**state, "docs": docs, "context": context}

# --- Agent 2: Planner (confidence check) ---
def planner_agent(state: AgentState) -> AgentState:
    logger.info("🧠 Planner Agent: checking relevance...")
    check = llm.invoke(
        f"Context: {state['context']}\nQuery: {state['query']}\n"
        f"Is this context relevant? Reply only 'HIGH' or 'LOW'."
    )
    confidence = check.content.strip().upper()
    escalate = "LOW" in confidence
    logger.info(f"📊 Confidence: {confidence} | Escalate: {escalate}")
    return {**state, "confidence": confidence, "escalate": escalate}

# --- Agent 3: Executor ---
def executor_agent(state: AgentState) -> AgentState:
    logger.info("⚙️ Executor Agent: generating response...")
    attempts = state.get("attempts", 0) + 1
    response = llm.invoke(
        f"Context: {state['context']}\nQuestion: {state['query']}\nAnswer briefly:"
    )
    logger.success(f"✅ Response generated (attempt {attempts})")
    return {**state, "response": response.content, "attempts": attempts}

# --- Agent 4: Escalation ---
def escalation_agent(state: AgentState) -> AgentState:
    logger.warning(f"⚠️ Escalation Agent: ticket escalated for '{state['query']}'")
    return {**state, "response": "⚠️ This ticket requires human review. Escalating now."}

# --- Router ---
def route(state: AgentState) -> str:
    return "escalate" if state["escalate"] else "execute"

# --- Build Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retriever_agent)
workflow.add_node("plan", planner_agent)
workflow.add_node("execute", executor_agent)
workflow.add_node("escalate", escalation_agent)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "plan")
workflow.add_conditional_edges("plan", route, {
    "execute": "execute",
    "escalate": "escalate"
})
workflow.add_edge("execute", END)
workflow.add_edge("escalate", END)

app = workflow.compile()

# --- Run ---
if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    
    test_queries = [
        "How to reset Windows password?",
        "VPN not connecting",
        "How to bake a cake?"  # Unknown - should escalate
    ]
    
    print("\n" + "="*50)
    print("🤖 MULTI-AGENT IT COPILOT - STAGE 5")
    print("="*50)
    
    for query in test_queries:
        print(f"\n📥 Ticket: {query}")
        result = app.invoke({
            "query": query,
            "docs": [],
            "context": "",
            "response": "",
            "confidence": "",
            "escalate": False,
            "attempts": 0
        })
        print(f"🤖 Response: {result['response']}")
        print(f"📊 Confidence: {result['confidence']}")
        print(f"🚨 Escalated: {result['escalate']}")
        print("-"*50)
    
    print("\n✅ Stage 5 Multi-Agent Complete! Check logs/agent.log")
