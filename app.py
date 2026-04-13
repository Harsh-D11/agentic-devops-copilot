import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="AI Service Desk Copilot",
    page_icon="🛠️",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 40px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 30px;
        border: 1px solid #00d4ff33;
    }
    .hero h1 { color: #00d4ff; font-size: 2.8em; margin: 0; }
    .hero p { color: #8892b0; font-size: 1.1em; margin-top: 10px; }
    .stat-card {
        background: #1a1a2e;
        border: 1px solid #00d4ff33;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stat-card h3 { color: #00d4ff; margin: 0; font-size: 2em; }
    .stat-card p { color: #8892b0; margin: 5px 0 0 0; }
    .agent-step {
        background: #1a1a2e;
        border-left: 3px solid #00d4ff;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #ccd6f6;
    }
    .response-box {
        background: #1a1a2e;
        border: 1px solid #00d4ff55;
        border-radius: 12px;
        padding: 24px;
        color: #e6f1ff;
        font-size: 1.1em;
        line-height: 1.6;
    }
    .badge-high {
        background: #00d4ff22;
        color: #00d4ff;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid #00d4ff;
        font-weight: bold;
    }
    .badge-low {
        background: #ff006622;
        color: #ff4d6d;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid #ff4d6d;
        font-weight: bold;
    }
    .stTextInput > div > div > input {
        background: #1a1a2e !important;
        color: #e6f1ff !important;
        border: 1px solid #00d4ff55 !important;
        border-radius: 8px !important;
        font-size: 1em !important;
        padding: 12px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0070f3) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        font-size: 1em !important;
        font-weight: bold !important;
        width: 100% !important;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero">
    <h1> AI Service Desk Copilot</h1>
    <p>Autonomous IT ticket resolution powered by RAG + LangGraph + Groq LLaMA</p>
    <p style="color:#00d4ff; font-size:0.9em;">Built by Harsh-D11 | Agentic AI Portfolio Project</p>
</div>
""", unsafe_allow_html=True)

# --- Stats Row ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-card"><h3>4</h3><p>AI Agents</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-card"><h3>RAG</h3><p>Vector Search</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-card"><h3>LLaMA</h3><p>Groq Powered</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-card"><h3>Auto</h3><p>CI/CD Pipeline</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Setup ---
@st.cache_resource
def setup():
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
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
    return vectorstore, llm

vectorstore, llm = setup()

# --- Agent State ---
class AgentState(TypedDict):
    query: str
    docs: list
    context: str
    response: str
    confidence: str
    escalate: bool
    attempts: int

# --- Agents ---
def retriever_agent(state):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(state["query"])
    context = "\n".join([doc.page_content for doc in docs])
    return {**state, "docs": docs, "context": context}

def planner_agent(state):
    check = llm.invoke(
        f"Context: {state['context']}\nQuery: {state['query']}\n"
        f"Is this context relevant to IT support? Reply only 'HIGH' or 'LOW'."
    )
    confidence = check.content.strip().upper()
    escalate = "LOW" in confidence
    return {**state, "confidence": confidence, "escalate": escalate}

def executor_agent(state):
    response = llm.invoke(
        f"Context: {state['context']}\nQuestion: {state['query']}\nAnswer briefly:"
    )
    return {**state, "response": response.content, "attempts": 1}

def escalation_agent(state):
    return {**state, "response": "⚠️ This ticket requires human review. Escalating to support team."}

def route(state):
    return "escalate" if state["escalate"] else "execute"

@st.cache_resource
def build_graph():
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
    return workflow.compile()

app_graph = build_graph()

# --- Main Layout ---
left, right = st.columns([1.2, 1])

with left:
    st.markdown("### 📥 Submit IT Ticket")
    query = st.text_input("", placeholder="e.g. My VPN is not connecting...")

    examples = ["How to reset Windows password?", "VPN not connecting", "Printer is offline", "Error 404 on server"]
    st.markdown("**Quick examples:**")
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=ex):
            query = ex

    resolve = st.button("🚀 Resolve Ticket", type="primary")

with right:
    st.markdown("### 🤖 Agent Pipeline")
    st.markdown('<div class="agent-step">🔍 <b>Retriever Agent</b> — Searches knowledge base</div>', unsafe_allow_html=True)
    st.markdown('<div class="agent-step">🧠 <b>Planner Agent</b> — Scores confidence</div>', unsafe_allow_html=True)
    st.markdown('<div class="agent-step">⚙️ <b>Executor Agent</b> — Generates resolution</div>', unsafe_allow_html=True)
    st.markdown('<div class="agent-step">🚨 <b>Escalation Agent</b> — Routes to human</div>', unsafe_allow_html=True)

# --- Result ---
if resolve and query.strip():
    st.markdown("---")
    with st.spinner("🤖 Agents processing your ticket..."):
        result = app_graph.invoke({
            "query": query,
            "docs": [],
            "context": "",
            "response": "",
            "confidence": "",
            "escalate": False,
            "attempts": 0
        })

    r1, r2, r3 = st.columns(3)
    with r1:
        status = "🚨 Escalated" if result["escalate"] else "✅ Auto-Resolved"
        st.metric("Status", status)
    with r2:
        st.metric("Confidence", result["confidence"])
    with r3:
        st.metric("Escalated", "Yes" if result["escalate"] else "No")

    st.markdown("### 💬 Agent Response")
    st.markdown(f'<div class="response-box">{result["response"]}</div>', unsafe_allow_html=True)

    with st.expander("📄 View Retrieved Knowledge Base Context"):
        st.code(result["context"], language="text")

elif resolve and query.strip() == "":
    st.warning("⚠️ Please enter a ticket description first!")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8892b0; font-size:0.85em;">
     AI Service Desk Copilot |  
    <a href="https://github.com/Harsh-D11/agentic-devops-copilot" style="color:#00d4ff;">GitHub</a> | 
    Built by Harsh-D11 | April 2026
</div>
""", unsafe_allow_html=True)
