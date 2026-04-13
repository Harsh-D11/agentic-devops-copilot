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

st.set_page_config(
    page_title="Desk.AI — Autonomous IT Support",
    page_icon="🛠️",
    layout="centered"
)

st.markdown("""
<style>
    /* Base */
    html, body, .stApp { background-color: #0d1117 !important; }
    .block-container { padding-top: 2rem; max-width: 780px; }

    /* Navbar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 0 32px 0;
        border-bottom: 1px solid #ffffff11;
        margin-bottom: 48px;
    }
    .nav-logo { color: white; font-size: 1.3em; font-weight: 700; }
    .nav-logo span { color: #00e5a0; }
    .nav-links { color: #8892b0; font-size: 0.9em; }
    .nav-links a { color: #8892b0; text-decoration: none; margin-left: 24px; }
    .nav-links a:hover { color: #00e5a0; }

    /* Badge */
    .badge {
        display: inline-flex;
        align-items: center;
        background: #00e5a011;
        border: 1px solid #00e5a044;
        color: #00e5a0;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85em;
        margin-bottom: 24px;
    }
    .badge-dot {
        width: 8px; height: 8px;
        background: #00e5a0;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Hero */
    .hero-title {
        font-size: 4em;
        font-weight: 800;
        color: white;
        margin: 0;
        line-height: 1.1;
        text-align: center;
    }
    .hero-title span { color: #00e5a0; }
    .hero-sub {
        color: #8892b0;
        font-size: 1.15em;
        text-align: center;
        margin: 16px auto 0 auto;
        max-width: 560px;
        line-height: 1.6;
    }

    /* Divider */
    .accent-line {
        width: 60px;
        height: 3px;
        background: #f0a500;
        margin: 36px auto;
        border-radius: 2px;
    }

    /* Steps */
    .steps-row {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 48px;
    }
    .step-item {
        flex: 1;
        text-align: center;
    }
    .step-icon {
        width: 72px; height: 72px;
        background: #161b22;
        border: 1px solid #ffffff11;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 16px auto;
        font-size: 1.6em;
    }
    .step-text {
        color: #ccd6f6;
        font-size: 0.9em;
        font-weight: 600;
        line-height: 1.4;
    }

    /* Input Area */
    .input-card {
        background: #161b22;
        border: 1px solid #ffffff11;
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
    }
    .input-label {
        color: #8892b0;
        font-size: 0.85em;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stTextInput > div > div > input {
        background: #0d1117 !important;
        color: #e6f1ff !important;
        border: 1px solid #ffffff22 !important;
        border-radius: 8px !important;
        font-size: 1em !important;
        padding: 14px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00e5a0 !important;
        box-shadow: 0 0 0 2px #00e5a022 !important;
    }

    /* Quick buttons */
    .quick-label {
        color: #8892b0;
        font-size: 0.8em;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 20px 0 10px 0;
    }

    /* Resolve button */
    .stButton > button {
        background: #00e5a0 !important;
        color: #0d1117 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 32px !important;
        font-size: 1em !important;
        font-weight: 700 !important;
        width: 100% !important;
        margin-top: 16px !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: #00c988 !important;
        transform: translateY(-1px) !important;
    }

    /* Result card */
    .result-card {
        background: #161b22;
        border: 1px solid #00e5a033;
        border-radius: 16px;
        padding: 28px;
        margin-top: 24px;
    }
    .result-title {
        color: #00e5a0;
        font-size: 0.8em;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }
    .result-text {
        color: #e6f1ff;
        font-size: 1.05em;
        line-height: 1.7;
    }
    .escalate-card {
        background: #161b22;
        border: 1px solid #ff4d6d33;
        border-radius: 16px;
        padding: 28px;
        margin-top: 24px;
    }
    .escalate-title { color: #ff4d6d; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; }

    /* Metrics */
    .metric-row {
        display: flex;
        gap: 16px;
        margin: 20px 0;
    }
    .metric-box {
        flex: 1;
        background: #0d1117;
        border: 1px solid #ffffff11;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-val { color: #00e5a0; font-size: 1.4em; font-weight: 700; }
    .metric-lbl { color: #8892b0; font-size: 0.8em; margin-top: 4px; }

    /* Footer */
    .footer {
        text-align: center;
        color: #8892b033;
        font-size: 0.8em;
        padding: 32px 0 16px 0;
        border-top: 1px solid #ffffff08;
        margin-top: 48px;
    }
    .footer a { color: #00e5a066; text-decoration: none; }

    /* Hide streamlit UI */
    #MainMenu, header, footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# --- Navbar ---
st.markdown("""
<div class="navbar">
    <div class="nav-logo">Desk<span>.AI</span></div>
    <div class="nav-links">
        <a href="#">Home</a>
        <a href="#">How it works</a>
        <a href="https://github.com/Harsh-D11/agentic-devops-copilot">GitHub</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Badge ---
st.markdown("""
<div style="text-align:center;">
    <div class="badge" style="display:inline-flex;">
        <div class="badge-dot"></div>
        Powered by Groq &nbsp;·&nbsp; LLaMA 3.3 70B
    </div>
</div>
""", unsafe_allow_html=True)

# --- Hero Title ---
st.markdown("""
<div style="text-align:center; margin-bottom: 8px;">
    <div class="hero-title">Desk<span>.AI</span></div>
    <div class="hero-sub">
        Instant IT ticket resolution. No waiting, no queues.<br>
        AI agents search, reason and resolve your issues in seconds.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Accent Line ---
st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

# --- Steps ---
st.markdown("""
<div class="steps-row">
    <div class="step-item">
        <div class="step-icon">📋</div>
        <div class="step-text">Describe your IT issue in plain English</div>
    </div>
    <div class="step-item">
        <div class="step-icon">🧠</div>
        <div class="step-text">AI agents search & reason over knowledge base</div>
    </div>
    <div class="step-item">
        <div class="step-icon">✅</div>
        <div class="step-text">Get instant resolution or escalate to engineer</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Setup ---
@st.cache_resource
def setup():
    docs = [
        "To reset Windows password: Boot to safe mode, run net user admin newpass.",
        "Restart service: systemctl restart nginx or use services.msc on Windows.",
        "Common error 404: Check Apache/Nginx config and syntax carefully.",
        "VPN not connecting: Check firewall rules and restart VPN client service.",
        "Printer offline: Restart print spooler via services.msc and reconnect.",
        "Blue screen of death: Run sfc /scannow in admin CMD to fix system files.",
        "Slow computer: Open Task Manager, end high CPU processes, clear temp files.",
        "No internet: Reset network adapter via Device Manager or run netsh winsock reset."
    ]
    documents = [Document(page_content=d) for d in docs]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )
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
        f"Context: {state['context']}\nQuestion: {state['query']}\n"
        f"Provide a clear step-by-step resolution:"
    )
    return {**state, "response": response.content, "attempts": 1}

def escalation_agent(state):
    return {**state, "response": "A support engineer has been notified and will respond within 30 minutes."}

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

# --- Session State ---
if "query" not in st.session_state:
    st.session_state.query = ""
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# --- Input Card ---
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="input-label">Describe your issue</div>', unsafe_allow_html=True)

query = st.text_input(
    "",
    value=st.session_state.query,
    placeholder="e.g. My VPN is not connecting since this morning...",
    key="input_box",
    label_visibility="collapsed"
)
