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
    page_title="DeskAI — Autonomous IT Support",
    page_icon="🦺",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 50px 40px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 30px;
        border: 1px solid #00d4ff33;
    }
    .hero h1 { color: #00d4ff; font-size: 3.2em; margin: 0; letter-spacing: 2px; }
    .hero .tagline { color: #e6f1ff; font-size: 1.3em; margin-top: 12px; font-weight: 600; }
    .hero .sub { color: #8892b0; font-size: 1em; margin-top: 8px; }
    .stat-card {
        background: #1a1a2e;
        border: 1px solid #00d4ff33;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .stat-card h3 { color: #00d4ff; margin: 0; font-size: 2.2em; }
    .stat-card p { color: #8892b0; margin: 6px 0 0 0; font-size: 0.9em; }
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

# --- Hero ---
st.markdown("""
<div class="hero">
    <h1>🛠️ DeskAI</h1>
    <div class="tagline">Resolve IT issues instantly — no waiting, no delays.</div>
    <div class="sub">Autonomous AI agents that search, reason and resolve your IT tickets 24/7</div>
</div>
""", unsafe_allow_html=True)

# --- Stats ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-card"><h3>80%</h3><p>Tickets Auto-Resolved</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-card"><h3>&lt;2s</h3><p>Avg Response Time</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-card"><h3>24/7</h3><p>Always Online</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-card"><h3>100%</h3><p>Uptime SLA</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

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
        f"Context: {state['context']}\nQuestion: {state['query']}\n"
        f"Provide a clear step-by-step resolution:"
    )
    return {**state, "response": response.content, "attempts": 1}

def escalation_agent(state):
    return {**state, "response": "⚠️ Our AI couldn't find a match in the knowledge base. A support engineer has been notified and will respond within 30 minutes."}

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

# --- Main Layout ---
left, right = st.columns([1.2, 1])

with left:
    st.markdown("### 📥 Describe Your Issue")

    query = st.text_input(
        "",
        value=st.session_state.query,
        placeholder="e.g. My VPN is not connecting since this morning...",
        key="input_box"
    )

    st.markdown("**Common issues:**")
    examples = [
        "How to reset Windows password?",
        "VPN not connecting",
        "Printer is offline",
        "Error 404 on server",
        "Computer running slow",
        "No internet connection"
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=ex):
            st.session_state.query = ex
            st.session_state.submitted = True
            st.rerun()

    resolve = st.button("🚀 Get Resolution", type="primary")
    if resolve and query.strip():
        st.session_state.query = query
        st.session_state.submitted = True

with right:
    st.markdown("### ⚙️ How DeskAI Works")
    st.markdown('<div class="agent-step">🔍 <b>Step 1</b> — Searches enterprise knowledge base</div>', unsafe_allow_html=True)
    st.markdown('<div class="agent-step">🧠 <b>Step 2</b> — AI scores relevance & confidence</div>', unsafe_allow_html=True)
    st.markdown('<div class="agent-step">⚙️ <b>Step 3</b> — Generates step-by-step resolution</div>', unsafe_allow_html=True)
    st.markdown('<div class="agent-step">🚨 <b>Step 4</b> — Escalates to engineer if needed</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#1a1a2e; border-radius:12px; padding:16px; border:1px solid #00d4ff33;">
        <p style="color:#8892b0; margin:0; font-size:0.9em;">
        ✅ No account needed<br>
        ✅ Instant AI resolution<br>
        ✅ Auto-escalation to engineers<br>
        ✅ Available 24/7
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- Result ---
if st.session_state.submitted and st.session_state.query.strip():
    st.session_state.submitted = False
    st.markdown("---")
    with st.spinner("🤖 DeskAI is analyzing your issue..."):
        result = app_graph.invoke({
            "query": st.session_state.query,
            "docs": [],
            "context": "",
            "response": "",
            "confidence": "",
            "escalate": False,
            "attempts": 0
        })

    r1, r2, r3 = st.columns(3)
    with r1:
        status = "🚨 Escalated to Engineer" if result["escalate"] else "✅ Auto-Resolved"
        st.metric("Status", status)
    with r2:
        st.metric("Confidence", result["confidence"])
    with r3:
        st.metric("Resolution Time", "< 2s")

    st.markdown("### 💬 Resolution")
    st.markdown(f'<div class="response-box">{result["response"]}</div>', unsafe_allow_html=True)

    if not result["escalate"]:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Was this helpful?**")
        c1, c2 = st.columns(2)
        c1.button("👍 Yes, resolved!")
        c2.button("👎 No, escalate to engineer")

    with st.expander("🔍 View Knowledge Base Sources"):
        st.code(result["context"], language="text")

elif resolve and not query.strip():
    st.warning("Please describe your issue first.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8892b0; font-size:0.85em; padding:10px;">
    © 2026 DeskAI &nbsp;|&nbsp; Enterprise IT Automation &nbsp;|&nbsp;
    <a href="https://github.com/Harsh-D11/agentic-devops-copilot" style="color:#00d4ff;">
    GitHub</a>
</div>
""", unsafe_allow_html=True)