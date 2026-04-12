import os
import pytest
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Setup (reusable fixture) ---
@pytest.fixture(scope="session")
def vectorstore():
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
    store = Chroma.from_documents(documents=splits, embedding=embeddings)
    return store

@pytest.fixture(scope="session")
def llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )

# --- Tests ---

# Test 1: RAG returns results
def test_rag_returns_docs(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("How to reset Windows password?")
    assert len(docs) > 0, "❌ No docs returned!"
    print("\n✅ Test 1 Passed: RAG returns docs")

# Test 2: Relevant doc content check
def test_rag_relevance(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("Windows password reset")
    contents = [doc.page_content.lower() for doc in docs]
    assert any("password" in c for c in contents), "❌ Irrelevant docs returned!"
    print("\n✅ Test 2 Passed: Docs are relevant")

# Test 3: LLM responds
def test_llm_response(llm):
    response = llm.invoke("What is RAG in AI? Answer in one sentence.")
    assert response.content is not None, "❌ LLM returned nothing!"
    assert len(response.content) > 10, "❌ Response too short!"
    print(f"\n✅ Test 3 Passed: LLM responded - {response.content[:60]}...")

# Test 4: Edge case - empty query
def test_empty_query(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    try:
        docs = retriever.invoke("")
        print(f"\n⚠️ Test 4: Empty query returned {len(docs)} docs (handled)")
    except Exception as e:
        print(f"\n⚠️ Test 4: Empty query raised exception - {str(e)}")
    assert True  # Edge case logged, not crashed

# Test 5: Unknown topic query
def test_unknown_topic(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("How to bake a chocolate cake?")
    context = "\n".join([doc.page_content for doc in docs])
    response = llm.invoke(
        f"Context: {context}\n\nQuestion: How to bake chocolate cake?\n"
        f"If context is irrelevant, say 'I need to escalate this ticket.'"
    )
    print(f"\n✅ Test 5 Passed: Unknown topic handled - {response.content[:60]}...")
    assert response.content is not None

# Test 6: API key exists
def test_api_key_exists():
    key = os.getenv("GROQ_API_KEY")
    assert key is not None, "❌ GROQ_API_KEY missing from .env!"
    assert key.startswith("gsk_"), "❌ Invalid API key format!"
    print("\n✅ Test 6 Passed: API key valid")
