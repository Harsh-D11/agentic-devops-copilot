import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Dummy IT Docs ---
docs = [
    "To reset Windows password: Boot to safe mode, run net user admin newpass.",
    "Restart service: systemctl restart nginx or use services.msc on Windows.",
    "Common error 404: Check Apache/Nginx config and syntax carefully.",
    "VPN not connecting: Check firewall rules and restart VPN client service.",
    "Printer offline: Restart print spooler via services.msc and reconnect."
]
documents = [Document(page_content=d) for d in docs]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# --- Embed & Store ---
print("⏳ Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("✅ Vector DB ready!")

# --- Query + LLM ---
llm = llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "How to reset Windows password?"
relevant_docs = retriever.invoke(query)

print(f"\n🔍 Query: {query}")
print("📄 Relevant Docs Found:")
for doc in relevant_docs:
    print(f"  - {doc.page_content[:80]}...")

context = "\n".join([doc.page_content for doc in relevant_docs])
response = llm.invoke(f"Context: {context}\n\nQuestion: {query}\nAnswer briefly:")

print(f"\n🤖 Agent Answer: {response.content}")
print("\n✅ Stage 2 RAG POC Complete!")
