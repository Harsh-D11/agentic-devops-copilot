# 🛠️ AI Service Desk Copilot

[![LangChain](https://img.shields.io/badge/LangChain-black?style=flat&logo=langchain)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker)](https://docker.com)
[![GitHub Actions](https://img.shields.io/badge/CI--CD-2088FF?style=flat&logo=github-actions)](https://github.com/features/actions)

**Agentic AI** for enterprise IT: **80% automated ticket resolution** via RAG + tools + DevOps pipeline. LangChain/LangGraph | Python | Docker.

## 🚀 Overview
Autonomous workflow:
- **Perceive**: Docs → ChromaDB RAG
- **Reason**: LLM planning (Groq)
- **Act**: APIs (Jira/Email)
- **Deploy**: GitHub Actions → Cloud

**Live Demo**: [Post-Stage 6]

## 📋 User Stories
As **IT Agent**:
1. Ingest docs → Vector DB
2. Analyze ticket → Match solutions
3. Auto-resolve (APIs)
4. Escalate if needed
5. Log for audits

As **DevOps**:
6. CI/CD deploy

## 🗺️ Architecture

```mermaid
graph TD
    A[Ticket Input] --> B[Retriever Agent<br/>RAG Query]
    B --> C[Planner Agent<br/>LangGraph]
    C --> D{Confidence >80%?}
    D -->|Yes| E[Executor Agent<br/>API Tools]
    D -->|No| F[Human Escalate]
    E --> G[Resolution Email]
    G --> H[Monitor & Reflect]
    F --> H
    style C fill:#f0f8ff
