# DWP Policy Agent

A LangGraph-based AI agent that answers questions about Universal Credit sanctions.

## Demo

Loom Demo Link - https://www.loom.com/share/d8d26a428d8a4313bc176defa3a15d87

## Features

| Feature | Implementation |
|---------|----------------|
| Policy Q&A | RAG over DWP ADM Chapter K1 document |
| Web search | DuckDuckGo + Groq answer synthesis |
| Conversation memory | LangGraph MemorySaver |
| Tool routing | Hybrid keyword detection + model choice |

## Tech Stack

- LangGraph StateGraph
- Groq (Llama 3.1 8B)
- Chroma vector store
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- DuckDuckGo Search
- PyPDF for PDF extraction

## Setup

1. Clone the repository
2. Create `.env` file with `GROQ_API_KEY=your_key_here`
3. Place `sanctions.pdf` in root folder
4. Run `python week_7.py`

## Example Questions

**Policy (uses RAG):**
- "What is a sanction under Universal Credit?"
- "What are the different sanction levels?"
- "How long does a high level sanction last?"

**General (uses web search):**
- "What's the weather in London?"
- "Latest AI jobs in the UK"

**Memory test:**
- Ask "What is a sanction?" then "How long does it last?"

## Author

[Your Name] - Portfolio piece for DWP Digital applications
