import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal
import operator
import os
from langgraph.checkpoint.memory import MemorySaver
from ddgs import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

@tool
def web_search(query: str) -> str:
    """Search the web for current information and provide a direct answer."""
    print(f"🌐 Web search: {query[:80]}...")

    try:
        time.sleep(0.5)
        with DDGS() as ddg:
            results = list(ddg.text(
                query,
                max_results=5,
                backend="auto",
                region="wt-wt"
            ))

            if not results:
                return "No results found. Please try a different search term."
            
            search_context = []
            for i, r in enumerate(results[:5], 1):
                title = r.get('title', 'No title')
                body = r.get('body', '')
                search_context.append(f"Source {i}: {title}\n{body}")
            
            context = "\n\n".join(search_context)
            
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
            
            prompt = f"""Based on these web search results, answer the user's question concisely.

SEARCH RESULTS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Answer directly based ONLY on the search results above
- Be specific and factual
- Keep answer to 2-3 sentences
- If the results don't contain the answer, say "I couldn't find current information about that"

ANSWER:"""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        
    except ImportError:
        return "Search library not available."
    except Exception as e:
        return f"Search temporarily unavailable: {str(e)[:100]}"
    

retriever = None
RAG_AVAILABLE = False

try:
    reader = PdfReader("sanctions.pdf")
    doc_text =  ""
    for page in reader.pages:
        doc_text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(doc_text)
    
    # Create embeddings
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    docs = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = Chroma.from_documents(docs, embed_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    RAG_AVAILABLE = True
    
except FileNotFoundError:
    print("⚠️ Sanctions.pdf . RAG tool will be unavailable.")
    print("   Place the PDF in the same folder and restart.")
except Exception as e:
    print(f"⚠️ RAG setup error: {e}")
    print("   RAG tool will be unavailable.")

@tool
def ask_document(question: str) -> str:
    """Answer questions about Universal Credit sanctions using the official DWP ADM Chapter K1 document."""
    
    if not RAG_AVAILABLE or retriever is None:
        return "Document search not available."
    
    try:
        results = retriever.invoke(question)
        
        if not results:
            return "NO_INFO: The document does not contain information about this."
        
        context = "\n\n".join([doc.page_content for doc in results[:3]])
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        
        prompt = f"""Based ONLY on this DWP document, answer the question concisely.

DOCUMENT:
{context}

QUESTION: {question}

RULES:
- Answer in 2-3 sentences maximum
- Be specific and complete
- If the answer is not in the document, say exactly: "NO_INFO: The document does not contain this information."

ANSWER:"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
        
    except Exception as e:
        return f"ERROR: {str(e)}"



if RAG_AVAILABLE:
    doc_result = ask_document.invoke({"question": "What is a sanctionable failure?"})

else:
    print("   ask_document: NOT AVAILABLE (PDF not found)")


available_tools = [web_search]
if RAG_AVAILABLE:
    available_tools.append(ask_document)

tools = available_tools
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)



class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    tool_used: bool

print(f"✅ Stage 5 complete: State defined with messages, llm_calls, tool_used")


SYSTEM_PROMPT = """You are a helpful assistant with tools.

CRITICAL INSTRUCTION - CHAIN OF THOUGHT:
Before answering any question, you MUST think step by step.

Format your response as:
Step 1: [What you need to do first]
Step 2: [What you need to do next]
...
Final Answer: [Your answer]

IMPORTANT: After providing the Final Answer, stop. Do not add extra text.

Now follow this format for every question."""

def call_model(state: AgentState):
    time.sleep(0.5)
    
    response = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,  # ← INCREMENTS! CRITICAL!
        "tool_used": len(response.tool_calls) > 0 if hasattr(response, "tool_calls") else False
    }

def call_tool(state: AgentState):
    results = []
    last_message = state["messages"][-1]
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]  # CRITICAL!
        
        if tool_name in tools_by_name:
            try:
                tool_result = tools_by_name[tool_name].invoke(tool_args)
                results.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                    name=tool_name
                ))
            except Exception as e:
                results.append(ToolMessage(
                    content=f"Tool error: {e}",
                    tool_call_id=tool_call_id,
                    name=tool_name
                ))
    
    return {
        "messages": results,
        "tool_used": True
    } 
def should_continue(state: AgentState) -> Literal["tool_node", END]:
    last_message = state["messages"][-1]
    llm_calls = state.get("llm_calls", 0)  
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    
    if last_message.content:
        return END
    
    if llm_calls >= 5:  
        return END
    
    # Fallback
    return END





graph = StateGraph(AgentState)
graph.add_node("llm_call", call_model)
graph.add_node("tool_node", call_tool)
graph.add_edge(START, "llm_call" )
graph.add_conditional_edges("llm_call", should_continue, {"tool_node": "tool_node", END: END})
graph.add_edge("tool_node", "llm_call")

memory = MemorySaver()
agent = graph.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "main_conversation"}}

state = {
    "messages": [SystemMessage(content=SYSTEM_PROMPT)],
    "llm_calls": 0,
    "tool_used": False
}




print("\n" + "="*50)
print("💼 DWP POLICY ASSISTANT - CHAT MODE")
print("="*50)
print("Type 'quit' to exit")
print("Type 'clear' to reset conversation")
print("-" * 40)
print("📋 Try asking (tests each tool):")
print()
print("  📄 [ask_document tool] Universal Credit questions:")
print("     • What is a sanction under Universal Credit?")
print("     • What are the different sanction levels?")
print("     • What happens if I miss a work search appointment?")
print("     • What is a compliance condition?")
print("     • How long does a high level sanction last?")
print()
print("  🌐 [web_search tool] General questions:")
print("     • What are the latest AI jobs in the UK?")
print("     • What's the weather in London today?")
print("     • What is the Bank of England base rate?")
print("     • Latest UK technology news")
print()
print("  🔄 [Memory test] Follow-up questions:")
print("     • First ask: 'What is a sanction?'")
print("     • Then ask: 'How long does it last?' (should remember context)")
print("-" * 40)

while True:
    user_input = input("\nYou: ").strip()
    
    if user_input.lower() == "quit":
        print("Goodbye! 👋")
        break
    
    if user_input.lower() == "clear":
        # Create a new thread ID to reset conversation
        import uuid
        new_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": new_id}}
        state = {
            "messages": [SystemMessage(content=SYSTEM_PROMPT)],
            "llm_calls": 0,
            "tool_used": False
        }
        print(f"Conversation cleared! New thread ID: {new_id[:8]}...")
        continue
    
    if not user_input:
        continue
    
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)], "llm_calls": 0, "tool_used": False},
        config
    )
    
    state["messages"].append(HumanMessage(content=user_input))
    state["messages"].append(result["messages"][-1])
    
    response = result["messages"][-1].content
    print(f"\n🤖 Assistant: {response}")

print("\n" + "="*50)
print("✅ Session complete.")
print("="*50)