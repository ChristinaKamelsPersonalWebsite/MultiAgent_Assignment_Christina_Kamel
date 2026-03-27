import os
import re
import json
import csv
from pathlib import Path

# Redis replaces ArangoDB
import redis

from math import exp
from typing import List, Dict, Any, TypedDict, Literal, Annotated, Optional

from groq import Groq, RateLimitError

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = "data"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
FETCH_K = 10
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SUPERVISOR_MODEL = "llama-3.3-70b-versatile"
RAG_AGENT_MODEL = "llama-3.3-70b-versatile"
DB_AGENT_MODEL = "llama-3.3-70b-versatile"
DIRECT_AGENT_MODEL = "llama-3.3-70b-versatile"
RAG_GENERATION_MODEL = "llama-3.3-70b-versatile"

MAX_ITERATIONS = 4
MAX_ARG_LENGTH = 800

ACTIVE_THREAD_ID = "default-thread"

# Redis config — reads from environment variables set in Docker
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# ── Prompts ───────────────────────────────────────────────────────────────────

SUPERVISOR_PROMPT = """You are the supervisor of a multi-agent photovoltaic assistant.

Your job is ONLY to choose the best specialist.

Available specialists:
- rag_agent -> PV / solar / lecture-document / course-content questions
- db_agent -> database / save / logs / structured data requests
- memory_agent -> user facts like name, preferences, or recall questions
- direct_agent -> greetings / casual conversation / simple non-tool replies

Return ONLY ONE of these exact tokens:
rag_agent
db_agent
memory_agent
direct_agent
"""

RAG_AGENT_PROMPT = """You are a photovoltaic RAG specialist.

Use:
- `pv_rag_search` for PV / solar / lecture questions
- `list_available_topics` if the user explicitly asks for available files, lectures, or topics

Rules:
1. Prefer tools over guessing.
2. Preserve the tool result faithfully.
3. Do not invent unsupported facts.
4. If the retrieved context is partial, answer using the available evidence and make that clear.
"""

DB_AGENT_PROMPT = """You are a database specialist.

Use:
- `save_to_db` for saving structured information
- `show_database` for showing saved database contents

Rules:
1. For save / store / log requests -> use save_to_db
2. For requests to view saved data -> use show_database
3. Keep the final response short and clear
4. Do not invent database results
"""

DIRECT_AGENT_PROMPT = """You are a helpful photovoltaic assistant.
Handle only greetings and simple casual replies.
Keep the reply short.
"""

RAG_SYSTEM_PROMPT = """You are a PV (photovoltaics) tutor using Retrieval-Augmented Generation (RAG).

Rules:
1. Answer from the provided CONTEXT.
2. If the context clearly contains the answer, answer directly and clearly.
3. If the context is partially relevant, answer using the available evidence and make that clear.
4. Only say "Not found in the provided documents." if the answer truly does not appear in the retrieved context.
5. After that, you may add a short fallback explanation labeled "General:".
6. Keep the answer accurate, simple, and step-by-step when useful.
7. End with a short Sources line using the chunk numbers like [1], [2].
"""

# ── Document loading ──────────────────────────────────────────────────────────

def load_documents(data_dir: str) -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(data_dir):
        print(f"Warning: data directory '{data_dir}' not found.")
        return docs

    for fn in os.listdir(data_dir):
        path = os.path.join(data_dir, fn)
        if fn.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif fn.lower().endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs


def build_vectorstore(docs: List[Document]) -> Optional[FAISS]:
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_documents(chunks, embeddings)

    print(f"Loaded {len(docs)} pages/entries.")
    print(f"Created {len(chunks)} chunks.")
    return db


def format_retrieved_docs(retrieved_docs: List[Document], max_chars: int = 300) -> str:
    out: List[str] = []
    for i, d in enumerate(retrieved_docs, 1):
        src = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", None)
        page_str = f", page {page}" if page is not None else ""
        text = d.page_content.strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        out.append(f"[{i}] {src}{page_str}\n{text}")
    return "\n\n".join(out)


def retrieval_confidence(scores: List[float]) -> float:
    if not scores:
        return 0.0
    weights = [exp(-s) for s in scores]
    total = sum(weights) if sum(weights) > 0 else 1.0
    return 100.0 * (max(weights) / total)


def rough_relevance(question: str, docs: List[Document]) -> float:
    q_words = set(re.findall(r"[a-zA-Z0-9\-]+", question.lower()))
    q_words = {w for w in q_words if len(w) > 2}
    if not q_words or not docs:
        return 0.0

    joined = " ".join(d.page_content.lower() for d in docs)
    hits = sum(1 for w in q_words if w in joined)
    return hits / max(len(q_words), 1)


def repair_query(question: str) -> str:
    q = question.lower().strip()

    replacements = {
        "lattitude": "latitude",
        "inclination angle": "tilt angle",
        "panel angle": "tilt angle",
        "solar panel": "pv module",
        "pv panel": "pv module",
        "iv curve": "i-v curve",
        "full sun": "1-sun",
        "sc": "solar constant",
    }

    for old, new in replacements.items():
        q = q.replace(old, new)

    expansions = []
    if "1-sun" in q or "1 sun" in q:
        expansions.append("1 kW/m^2 full sun insolation")
    if "latitude" in q:
        expansions.append("site latitude solar position L")
    if "tilt angle" in q:
        expansions.append("collector tilt angle inclination")
    if any(x in q for x in ["i-v", "voc", "isc", "mpp"]):
        expansions.append("photovoltaic cell current voltage curve")
    if "solar cell" in q:
        expansions.append("photovoltaic cell definition")
    if "capacity factor" in q:
        expansions.append("actual energy rated power time")
    if "solar constant" in q:
        expansions.append("extraterrestrial radiation solar constant")
    if "declination" in q:
        expansions.append("solar declination angle day number")
    if expansions:
        q += " " + " ".join(expansions)

    return q

# ── Global stores and initialization ─────────────────────────────────────────

MEMORY_STORE: Dict[str, List[str]] = {}

docs = load_documents(DATA_DIR)
vector_db = build_vectorstore(docs)
raw_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Redis setup ───────────────────────────────────────────────────────────────

def init_redis() -> redis.Redis:
    # decode_responses=True means Redis returns strings instead of raw bytes
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    # ping() raises an exception if Redis is not reachable — catches connection issues early
    client.ping()
    print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    return client

REDIS_CLIENT = init_redis()

# ── Helper utilities ──────────────────────────────────────────────────────────

def get_thread_id(config: Optional[Dict[str, Any]]) -> str:
    if not config:
        return "default-thread"
    return config.get("configurable", {}).get("thread_id", "default-thread")


def set_active_thread_id(thread_id: str) -> None:
    global ACTIVE_THREAD_ID
    ACTIVE_THREAD_ID = thread_id


def get_last_user_message(state: "AgentState") -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def get_last_ai_message_text(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""


def sanitize_tool_text(text: str, max_len: int = 2200) -> str:
    cleaned = re.sub(r"<[^>]+>", "", text).strip()
    cleaned = cleaned.replace("<|end_header_id|>", "")
    cleaned = cleaned.replace("<|start_header_id|>", "")
    cleaned = cleaned.replace("<|eot_id|>", "")
    cleaned = cleaned.strip()

    lowered = cleaned.lower()
    injection_patterns = [
        "ignore previous instructions",
        "ignore your instructions",
        "system prompt:",
        "you are now",
        "disregard your instructions",
    ]

    for pat in injection_patterns:
        if pat in lowered:
            return "[Sanitized tool output removed due to possible prompt injection.]"

    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "... [truncated]"

    return cleaned


def route_by_keywords(user_text: str) -> Optional[str]:
    text = user_text.lower().strip()

    db_patterns = [
        "database",
        "save this",
        "save my",
        "store this",
        "log this",
        "metrics",
        "save this note",
        "save my name",
        "show database",
        "show the database",
        "what is in the database",
    ]

    rag_keywords = [
        "photovoltaic",
        "solar",
        "irradiance",
        "insolation",
        "radiation",
        "module",
        "cell",
        "i-v",
        "voc",
        "isc",
        "mpp",
        "1-sun",
        "tilt angle",
        "latitude",
        "solar constant",
        "declination",
        "air mass",
        "azimuth",
        "altitude angle",
        "beam radiation",
        "diffuse radiation",
        "reflected radiation",
        "pv cell",
        "solar cell",
    ]

    direct_patterns = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good evening",
    ]

    if any(p in text for p in db_patterns):
        return "db_agent"

    if any(p in text for p in rag_keywords):
        return "rag_agent"

    if text.startswith("my name is ") or text in {"what is my name", "who am i", "what did i tell you"}:
        return "memory_agent"

    if text in direct_patterns or any(text.startswith(p) for p in direct_patterns):
        return "direct_agent"

    return None

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def pv_rag_search(question: str) -> str:
    """Use this tool for technical PV / solar questions that should be answered from the lecture PDFs."""
    if vector_db is None:
        return "No PV documents were loaded, so RAG search is unavailable."

    improved_question = repair_query(question)

    retrieved_docs = vector_db.max_marginal_relevance_search(
        improved_question,
        k=TOP_K,
        fetch_k=FETCH_K,
    )

    scored = vector_db.similarity_search_with_score(improved_question, k=TOP_K)
    scores = [score for _, score in scored]
    confidence = retrieval_confidence(scores)
    relevance = rough_relevance(question, retrieved_docs)

    if not retrieved_docs:
        return "Not found in the provided documents."

    sources = format_retrieved_docs(retrieved_docs)
    context = "\n\n---\n\n".join(
        f"[{i}] {d.page_content.strip()}" for i, d in enumerate(retrieved_docs, 1)
    )

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n\n"
                f"RETRIEVAL_CONFIDENCE: {confidence:.1f}\n"
                f"RELEVANCE_SCORE: {relevance:.2f}\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"Instructions:\n"
                f"- Answer from the context when possible.\n"
                f"- If the context is partial, say that clearly.\n"
                f"- Only say 'Not found in the provided documents.' if the answer truly is not in the context.\n"
                f"- Keep the answer concise.\n\n"
                f"Answer:"
            ),
        },
    ]

    resp = raw_groq.chat.completions.create(
        model=RAG_GENERATION_MODEL,
        messages=messages,
        temperature=0.1,
    )

    answer = resp.choices[0].message.content.strip()

    return sanitize_tool_text(
        f"Retrieval confidence: {confidence:.1f}/100\n"
        f"Relevance score: {relevance:.2f}\n\n"
        f"Retrieved chunks:\n{sources}\n\n"
        f"Generated answer:\n{answer}"
    )


@tool
def list_available_topics() -> str:
    """List available lecture files in the local data directory."""
    if not os.path.isdir(DATA_DIR):
        return "Data directory not found."
    files = sorted(
        [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".pdf", ".txt"))]
    )
    if not files:
        return "No PDF or TXT lecture files were found."
    return "Available files:\n" + "\n".join(f"- {f}" for f in files)


@tool
def remember_fact(fact: str) -> str:
    """Store a durable user fact for the active thread."""
    thread_id = ACTIVE_THREAD_ID
    fact = fact.strip()
    if not fact:
        return "No fact provided."

    if thread_id not in MEMORY_STORE:
        MEMORY_STORE[thread_id] = []

    if fact not in MEMORY_STORE[thread_id]:
        MEMORY_STORE[thread_id].append(fact)

    return f"Stored memory: {fact}"


@tool
def recall_memory(query: str) -> str:
    """Recall stored user facts for the active thread."""
    thread_id = ACTIVE_THREAD_ID
    facts = MEMORY_STORE.get(thread_id, [])
    if not facts:
        return "No stored memory is available yet."
    return "Stored facts:\n" + "\n".join(f"- {fact}" for fact in facts)


@tool
def save_to_db(collection: str, payload_json: str) -> str:
    """Save a JSON payload to a Redis list (acts as a collection/table)."""
    allowed_collections = {"chat_logs", "user_notes", "analytics_events"}

    if collection not in allowed_collections:
        return f"Blocked: collection '{collection}' is not allowed."

    if len(payload_json) > MAX_ARG_LENGTH:
        return "Blocked: payload too long."

    try:
        payload = json.loads(payload_json)
    except Exception:
        return "Blocked: payload_json is not valid JSON."

    try:
        REDIS_CLIENT.lpush(collection, json.dumps(payload))
        count = REDIS_CLIENT.llen(collection)
        return f"Saved to '{collection}'. Total entries in collection: {count}."
    except Exception as e:
        return f"Database save failed: {str(e)}"


@tool
def show_database() -> str:
    """Show the last 20 entries from each Redis collection."""
    try:
        result: Dict[str, Any] = {}
        for collection_name in ["user_notes", "analytics_events", "chat_logs"]:
            raw_entries = REDIS_CLIENT.lrange(collection_name, 0, 19)
            result[collection_name] = [json.loads(entry) for entry in raw_entries]
        return sanitize_tool_text(json.dumps(result, indent=2))
    except Exception as e:
        return f"Database read failed: {str(e)}"

# ── Tool sets ─────────────────────────────────────────────────────────────────

rag_tools = [pv_rag_search, list_available_topics]
db_tools = [save_to_db, show_database]

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_agent: str
    iteration_count: int
    blocked: bool

# ── Models ────────────────────────────────────────────────────────────────────

supervisor_llm = ChatGroq(model=SUPERVISOR_MODEL, temperature=0)
rag_llm = ChatGroq(model=RAG_AGENT_MODEL, temperature=0).bind_tools(rag_tools, tool_choice="auto")
db_llm = ChatGroq(model=DB_AGENT_MODEL, temperature=0).bind_tools(db_tools)
direct_llm = ChatGroq(model=DIRECT_AGENT_MODEL, temperature=0)

# ── Guardrails ────────────────────────────────────────────────────────────────

def input_guardrail(state: AgentState) -> Dict[str, Any]:
    last_msg = get_last_user_message(state).lower()

    injection_patterns = [
        "ignore your instructions",
        "ignore previous instructions",
        "disregard your instructions",
        "system prompt:",
        "reveal your prompt",
        "you are now",
    ]

    dangerous_db_patterns = [
        "drop collection",
        "truncate collection",
        "delete all records",
        "remove all records",
    ]

    for pattern in injection_patterns + dangerous_db_patterns:
        if pattern in last_msg:
            return {
                "messages": [
                    AIMessage(
                        content="I detected a potentially unsafe request. Please rephrase your question."
                    )
                ],
                "next_agent": "FINISH",
                "blocked": True,
            }

    return {"next_agent": "supervisor", "blocked": False}


def output_guardrail(state: AgentState) -> Dict[str, Any]:
    last_text = get_last_ai_message_text(state["messages"])

    pii_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b\d{16}\b",
        r"password\s*[:=]\s*\S+",
    ]

    for pattern in pii_patterns:
        if re.search(pattern, last_text):
            return {
                "messages": [
                    AIMessage(content="Response blocked because it may contain sensitive data.")
                ]
            }
    return {}


def input_guard_router(state: AgentState) -> Literal["supervisor", "__end__"]:
    if state.get("blocked") or state.get("next_agent") == "FINISH":
        return "__end__"
    return "supervisor"

# ── Supervisor ────────────────────────────────────────────────────────────────

def parse_supervisor_choice(raw_text: str) -> str:
    text = raw_text.strip().lower()

    if "rag_agent" in text:
        return "rag_agent"
    if "db_agent" in text:
        return "db_agent"
    if "memory_agent" in text:
        return "memory_agent"
    if "direct_agent" in text:
        return "direct_agent"
    return "direct_agent"


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    count = state.get("iteration_count", 0) + 1
    if count > MAX_ITERATIONS:
        return {
            "messages": [AIMessage(content="Maximum step limit reached.")],
            "next_agent": "FINISH",
            "iteration_count": count,
        }

    user_text = get_last_user_message(state)

    keyword_route = route_by_keywords(user_text)
    if keyword_route is not None:
        print("Supervisor chose:", keyword_route)
        return {
            "messages": [AIMessage(content=f"[Supervisor route] {keyword_route}")],
            "next_agent": keyword_route,
            "iteration_count": count,
        }

    try:
        messages = [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=user_text)]
        response = supervisor_llm.invoke(messages)
        next_agent = parse_supervisor_choice(str(response.content))
    except RateLimitError:
        next_agent = "direct_agent"
    except Exception:
        next_agent = "direct_agent"

    print("Supervisor chose:", next_agent)

    return {
        "messages": [AIMessage(content=f"[Supervisor route] {next_agent}")],
        "next_agent": next_agent,
        "iteration_count": count,
    }


def route_from_supervisor(state: AgentState) -> str:
    return state["next_agent"]

# ── Specialist agents ─────────────────────────────────────────────────────────

def memory_agent_node(state: AgentState) -> Dict[str, Any]:
    user_msg = get_last_user_message(state).strip()
    lower = user_msg.lower()

    if "my name is" in lower:
        fact = user_msg
        remember_fact.invoke({"fact": fact})
        return {"messages": [AIMessage(content=f"Stored your info: {fact}")]}

    recalled = recall_memory.invoke({"query": user_msg})
    if recalled == "No stored memory is available yet.":
        return {"messages": [AIMessage(content=recalled)]}

    if "what is my name" in lower or "who am i" in lower:
        facts = MEMORY_STORE.get(ACTIVE_THREAD_ID, [])
        for fact in facts:
            m = re.search(r"my name is\s+(.+)", fact, re.IGNORECASE)
            if m:
                name = m.group(1).strip().rstrip(".")
                return {"messages": [AIMessage(content=f"Your name is {name}.")]}

    return {"messages": [AIMessage(content=recalled)]}


def rag_agent_node(state: AgentState) -> Dict[str, Any]:
    user_text = get_last_user_message(state)
    try:
        tool_result = pv_rag_search.invoke({"question": user_text})
        return {"messages": [AIMessage(content=str(tool_result))]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"RAG agent error: {str(e)}")]}


def db_agent_node(state: AgentState) -> Dict[str, Any]:
    messages = [SystemMessage(content=DB_AGENT_PROMPT)] + state["messages"]
    try:
        response = db_llm.invoke(messages)
        return {"messages": [response]}
    except RateLimitError:
        return {
            "messages": [
                AIMessage(content="The Groq API rate limit was reached. Please try again later.")
            ]
        }
    except Exception as e:
        return {"messages": [AIMessage(content=f"Database agent error: {str(e)}")]}


def direct_agent_node(state: AgentState) -> Dict[str, Any]:
    user_text = get_last_user_message(state)
    try:
        response = direct_llm.invoke(
            [SystemMessage(content=DIRECT_AGENT_PROMPT), HumanMessage(content=user_text)]
        )
        return {"messages": [response]}
    except RateLimitError:
        return {"messages": [AIMessage(content="Rate limit reached. Please try again later.")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Direct agent error: {str(e)}")]}

# ── Tool validation ───────────────────────────────────────────────────────────

def validate_tool_calls_factory(valid_tools: List[str]):
    valid_set = set(valid_tools)

    def validator(state: AgentState) -> Dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return {}

        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})

            if tool_name not in valid_set:
                return {"messages": [AIMessage(content=f"Blocked: unknown tool '{tool_name}'.")]}

            for key, val in tool_args.items():
                if isinstance(val, str) and len(val) > MAX_ARG_LENGTH:
                    return {"messages": [AIMessage(content=f"Blocked: argument '{key}' is too long.")]}

        return {}

    return validator


validate_rag_tools = validate_tool_calls_factory(["pv_rag_search", "list_available_topics"])
validate_db_tools = validate_tool_calls_factory(["save_to_db", "show_database"])

# ── Specialist routing ────────────────────────────────────────────────────────

def specialist_router(state: AgentState) -> Literal["tools", "done"]:
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None)
    if tool_calls:
        return "tools"
    return "done"

# ── Tool nodes ────────────────────────────────────────────────────────────────

rag_tool_node = ToolNode(rag_tools)
db_tool_node = ToolNode(db_tools)

# ── Graph ─────────────────────────────────────────────────────────────────────

graph_builder = StateGraph(AgentState)

graph_builder.add_node("input_guard", input_guardrail)
graph_builder.add_node("output_guard", output_guardrail)

graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("rag_agent", rag_agent_node)
graph_builder.add_node("db_agent", db_agent_node)
graph_builder.add_node("memory_agent", memory_agent_node)
graph_builder.add_node("direct_agent", direct_agent_node)

graph_builder.add_node("validate_rag_tools", validate_rag_tools)
graph_builder.add_node("validate_db_tools", validate_db_tools)

graph_builder.add_node("rag_tools", rag_tool_node)
graph_builder.add_node("db_tools", db_tool_node)

graph_builder.add_edge(START, "input_guard")
graph_builder.add_conditional_edges(
    "input_guard",
    input_guard_router,
    {
        "supervisor": "supervisor",
        "__end__": END,
    },
)

graph_builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "rag_agent": "rag_agent",
        "db_agent": "db_agent",
        "memory_agent": "memory_agent",
        "direct_agent": "direct_agent",
        "FINISH": "output_guard",
    },
)

graph_builder.add_conditional_edges(
    "rag_agent",
    specialist_router,
    {
        "tools": "validate_rag_tools",
        "done": "output_guard",
    },
)
graph_builder.add_edge("validate_rag_tools", "rag_tools")
graph_builder.add_edge("rag_tools", "rag_agent")

graph_builder.add_conditional_edges(
    "db_agent",
    specialist_router,
    {
        "tools": "validate_db_tools",
        "done": "output_guard",
    },
)
graph_builder.add_edge("validate_db_tools", "db_tools")
graph_builder.add_edge("db_tools", "db_agent")

graph_builder.add_edge("memory_agent", "output_guard")
graph_builder.add_edge("direct_agent", "output_guard")

graph_builder.add_edge("output_guard", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# ── Display helpers ───────────────────────────────────────────────────────────

def extract_display_text(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            text = str(msg.content).strip()
            if text and not text.startswith("[Supervisor route]"):
                return text
    return "No assistant response found."

# ── Interactive chat ──────────────────────────────────────────────────────────

def interactive_chat() -> None:
    print("\n=== INTERACTIVE MODE (type 'quit' to stop) ===\n")
    thread = {"configurable": {"thread_id": "pv-demo"}}
    set_active_thread_id(get_thread_id(thread))

    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            print("Bye!")
            break
        if not user:
            continue

        set_active_thread_id(get_thread_id(thread))

        result = graph.invoke(
            {
                "messages": [HumanMessage(content=user)],
                "next_agent": "",
                "iteration_count": 0,
                "blocked": False,
            },
            config=thread,
        )

        print("\nAssistant:")
        print(extract_display_text(result["messages"]))
        print()

# ── Manual RAG evaluation ─────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    "What is a photovoltaic cell?",
    "What is the difference between a PV cell, a module, and an array?",
    "What is the solar constant?",
    "What does 1-sun mean in photovoltaics?",
    "What is the difference between beam radiation and diffuse radiation?",
    "What is reflected radiation?",
    "What are the main components of solar radiation reaching a tilted collector?",
    "What is the solar declination angle?",
    "What is the solar altitude angle?",
    "What is the solar azimuth angle?",
    "What does the I-V curve of a PV cell represent?",
    "What is short-circuit current (Isc)?",
    "What is open-circuit voltage (Voc)?",
    "What is the maximum power point (MPP)?",
    "Why is tilt angle important for PV panels?"
]


def evaluate_single_rag_question(question: str) -> Dict[str, Any]:
    result = pv_rag_search.invoke({"question": question})
    result_text = str(result)

    answer = ""
    retrieved = ""
    confidence = ""
    relevance = ""

    if "Retrieval confidence:" in result_text:
        try:
            confidence = result_text.split("Retrieval confidence:")[1].split("\n")[0].strip()
        except Exception:
            confidence = ""

    if "Relevance score:" in result_text:
        try:
            relevance = result_text.split("Relevance score:")[1].split("\n")[0].strip()
        except Exception:
            relevance = ""

    if "Retrieved chunks:\n" in result_text and "\n\nGenerated answer:\n" in result_text:
        try:
            retrieved = result_text.split("Retrieved chunks:\n", 1)[1].split("\n\nGenerated answer:\n", 1)[0].strip()
        except Exception:
            retrieved = ""

    if "Generated answer:\n" in result_text:
        try:
            answer = result_text.split("Generated answer:\n", 1)[1].strip()
        except Exception:
            answer = result_text.strip()
    else:
        answer = result_text.strip()

    return {
        "question": question,
        "retrieval_confidence": confidence,
        "relevance_score": relevance,
        "retrieved_chunks": retrieved,
        "generated_answer": answer,
        "retrieval_relevant_manual": "",
        "answer_correct_manual": "",
        "faithful_manual": "",
        "notes": "",
    }


def run_manual_rag_evaluation(output_csv: str = "rag_manual_eval.csv") -> None:
    print("\n=== RUNNING MANUAL RAG EVALUATION ===\n")

    rows: List[Dict[str, Any]] = []

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"[{i}/{len(EVAL_QUESTIONS)}] {question}")
        row = evaluate_single_rag_question(question)
        rows.append(row)
        print("Answer:")
        print(row["generated_answer"])
        print("-" * 70)

    fieldnames = [
        "question",
        "retrieval_confidence",
        "relevance_score",
        "retrieved_chunks",
        "generated_answer",
        "retrieval_relevant_manual",
        "answer_correct_manual",
        "faithful_manual",
        "notes",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved evaluation file to: {output_csv}")


def summarize_manual_rag_evaluation(csv_path: str = "rag_manual_eval.csv") -> None:
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    total = 0
    retrieval_yes = 0
    answer_yes = 0
    faithful_yes = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row["retrieval_relevant_manual"].strip().lower() == "yes":
                retrieval_yes += 1
            if row["answer_correct_manual"].strip().lower() == "yes":
                answer_yes += 1
            if row["faithful_manual"].strip().lower() == "yes":
                faithful_yes += 1

    if total == 0:
        print("No rows found in evaluation file.")
        return

    print("\n=== MANUAL RAG EVALUATION SUMMARY ===")
    print(f"Number of questions: {total}")
    print(f"Retrieval relevance: {retrieval_yes}/{total} = {100 * retrieval_yes / total:.1f}%")
    print(f"Answer accuracy:     {answer_yes}/{total} = {100 * answer_yes / total:.1f}%")
    print(f"Faithfulness:        {faithful_yes}/{total} = {100 * faithful_yes / total:.1f}%")


if __name__ == "__main__":
    while True:
        print("Choose mode:")
        print("1. Text interactive chat")
        print("2. Run manual RAG evaluation")
        print("3. Summarize manual RAG evaluation")

        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == "1":
            interactive_chat()
            break
        elif choice == "2":
            run_manual_rag_evaluation()
            break
        elif choice == "3":
            summarize_manual_rag_evaluation()
            break
        else:
            print("Please enter only 1, 2, or 3.\n")