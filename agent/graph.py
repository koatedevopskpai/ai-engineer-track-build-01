# agent/graph.py #

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from rag.pipeline import retrieve

llm = ChatOllama(model="llama3.2", temperature=0)
PROMPT = PromptTemplate.from_template(
    "Answer using ONLY the context. If the context cannot answer, say so and suggest escalation.\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
checkpointer = MemorySaver()


class State(TypedDict):
    ticket: str
    tier: str
    context: list
    draft: str
    confidence: float


def triage(state):
    r = llm.invoke(f"Return only one of L1/L2/L3 for: {state['ticket']}")
    return {"tier": r.content.strip()}


def resolve(state):
    ctx = retrieve(state["ticket"], k=3)
    draft = llm.invoke(
        PROMPT.format(context="\n\n".join(ctx), question=state["ticket"])
    )
    return {"context": ctx, "draft": draft.content}


def qa(state):
    # grounding heuristic: does the draft reuse context words?
    ctx_words = set(" ".join(state["context"]).lower().split())
    draft_words = set(state["draft"].lower().split())
    overlap = len(ctx_words & draft_words) / max(1, len(draft_words))
    return {"confidence": min(1.0, overlap + 0.4)}


def route(state):
    return "escalate" if state["confidence"] < 0.7 else "approve"


g = StateGraph(State)
g.add_node("triage", triage)
g.add_node("resolve", resolve)
g.add_node("qa", qa)
g.set_entry_point("triage")
g.add_edge("triage", "resolve")
g.add_edge("resolve", "qa")
g.add_conditional_edges("qa", route, {"approve": END, "escalate": END})
graph = g.compile(checkpointer=checkpointer)
