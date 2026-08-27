# agent/hitl.py — wraps the graph with a human decision layer
from agent.graph import checkpointer


def run_with_approval(graph, ticket, thread_id="case-1"):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"ticket": ticket}, config=config)
    if result["confidence"] < 0.7:
        # Low confidence → escalate with full context, never auto-send
        return {
            "status": "ESCALATE",
            "context": result["context"],
            "draft": result["draft"],
        }
    decision = input(f"Approve this reply? (y/n)\n{result['draft']}\n> ")
    return {
        "status": "SENT" if decision.lower() == "y" else "BLOCKED",
        "context": result["context"],
        "draft": result["draft"],
    }