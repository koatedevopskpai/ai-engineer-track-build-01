# api/app.py #

from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import graph
from agent.hitl import run_with_approval

app = FastAPI(title="AI Customer Service Agent")


class Ticket(BaseModel):
    text: str


@app.post("/resolve")
def resolve(ticket: Ticket):
    return run_with_approval(graph, ticket.text)


@app.get("/health")
def health():
    return {"status": "ok"}
