# test_prompt.py — compare grounding prompt vs baseline on first 3 cases
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.pipeline import retrieve
from langchain_ollama import ChatOllama

cases = json.load(open(ROOT / "golden_set.json"))
llm = ChatOllama(model="llama3.2")

STRICT = (
    "You are a support assistant. Answer ONLY using the facts in the CONTEXT below. "
    "Do NOT add any information not in the context. If the context cannot answer, "
    'say exactly: "I cannot answer from the available knowledge base."\n\n'
    "CONTEXT:\n{ctx}\n\nQUESTION: {q}\n\nANSWER:"
)

for c in cases[:3]:
    ctx = retrieve(c["ticket"], 3)
    draft = llm.invoke(STRICT.format(ctx="\n".join(ctx), q=c["ticket"])).content
    print("Q:", c["ticket"])
    print("GROUND:", c["answer"])
    print("DRAFT:", draft)
    print("---")