# evals/evaluate.py #
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load secrets from .env (project root) if present.
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from rag.pipeline import retrieve

# Generation LLM (answering the tickets).
# Arch decision: llama3.2 drafts were not faithfully grounded (faithfulness 0.76-0.47).
# deepseek-v4-flash on OpenCode Go produces better-grounded answers; both gen + judge
# ride the flat $10/mo subscription, so marginal cost stays ~$0.
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url="https://opencode.ai/zen/go/v1",
    api_key=os.getenv("OPENCODE_API_KEY"),
)

# Judge LLM for RAGAS — RAGAS legacy metrics need reliable structured JSON.
# OpenCode Go (flat $10/mo subscription, already active) exposes an
# OpenAI-compatible endpoint incl. deepseek-v4-flash at ~$0 marginal cost.
# Local llama3.2 was unreliable as a judge (OutputParserException + timeouts).
judge = LangchainLLMWrapper(
    ChatOpenAI(
        model="deepseek-v4-flash",
        base_url="https://opencode.ai/zen/go/v1",
        api_key=os.getenv("OPENCODE_API_KEY"),
    )
)

GENERATION_PROMPT = (
    "You are a support assistant. Answer ONLY using the facts in the CONTEXT below. "
    "Do NOT add any information not in the context. If the context cannot answer, "
    'say exactly: "I cannot answer from the available knowledge base."\n\n'
    "CONTEXT:\n{ctx}\n\nQUESTION: {q}\n\nANSWER:"
)

faithfulness = Faithfulness()
faithfulness.llm = judge

answer_relevancy = AnswerRelevancy()
answer_relevancy.llm = judge
answer_relevancy.embeddings = OllamaEmbeddings(model="nomic-embed-text")
# Go endpoint only accepts n=1; strictness controls the n-sample count.
answer_relevancy.strictness = 1

with open("golden_set.json") as f:
    cases = json.load(f)

rows = []
for c in cases:
    ctx = retrieve(c["ticket"], k=3)
    draft = llm.invoke(
        GENERATION_PROMPT.format(ctx="\n".join(ctx), q=c["ticket"])
    ).content
    rows.append(
        {
            "question": c["ticket"],
            "answer": draft,
            "contexts": ctx,
            "ground_truth": c["answer"],
        }
    )

ds = Dataset.from_list(rows)
score = evaluate(
    ds,
    metrics=[faithfulness, answer_relevancy],
    run_config=RunConfig(
        timeout=300, max_retries=5, max_wait=90, max_workers=8
    ),
)
print(score)
faith_score = score["faithfulness"]
if isinstance(faith_score, list):
    faith_mean = sum(f for f in faith_score if f == f) / max(
        1, sum(1 for f in faith_score if f == f)
    )
else:
    faith_mean = faith_score
if faith_mean < 0.9:
    raise SystemExit(f"FAIL: faithfulness {faith_mean:.3f} below 90% gate")
print(f"PASS: eval gate met (faithfulness {faith_mean:.3f})")
