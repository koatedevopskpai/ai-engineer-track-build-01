# evals/evaluate.py #
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from rag.pipeline import retrieve
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")
with open("golden_set.json") as f:
    cases = json.load(f)

rows = []
for c in cases:
    ctx = retrieve(c["ticket"], k=3)
    draft = llm.invoke(
        f"Answer from context only.\nContext:\n{ctx}\nQ: {c['ticket']}"
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
score = evaluate(ds, metrics=[faithfulness, answer_relevancy])
print(score)
if score["faithfulness"] < 0.9:
    raise SystemExit("FAIL: faithfulness below 90% gate")
print("PASS: eval gate met")
