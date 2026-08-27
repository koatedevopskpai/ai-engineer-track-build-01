# guardrails/security.py #

import re
import time
import os

PII = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b|\b\d{4}-\d{4}-\d{4}\b")


def redact(text: str) -> str:
    return PII.sub("[REDACTED]", text)


INJECTION = ["ignore previous instructions", "system:", "forget your rules"]


def block_injection(prompt: str) -> bool:
    return any(tok in prompt.lower() for tok in INJECTION)


class CostLog:
    def __init__(self):
        self.runs = []

    def record(self, model, tokens_in, tokens_out):
        # $/1k tokens example for llama3.2 ~ $0.17/$0.17 (adjust to real rates)
        cost = (tokens_in + tokens_out) / 1000 * 0.17
        self.runs.append(
            {"model": model, "cost_usd": round(cost, 4), "at": time.time()}
        )
        return cost
