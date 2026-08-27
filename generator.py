# generator.py #

import json
import random
import uuid

random.seed(42)  # reproducible + seeded known-answer cases

KB = [
    (
        "password reset",
        "Users reset a forgotten password via the self-service portal at /account/reset.",
    ),
    (
        "billing cycle",
        "Invoices are generated on the 1st of each month and due within 30 days.",
    ),
    (
        "refund policy",
        "Refunds are available within 14 days of purchase if the service is unused.",
    ),
    (
        "cancel plan",
        "Subscriptions can be cancelled anytime from Settings > Plan; access ends at the cycle end.",
    ),
    (
        "api key",
        "Generate an API key under Settings > Developer; keys rotate every 90 days.",
    ),
]
TICKETS = [
    "I forgot my password and cannot log in.",
    "When is my next invoice due?",
    "Can I get a refund for last month?",
    "How do I cancel my plan?",
    "My API key stopped working, how do I rotate it?",
    "The dashboard shows an error on load.",
    "How do I add a second admin user?",
    "What are your uptime guarantees?",
]


def gen_case(i):
    topic, answer = KB[i % len(KB)]
    return {
        "id": str(uuid.uuid4()),
        "topic": topic,
        "ticket": TICKETS[i % len(TICKETS)],
        "answer": answer,
    }


cases = [gen_case(i) for i in range(30)]
with open("golden_set.json", "w") as f:
    json.dump(cases, f, indent=2)

with open("corpus.txt", "w", encoding="utf-8") as f:
    for _, text in KB:
        f.write(text + "\n\n")
print("generated 30-case golden set + corpus")
