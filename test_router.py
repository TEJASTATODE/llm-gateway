from gateway.routing.cost_router import CostRouter
from gateway.providers.base import Message

router = CostRouter()

test_cases = [
    ("Greeting",          [Message(role="user", content="hey how are you")]),
    ("Simple factual",    [Message(role="user", content="what is the capital of France")]),
    ("Medium explain",    [Message(role="user", content="explain how neural networks work")]),
    ("Complex analysis",  [Message(role="user", content="compare and analyse tradeoffs between microservices and monolith architecture in detail")]),
    ("Code block",        [Message(role="user", content="```python\ndef sort(arr): pass\n``` fix this function")]),
    ("Ambiguous",         [Message(role="user", content="tell me about machine learning applications")]),
    ("Deep conversation", [
        Message(role="user", content="what is recursion"),
        Message(role="assistant", content="recursion is..."),
        Message(role="user", content="can you give an example"),
        Message(role="assistant", content="sure..."),
        Message(role="user", content="now explain dynamic programming"),
        Message(role="assistant", content="dp is..."),
        Message(role="user", content="how does memoization relate"),
    ]),
]

print("=" * 75)
print(f"{'Description':<20} {'Score':>5} {'Tier':<10} {'By':<15} {'Model'}")
print("=" * 75)

for desc, messages in test_cases:
    d = router.decide(messages, preferred_provider="gemini")
    print(f"{desc:<20} {d.complexity_score:>5} {d.tier:<10} {d.classified_by:<15} {d.model}")
    print(f"  → {d.reasoning}")
    print()

print("=" * 75)
print("COST COMPARISON")
print("=" * 75)
models = [
    ("gemini-2.5-flash-lite", 100, 200),
    ("gpt-4o-mini",           100, 200),
    ("gpt-4o",                100, 200),
]
for model, pt, ct in models:
    cost = router.estimate_cost(model, pt, ct)
    saving = router.savings_vs_powerful(model, pt, ct)
    print(f"{model:<30} ${cost:.6f}   saves ${saving:.6f} vs gpt-4o")