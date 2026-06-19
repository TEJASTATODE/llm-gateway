import requests

questions = [
    "what is machine learning",
    "what is machine learning",
    "explain machine learning to me",
    "compare microservices vs monolith architecture in detail",
    "what is recursion",
    "implement binary search in Python",
    "how are you",
    "what is recursion",
]

for q in questions:
    r = requests.post("http://localhost:8000/v1/chat/completions", json={
        "messages": [{"role": "user", "content": q}],
        "preferred_provider": "gemini"
    })
    d = r.json()
    cache = d.get("cache_hit", False)
    provider = d.get("provider_used", "?")
    cost = d.get("cost", {}).get("estimated_usd", 0)
    tier = d.get("routing", {}).get("tier", "cache") if not cache else "cache"
    print(f"cache={cache} | provider={provider} | tier={tier} | cost=${cost:.6f} | q={q[:40]}")