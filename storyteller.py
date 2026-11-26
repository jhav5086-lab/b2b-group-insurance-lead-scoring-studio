"""LLM as storyteller, with a safety net.

The LLM only rewrites text that Python already wrote from exact numbers. Every number in its answer must appear in
those facts (rounding allowed). If it adds a number, returns nothing, or the API fails, the computed text is used.
Only the aggregate sentences are sent to the API, never account rows.
"""
import re

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")

SYSTEM = ("You are a RevOps analyst writing for a VP of Sales. You receive FACTS that were calculated in Python. "
          "Rewrite them as exactly two sentences of insight followed by exactly one sentence of recommendation. "
          "Use ONLY numbers that appear in FACTS. Never add, estimate or round to a new number. Plain business English, no jargon. "
          "Answer in this exact format:\nINSIGHT: <two sentences>\nRECOMMENDATION: <one sentence>")


def _numbers(text):
    out = []
    for n in NUM.findall(text or ""):
        try:
            out.append(float(n.replace(",", "")))
        except ValueError:
            pass
    return out


def allowed_numbers(facts):
    allowed = set()
    for n in _numbers(facts):
        allowed |= {n, round(n), round(n, 1), round(n, 2)}
    return allowed


def unverified_numbers(text, facts):
    allowed = allowed_numbers(facts)
    return sorted({n for n in _numbers(text) if n not in allowed and round(n, 1) not in allowed})


def parse(text):
    m = re.search(r"INSIGHT:\s*(.+?)\s*RECOMMENDATION:\s*(.+)", text or "", flags=re.S | re.I)
    if not m:
        return None
    ins, act = " ".join(m.group(1).split()), " ".join(m.group(2).split())
    return (ins, act) if ins and act else None


def polish(client, model, title, insight, action, max_tokens=350):
    """Returns dict(insight, action, status, note). status is 'ai' only when every number checked out."""
    base = {"insight": insight, "action": action, "status": "computed", "note": ""}
    facts = f"Chart: {title}\nInsight: {insight}\nAction: {action}"
    try:
        extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
        resp = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=max_tokens, **extra,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"FACTS:\n{facts}"}])
        text = resp.choices[0].message.content or ""
    except Exception as e:
        msg = str(e).lower()
        base["note"] = "rate limit" if ("429" in msg or "rate" in msg) else "api error"
        return base
    parsed = parse(text)
    if parsed is None:
        base["note"] = "unexpected format"
        return base
    bad = unverified_numbers(" ".join(parsed), facts)
    if bad:
        base["note"] = f"rejected: new numbers {bad[:3]}"
        return base
    return {"insight": parsed[0], "action": parsed[1], "status": "ai", "note": ""}
