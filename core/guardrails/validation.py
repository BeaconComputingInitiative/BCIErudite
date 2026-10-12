# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
from core.config import ROOT, read
POLICY = read(ROOT / "config/guardrails.yaml")

def messages(items, generation=False):
    if not isinstance(items, list) or not 1 <= len(items) <= POLICY["max_messages"]:
        raise ValueError("Expected nonempty bounded conversation")
    cleaned = []; last = None; total = 0
    for index, item in enumerate(items):
        if not isinstance(item, dict): raise ValueError("Message must be an object")
        role, content = item.get("role"), item.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > POLICY["max_message_chars"]:
            raise ValueError("Empty/non-text/overlength content")
        if role == "system":
            if index != 0: raise ValueError("System message must be first")
        elif role not in {"user", "assistant"} or (last is None and role != "user") or (last is not None and role == last):
            raise ValueError("Conversation must alternate user/assistant")
        else: last = role
        total += len(content)
        cleaned.append({"role": role, "content": content})
    # Accept iff schema, ordering, final role and aggregate size all satisfy the contract.
    expected = "user" if generation else "assistant"
    if last != expected or total > POLICY["max_total_chars"]: raise ValueError(f"Conversation must end with {expected} and fit size limits")
    return cleaned

def record(item):
    if not isinstance(item, dict): raise ValueError("Dataset record must be an object")
    if "text" in item:
        if not isinstance(item["text"], str) or not item["text"].strip(): raise ValueError("Empty/non-text document")
        return {"text": item["text"].strip()}
    if "messages" in item: return {"messages": messages(item["messages"])}
    if "instruction" in item and "response" in item:
        return {"messages": messages([{"role": "user", "content": item["instruction"]}, {"role": "assistant", "content": item["response"]}])}
    raise ValueError("Expected text, messages, or instruction/response; arbitrary logs are not answers")
