import json
import re
from pathlib import Path

import numpy as np
import torch


_OPAQUE = re.compile(r"(chatcmpl|call_|toolu_|[0-9a-f]{12,})", re.I)
_GENERIC = {"assistant", "user", "tool", "system", "environment", "human", ""}
_MARKER = re.compile(r"^\[(Thought|Action|Content|Reasoning)\][ \t]*", re.M)
_TOOL_MARKER = "tool-role-probe"
_TOOL_CACHE = {}


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Line {number} must contain a JSON object")
                yield value


def load_trajectories(path, require_outcomes=True):
    records = {}
    for record in _jsonl(path):
        tid = str(record["trajectory_id"])
        if tid in records:
            raise ValueError(f"Duplicate trajectory_id: {tid}")
        record["trajectory_id"] = tid
        if require_outcomes and "parent_group_id" not in record:
            raise ValueError("Training trajectories require parent_group_id")
        record["parent_group_id"] = str(record.get("parent_group_id", tid))
        if (require_outcomes or "outcome" in record) and record.get("outcome") not in (0, 1):
            raise ValueError("outcome must be 0 for success or 1 for failure")
        events = record.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError("Each trajectory must contain a nonempty events list")
        if not isinstance(record.get("public_context", ""), str):
            raise ValueError("public_context must be a string")
        for index, event in enumerate(events, 1):
            if not isinstance(event, dict):
                raise ValueError("Each event must be a JSON object")
            if event.get("event_idx", index) != index:
                raise ValueError("event_idx must be contiguous and one-based")
            if event.get("done", False) and index != len(events):
                raise ValueError("Only the terminal event can have done=true")
            for key in ("actor_role", "actor_id", "event_type", "action_text", "observation_text"):
                if key in event and not isinstance(event[key], str):
                    raise ValueError(f"Event {key} must be a string")
        records[tid] = record
    if not records:
        raise ValueError("No trajectories were loaded")
    return records


def load_prefix_index(path, trajectories, require_outcomes=True):
    result, seen = [], set()
    for item in _jsonl(path):
        tid, t = str(item["trajectory_id"]), item["prefix_t"]
        if tid not in trajectories:
            raise ValueError(f"Unknown trajectory_id: {tid}")
        if isinstance(t, bool) or not isinstance(t, int) or not 1 <= t <= len(trajectories[tid]["events"]):
            raise ValueError("prefix_t must be an integer between 1 and trajectory length")
        if (tid, t) in seen:
            raise ValueError(f"Duplicate prefix: {(tid, t)}")
        seen.add((tid, t))
        record = trajectories[tid]
        row = {"trajectory_id": tid, "prefix_t": t,
               "parent_group_id": record["parent_group_id"]}
        if "outcome" in record:
            row["outcome"] = int(record["outcome"])
        elif require_outcomes:
            raise ValueError("Training prefixes require terminal outcomes")
        result.append(row)
    if not result:
        raise ValueError("The training-prefix index is empty")
    return result


def _prefix_events(record, t):
    if isinstance(t, bool) or not isinstance(t, int) or not 1 <= t <= len(record["events"]):
        raise ValueError("prefix_t must be an integer between 1 and trajectory length")
    return record["events"][:t]


def _unmark(text):
    parts = _MARKER.split(text)
    if len(parts) == 1:
        return text.strip(), False
    segments = [parts[0].strip()] if parts[0].strip() else []
    for index in range(1, len(parts) - 1, 2):
        tag, body = parts[index], parts[index + 1].strip()
        if body:
            segments.append(body if tag == "Content" else f"{tag}: {body}")
    return "\n\n".join(segments), True


def _event_content(event):
    kind = event.get("event_type", "message")
    action, marked = _unmark(event.get("action_text", "").strip())
    observation = event.get("observation_text", "").strip()
    if kind == "tool_call":
        return (action if marked else f"Action: {action}") if action else "Action: (none recorded)"
    if kind in ("tool_result", "observation"):
        body = observation or action
        return f"Tool result: {body}" if body else "Tool result: (empty)"
    if kind == "control":
        return "The run has ended." if event.get("done", False) else "Control event."
    return "\n\n".join(value for value in (action, observation) if value) or "(no content)"


def _chat_role(event):
    kind, actor = event.get("event_type", "message"), event.get("actor_role", "assistant")
    if kind == "tool_result":
        return "tool"
    if kind == "observation":
        return "assistant" if actor == "assistant" else "tool"
    if kind == "tool_call":
        return "assistant"
    if kind == "control":
        return "system"
    return actor if actor in ("user", "system") else "assistant"


def _speaker_name(event):
    raw = event.get("actor_id", "").strip()
    name = raw.split(":", 1)[-1] if raw.startswith("tool:") else raw
    return None if not name or name.lower() in _GENERIC or _OPAQUE.search(name) else name[:64]


def _keeps_tool_role(tokenizer):
    key = id(tokenizer)
    if key not in _TOOL_CACHE:
        messages = [{"role": "system", "content": "check"},
                    {"role": "user", "content": "check"},
                    {"role": "tool", "content": _TOOL_MARKER + ' [{"k": "v"}]'}]
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            _TOOL_CACHE[key] = _TOOL_MARKER in (rendered or "")
        except Exception:
            _TOOL_CACHE[key] = False
    return _TOOL_CACHE[key]


def serialize_prefix(tokenizer, record, t):
    events = _prefix_events(record, t)
    header = record.get("public_context", "").strip()
    first = " ".join(_event_content(events[0]).split()).lower()
    if header and first and " ".join(header.split()).lower() in first:
        header = ""
    framework = record.get("framework_description", "").strip()
    if framework:
        header = f"{framework}\n\n{header}" if header else framework
    messages = [{"role": "system", "content": header}] if header else []
    keeps_tool = _keeps_tool_role(tokenizer)
    for event in events:
        role, content, name = _chat_role(event), _event_content(event), _speaker_name(event)
        if role == "tool" and not keeps_tool:
            role, content = "user", f"[TOOL RESULT]\n{content}"
        if name and not content.startswith(f"{name}:"):
            content = f"{name}: {content}"
        message = {"role": role, "content": content}
        if name:
            message["name"] = name
        messages.append(message)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def encode_batch(tokenizer, records, index_rows, device, max_length=2048):
    if not index_rows:
        raise ValueError("Cannot encode an empty batch")
    if max_length <= 256:
        raise ValueError("max_length must exceed the 256-token retained head")
    token_lists = []
    for row in index_rows:
        text = serialize_prefix(tokenizer, records[row["trajectory_id"]], int(row["prefix_t"]))
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > max_length:
            ids = ids[:256] + ids[-(max_length - 256):]
        if not ids:
            raise ValueError("A serialized prefix produced zero tokens")
        token_lists.append(ids)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("The tokenizer requires a padding or EOS token")
    width = max(map(len, token_lists))
    input_ids = torch.full((len(index_rows), width), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, ids in enumerate(token_lists):
        input_ids[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[index, :len(ids)] = 1
    result = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
    if all("outcome" in records[row["trajectory_id"]] for row in index_rows):
        labels = torch.tensor([records[row["trajectory_id"]]["outcome"] for row in index_rows], dtype=torch.float32)
        result["labels"] = labels.to(device)
    return result


class BalancedSampler:
    def __init__(self, index):
        self.by_class = {0: {}, 1: {}}
        self.index = index
        trajectory_labels = {}
        for row in index:
            tid, group, label = row["trajectory_id"], row["parent_group_id"], int(row["outcome"])
            if label not in (0, 1):
                raise ValueError("Sampler labels must be binary")
            previous = trajectory_labels.setdefault(tid, (label, group))
            if previous != (label, group):
                raise ValueError("A trajectory must have one outcome and parent group")
            self.by_class[label].setdefault(group, {}).setdefault(tid, []).append(row)
        if any(not value for value in self.by_class.values()):
            raise ValueError("Training requires both outcome classes")

    def sample(self, rng, n=32):
        if n <= 0 or n % 2:
            raise ValueError("Logical batch size must be positive and even")
        labels = np.repeat(np.asarray([0, 1], dtype=np.int64), n // 2)
        rng.shuffle(labels)
        result = []
        for label in labels:
            groups = self.by_class[int(label)]
            group_keys = tuple(sorted(groups))
            group = group_keys[int(rng.integers(0, len(group_keys)))]
            trajectory_keys = tuple(sorted(groups[group]))
            tid = trajectory_keys[int(rng.integers(0, len(trajectory_keys)))]
            rows = groups[group][tid]
            probabilities = np.full(len(rows), 1.0 / len(rows))
            result.append(rows[int(rng.choice(len(rows), p=probabilities))])
        return result
