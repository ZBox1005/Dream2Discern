import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from model import Auditor


BACKBONES = {
    "qwen3-4b": {"model_id": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c", "gamma": 0.0003},
    "qwen3-8b": {"model_id": "Qwen/Qwen3-8B", "revision": None, "gamma": 0.003},
    "llama3.2-3b": {"model_id": "meta-llama/Llama-3.2-3B-Instruct", "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95", "gamma": 0.003},
    "llama3.1-8b": {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "gamma": 0.0003},
}
STEPS = {"aftraj": 1057, "stepshield": 979}


def configure_numerics(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_auditor(config, device, model_path=None):
    source = model_path or config["model_id"]
    revision = None if model_path else config.get("revision")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config["precision"]]
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    if not tokenizer.chat_template:
        raise ValueError("The backbone tokenizer must provide its chat template")
    auditor = Auditor.from_pretrained(source, revision=revision, dtype=dtype).to(device)
    auditor.eval()
    return auditor, tokenizer


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "dream2discern-v1":
        raise ValueError("Unsupported checkpoint format")
    return checkpoint


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
