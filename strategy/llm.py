"""Local Qwen adapter.

The allocation model on disk sets the weights. Qwen, when a local server is
reachable, may only write a short note. This machine cannot load Qwen3.8-27B.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from strategy.universe import QWEN_BASE_URL, QWEN_MODEL


def hardware() -> dict[str, str | bool]:
    mem_kb = 0
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemTotal:"):
                mem_kb = int(line.split()[1])
                break
    except OSError:
        mem_kb = 0
    gpu = os.path.exists("/dev/nvidia0") or os.path.exists("/proc/driver/nvidia/version")
    return {
        "memoryGb": f"{mem_kb / 1024 / 1024:.1f}" if mem_kb else "unknown",
        "gpu": gpu,
    }


def qwen_config() -> tuple[str, str]:
    return os.environ.get("QWEN_BASE_URL", QWEN_BASE_URL), os.environ.get("QWEN_MODEL", QWEN_MODEL)


def qwen_status() -> dict:
    base_url, model = qwen_config()
    reachable = False
    detail = "No local server answered."
    try:
        request = urllib.request.Request(base_url.rstrip("/") + "/models", method="GET")
        with urllib.request.urlopen(request, timeout=0.4) as response:
            reachable = response.status == 200
            detail = "Local server responded."
    except Exception as error:
        detail = str(error.__class__.__name__)
    facts = hardware()
    if facts["gpu"] or (facts["memoryGb"] != "unknown" and float(facts["memoryGb"]) >= 24):
        capacity = "This machine may be able to host a quantized 27B model."
    else:
        capacity = (
            f"This machine has {facts['memoryGb']} GB of RAM and no GPU, so Qwen3.8-27B is not loaded. "
            "Host it on your own hardware and set QWEN_BASE_URL. Weights still come from the local allocation model."
        )
    return {
        "model": model,
        "baseUrl": base_url,
        "reachable": reachable,
        "setsWeights": False,
        "hardware": facts,
        "detail": detail,
        "message": capacity,
    }


def explain(state: dict) -> str | None:
    """Ask the local model for a note. Returns None when it is not hosted here."""
    status = qwen_status()
    if not status["reachable"]:
        return None
    base_url, model = qwen_config()
    instruction = state.get("instruction", [])
    lines = [
        f"{row['name']} ({row['ticker']}) {row['weight'] * 100:.1f}%"
        for row in instruction
        if row.get("weight", 0) >= 0.01
    ]
    prompt = (
        "You are writing a note for the account owner to hand to a financial advisor. "
        "Use only the facts below. Do not invent numbers, do not change the weights, "
        "and do not give a different allocation. Four sentences at most.\n\n"
        f"Action: {state.get('action', {}).get('label')}\n"
        f"As of: {state.get('asOf')}\n"
        f"Weights:\n" + "\n".join(lines) + "\n"
        f"Reasons: {'; '.join(state.get('action', {}).get('reasons') or []) or 'None'}\n"
    )
    body = json.dumps(
        {
            "model": model,
            "temperature": 0.2,
            "max_tokens": 280,
            "messages": [
                {"role": "system", "content": "You explain an allocation you did not choose. You never output new weights."},
                {"role": "user", "content": prompt},
            ],
        }
    ).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    text = " ".join(str(text).split())
    return text[:1200] if text else None
