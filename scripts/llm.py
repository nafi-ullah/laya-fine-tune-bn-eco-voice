"""Gemini client (REST) used to restructure / enrich / generate the dataset.

JSON output mode, minimal thinking, retries with backoff on 429/5xx, token accounting.
Thread-safe: 02_llm_enrich.py calls `chat` from a thread pool.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

import httpx

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def extract_json(text: str):
    """Parse a JSON reply (tolerates ``` fences and chatter around the JSON)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in reply")
    start = min(starts)
    end = text.rfind("]" if text[start] == "[" else "}")
    if end <= start:
        raise ValueError("unterminated JSON in reply")
    return json.loads(text[start:end + 1])


class GeminiLLM:
    def __init__(self, cfg: dict):
        self.model = cfg["model"]
        self.thinking_level = cfg.get("thinking_level", "minimal")
        self.max_attempts = cfg.get("http_attempts", 6)
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise SystemExit("GEMINI_API_KEY not set (expected in ./.env)")
        self.client = httpx.Client(timeout=120)
        self._lock = threading.Lock()
        self.calls = self.in_tokens = self.out_tokens = 0
        self.t0 = time.perf_counter()

    def chat(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens,
                                 "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingLevel": self.thinking_level}},
        }
        delay = 2.0
        for attempt in range(self.max_attempts):
            try:
                r = self.client.post(_URL.format(model=self.model), json=body,
                                     headers={"x-goog-api-key": self.key})
            except httpx.TransportError:
                r = None
            if r is not None and r.status_code == 200:
                d = r.json()
                u = d.get("usageMetadata", {})
                with self._lock:
                    self.calls += 1
                    self.in_tokens += u.get("promptTokenCount", 0)
                    self.out_tokens += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
                try:
                    return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                except (KeyError, IndexError):
                    return ""                                  # blocked / empty → caller retries item
            if r is not None and r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"Gemini {r.status_code}: {r.text[:300]}")
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise RuntimeError("Gemini: retries exhausted")

    def usage(self) -> str:
        el = time.perf_counter() - self.t0
        return f"{self.calls} calls · {self.in_tokens:,} in / {self.out_tokens:,} out tokens · {el/60:.1f} min"
