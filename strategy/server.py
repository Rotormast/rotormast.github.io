"""Dashboard server. Binds immediately and trains in the background."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from strategy.data import ROOT, load_market
from strategy.engine import run_desk
from strategy.llm import explain, qwen_status
from strategy.memo import html_memo, plain_text

STATIC = Path(__file__).resolve().parent / "static"
MODEL_DIR = ROOT / "data" / "model"


class Desk:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state: dict = {"phase": "loading", "message": "Training the allocation model on daily closes."}
        self.narrative: str | None = None
        self.computing = False

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    def compute(self, refresh: bool = False) -> None:
        with self.lock:
            if self.computing:
                return
            self.computing = True
            if self.state.get("phase") == "ready":
                self.state = {**self.state, "phase": "refreshing"}
        try:
            prices, macro, warning = load_market(refresh=refresh)
            state = run_desk(prices, macro, MODEL_DIR)
            state["qwen"] = qwen_status()
            if warning:
                state["warning"] = warning
            with self.lock:
                self.state = state
                self.narrative = None
        except Exception as error:
            traceback.print_exc()
            with self.lock:
                previous = self.state if self.state.get("phase") == "ready" else None
                if previous:
                    self.state = {**previous, "warning": f"Refresh failed: {error}"}
                else:
                    self.state = {"phase": "error", "message": str(error)}
        finally:
            with self.lock:
                self.computing = False


def _finite(value):
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


DESK = Desk()


class Handler(BaseHTTPRequestHandler):
    server_version = "ResearchDeck/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[desk] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(_finite(payload), allow_nan=False).encode()
        self._send(code, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        state = DESK.snapshot()
        if path == "/api/health":
            self._json({"ok": state.get("phase") != "error", "phase": state.get("phase")})
            return
        if path == "/api/state":
            self._json(state)
            return
        if path == "/api/narrative":
            if state.get("phase") != "ready":
                self._json({"available": False, "text": None})
                return
            if DESK.narrative is None:
                DESK.narrative = explain(state) or ""
            self._json({"available": bool(DESK.narrative), "text": DESK.narrative or None})
            return
        if path in {"/memo", "/memo.txt"}:
            if state.get("phase") not in {"ready", "refreshing"}:
                self._send(503, b"The desk is still training.\n", "text/plain; charset=utf-8")
                return
            if path == "/memo.txt":
                self._send(200, plain_text(state).encode(), "text/plain; charset=utf-8")
            else:
                self._send(200, html_memo(state, DESK.narrative).encode(), "text/html; charset=utf-8")
            return
        if path in {"/", "/index.html"}:
            self._file(STATIC / "index.html")
            return
        if path.startswith("/static/"):
            self._file(STATIC / path.removeprefix("/static/"))
            return
        self._send(404, b"Not found\n", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/refresh":
            self._send(404, b"Not found\n", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length:
            self.rfile.read(min(length, 1_000_000))
        threading.Thread(target=DESK.compute, kwargs={"refresh": True}, daemon=True).start()
        self._json({"ok": True, "phase": "refreshing"}, 202)

    def _file(self, path: Path) -> None:
        resolved = path.resolve()
        root = STATIC.resolve()
        if not resolved.is_file() or root not in resolved.parents:
            self._send(404, b"Not found\n", "text/plain; charset=utf-8")
            return
        mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        if mime.startswith("text/") or mime.endswith("javascript"):
            mime = f"{mime}; charset=utf-8"
        self._send(200, resolved.read_bytes(), mime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Research Deck index desk")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    threading.Thread(target=DESK.compute, kwargs={"refresh": False}, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Index desk at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
