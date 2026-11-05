"""A fake of Gemini's batchEmbedContents, standard library only: the acceptance
suites' embedding provider (P7). tests/support/acceptance.py scripts it in-process;
the compose stack (tests/compose) runs this file as a service that always answers:

    python fake_gemini.py --host 0.0.0.0 --port 8080 --model M --dimensions D
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass(frozen=True)
class Reply:
    """How the fake answers one request. ``hold``: answer only once it is set (or
    the fake closes) — a provider still working, or one that hangs."""

    status: int = 200
    hold: threading.Event | None = None


@dataclass(frozen=True)
class ProviderRequest:
    path: str
    api_key: str | None
    texts: int


class FakeGemini:
    """Gemini's batchEmbedContents on a loopback port. Answers 200 with a vector per
    text unless scripted otherwise; records every request."""

    def __init__(
        self, model: str, dimensions: int, host: str = "127.0.0.1", port: int = 0
    ) -> None:
        self.requests: list[ProviderRequest] = []
        self._path = f"/v1beta/models/{model}:batchEmbedContents"
        self._dimensions = dimensions
        self._script: deque[Reply] = deque()
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._server = ThreadingHTTPServer((host, port), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1beta"

    def script(self, *replies: Reply) -> None:
        """The next requests get these replies, in order; later ones get a 200."""
        with self._lock:
            self._script.extend(replies)

    def close(self) -> None:
        self._closed.set()  # releases every held request
        self._server.shutdown()
        self._server.server_close()

    def _next(self, request: ProviderRequest) -> Reply:
        with self._lock:
            self.requests.append(request)
            return self._script.popleft() if self._script else Reply()

    def _vectors(self, texts: int) -> dict[str, Any]:
        return {
            "embeddings": [
                {"values": [(i % 10 + 1) / 10 for i in range(self._dimensions)]}
                for _ in range(texts)
            ]
        }

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — http.server's name
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                texts = len(body.get("requests", []))
                reply = fake._next(
                    ProviderRequest(
                        self.path, self.headers.get("x-goog-api-key"), texts
                    )
                )
                if self.path != fake._path:
                    self._send(404, {"error": {"message": f"no route {self.path}"}})
                    return
                if reply.hold is not None:
                    while not (reply.hold.is_set() or fake._closed.is_set()):
                        reply.hold.wait(0.05)
                if reply.status != 200:
                    self._send(reply.status, {"error": {"code": reply.status}})
                    return
                self._send(200, fake._vectors(texts))

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except BrokenPipeError, ConnectionResetError:
                    pass  # the worker gave up on this request (timeout, kill)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dimensions", type=int, required=True)
    args = parser.parse_args()
    fake = FakeGemini(args.model, args.dimensions, host=args.host, port=args.port)
    sys.stdout.write(f"fake gemini on {fake.url}\n")
    sys.stdout.flush()
    threading.Event().wait()


if __name__ == "__main__":
    main()
