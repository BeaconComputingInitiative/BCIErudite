# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
"""Local, bounded inference HTTP process; training is a separate command."""
import json
import logging
import socket
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from core.config import runtime
from core.guardrails.validation import POLICY, messages
from core.inference.engine import Engine
from core.telemetry import metrics
CONFIG = runtime(); ENGINE = Engine(CONFIG); GATE = threading.BoundedSemaphore(1)
LOGGER = logging.getLogger("Erudite")

class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, handler):
        super().__init__(address, handler); self.slots = threading.BoundedSemaphore(POLICY["max_http_connections"])
    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            # Reject overload without spawning another request worker or waiting queue.
            request.close(); return
        try: super().process_request(request, client_address)
        except BaseException:
            self.slots.release(); raise
    def process_request_thread(self, request, client_address):
        try: super().process_request_thread(request, client_address)
        finally: self.slots.release()

class Handler(BaseHTTPRequestHandler):
    def setup(self): super().setup(); self.connection.settimeout(10)
    def log_message(self, *_): pass
    def origin_allowed(self):
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        accepted_hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        return host in accepted_hosts and origin in {None, f"http://localhost:{CONFIG['ui_port']}", f"http://127.0.0.1:{CONFIG['ui_port']}"}
    def response(self, status, body):
        raw = json.dumps(body).encode(); self.send_response(status)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close"); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if not self.origin_allowed(): return self.response(403, {"error": "Origin/Host rejected"})
        if self.path == "/health": return self.health()
        if self.path == "/telemetry":
            try: return self.response(200, metrics.summary())
            except (sqlite3.Error, OSError): return self.response(503, {"error": "Telemetry unavailable"})
        return self.response(404, {"error": "Unknown endpoint"})
    def health(self):
        free = GATE.acquire(blocking=False)
        if free: GATE.release()
        self.response(200, {"api": "ready", "model": "Erudite", "checkpoint_loaded": ENGINE.model is not None,
                            "generation_busy": not free, "checkpoint_revision": ENGINE.revision})
    def do_POST(self):
        if not self.origin_allowed(): return self.response(403, {"error": "Origin/Host rejected"})
        if self.path == "/cancel":
            ENGINE.cancel.set(); return self.response(200, {"status": "cancellation_requested"})
        if self.path != "/chat": return self.response(404, {"error": "Unknown endpoint"})
        if self.headers.get("Transfer-Encoding") or self.headers.get_content_type() != "application/json":
            return self.response(415, {"error": "Expected fixed-length JSON"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= POLICY["max_request_bytes"]: raise ValueError("Request size out of bounds")
            body = self.rfile.read(length)
            if len(body) != length: raise ValueError("Incomplete body")
            payload = json.loads(body)
            if not isinstance(payload, dict) or set(payload) != {"messages"}: raise ValueError("Expected messages only")
            history = messages(payload["messages"], generation=True)
            if history[0]["role"] == "system": raise ValueError("Client cannot set system prompt")
        except (ValueError, TypeError, TimeoutError): return self.response(400, {"error": "Invalid conversation/body"})
        if not GATE.acquire(blocking=False): return self.response(429, {"error": "Erudite is generating; retry afterwards"})
        started = time.monotonic(); tokens = 0; state = "error"; stream = None; headers_sent = False
        try:
            stream = ENGINE.stream(history)
            first = next(stream)  # Load and validate checkpoint/context before sending success headers.
            self.send_response(200); self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store"); self.send_header("Connection", "close"); self.end_headers(); headers_sent = True
            self.event(first)
            for event in stream:
                self.event(event)
                if event["event"] == "done": state = "ok"; tokens = event["generated_tokens"]
                elif event["event"] == "cancelled": state = "cancelled"
        except (BrokenPipeError, ConnectionResetError, TimeoutError): state = "disconnected"; ENGINE.cancel.set()
        except Exception as error:
            LOGGER.exception("Generation failed")
            if headers_sent:
                try: self.event({"event": "error", "error": str(error)})
                except OSError: pass
            else: self.response(503, {"error": str(error)})
        finally:
            try:
                if stream is not None: stream.close()
            finally:
                # Logging is not part of neural-model ownership. Release first so
                # a slow/failed SQLite writer cannot stall the next generation.
                GATE.release()
                try: metrics.record(state, time.monotonic() - started, tokens)
                except (sqlite3.Error, OSError): LOGGER.exception("Metadata logging failed")
    def event(self, event):
        self.wfile.write((json.dumps(event) + "\n").encode()); self.wfile.flush()

def run(checkpoint=None):
    if checkpoint is not None: CONFIG["checkpoint"] = checkpoint
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)")
    server = Server(("127.0.0.1", CONFIG["api_port"]), Handler)
    try: server.serve_forever()
    except KeyboardInterrupt: ENGINE.cancel.set()
    finally: server.server_close()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", help="Native checkpoint pointer JSON")
    args = parser.parse_args(); run(args.checkpoint)
