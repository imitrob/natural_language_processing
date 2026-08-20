"""Standalone web visualizer for live speech-to-text predictions."""

import argparse
import json
import math
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_TOPIC = "/nlp/whisper"
SPEECH_ACTIVE_TOPIC = "/nlp/speech_active"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _probability(value):
    """Return a finite probability suitable for display."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def prediction_payload(text, words_json, stamp=None):
    """Turn WhisperText fields into the small model used by the page.

    Malformed or absent word metadata gracefully falls back to the plain
    transcript. Alternatives are sorted by likelihood and never repeat the
    selected word.
    """
    try:
        raw_words = json.loads(words_json) if words_json else []
    except (json.JSONDecodeError, TypeError):
        raw_words = []
    if not isinstance(raw_words, list):
        raw_words = []

    words = []
    for raw_word in raw_words:
        if not isinstance(raw_word, dict):
            continue
        selected = raw_word.get("word", "")
        if not isinstance(selected, str) or not selected.strip():
            continue
        selected = selected.strip()

        raw_alternatives = raw_word.get("alts", {})
        if not isinstance(raw_alternatives, dict):
            raw_alternatives = {}

        confidence = None
        alternatives = []
        for candidate, likelihood in raw_alternatives.items():
            candidate = str(candidate).strip()
            if not candidate:
                continue
            likelihood = _probability(likelihood)
            if candidate.casefold() == selected.casefold():
                confidence = likelihood
            else:
                alternatives.append({
                    "word": candidate,
                    "probability": likelihood,
                })
        alternatives.sort(key=lambda item: item["probability"], reverse=True)
        words.append({
            "word": selected,
            "confidence": confidence,
            "alternatives": alternatives,
        })

    text = str(text or "").strip()
    if text and (not words or len(words) != len(raw_words)):
        words = [{"word": word, "confidence": None, "alternatives": []}
                 for word in text.split()]
    elif words:
        # Build the sentence from the same records that own the alternatives.
        text = " ".join(word["word"] for word in words)

    return {"text": text, "words": words, "stamp": stamp}


class PredictionState:
    """Thread-safe latest-value handoff between ROS and SSE clients.

    The prediction and the listening flag live in their own slots and every
    event carries both: a client that misses the coalesced middle of a burst
    still ends up with the whole truth.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._revision = 0
        self._payload = None
        self._listening = False

    def publish(self, payload):
        with self._condition:
            self._revision += 1
            self._payload = payload
            self._condition.notify_all()
            return self._revision

    def set_listening(self, listening):
        with self._condition:
            self._revision += 1
            self._listening = bool(listening)
            self._condition.notify_all()
            return self._revision

    def wait_after(self, revision, timeout):
        with self._condition:
            self._condition.wait_for(
                lambda: self._revision > revision,
                timeout=timeout,
            )
            snapshot = dict(self._payload or {})
            snapshot["listening"] = self._listening
            return self._revision, snapshot

    def resume_revision(self, revision):
        """Reset an impossible EventSource cursor after a server restart."""
        with self._condition:
            if revision < 0 or revision > self._revision:
                return 0
            return revision


class VisualizationServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, page, state):
        self.page = page
        self.state = state
        super().__init__(address, VisualizationHandler)


class VisualizationHandler(BaseHTTPRequestHandler):
    server_version = "STTVisualizer/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802 -- BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._send_page()
        elif path == "/events":
            self._send_events()
        elif path == "/health":
            self._send_json({"status": "ok"})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _send_page(self):
        body = self.server.page
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, value):
        body = json.dumps(value).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_events(self):
        try:
            revision = int(self.headers.get("Last-Event-ID", "0"))
        except ValueError:
            revision = 0
        revision = self.server.state.resume_revision(revision)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            while True:
                next_revision, payload = self.server.state.wait_after(
                    revision, timeout=15.0)
                if next_revision <= revision or payload is None:
                    # Advance anyway: a revision that never gets consumed would
                    # spin this loop instead of waiting for the next one.
                    revision = next_revision
                    self.wfile.write(b": keep-alive\n\n")
                else:
                    body = json.dumps(payload, ensure_ascii=False,
                                      separators=(",", ":"))
                    event = f"id: {next_revision}\ndata: {body}\n\n"
                    self.wfile.write(event.encode("utf-8"))
                    revision = next_revision
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, _format, *_args):
        # A reconnecting EventSource would otherwise make the ROS console noisy.
        pass


def _page_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        installed = Path(get_package_share_directory(
            "natural_language_processing")) / "web" / "stt_visualizer.html"
        if installed.is_file():
            return installed
    except (ImportError, LookupError):
        pass
    return Path(__file__).resolve().parents[1] / "resource" / "stt_visualizer.html"


def _message_stamp(message):
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return None
    return stamp.sec + stamp.nanosec / 1_000_000_000


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize /nlp/whisper word alternatives in a browser")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="HTTP bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="HTTP port (default: %(default)s)")
    parser.add_argument("--topic", default=DEFAULT_TOPIC,
                        help="WhisperText topic (default: %(default)s)")
    args, ros_args = parser.parse_known_args(argv)

    import rclpy
    from hri_msgs.msg import WhisperText
    from rclpy.node import Node
    from std_msgs.msg import Bool

    page = _page_path().read_bytes()
    state = PredictionState()
    server = VisualizationServer((args.host, args.port), page, state)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    rclpy.init(args=ros_args)
    node = Node("speech_to_text_visualizer")

    def on_prediction(message):
        text = message.new_text or message.all_text
        state.publish(prediction_payload(
            text, message.words_json, stamp=_message_stamp(message)))

    node.create_subscription(WhisperText, args.topic, on_prediction, 10)
    node.create_subscription(
        Bool, SPEECH_ACTIVE_TOPIC,
        lambda message: state.set_listening(message.data), 10)
    shown_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"Speech visualizer listening to {args.topic}", flush=True)
    print(f"Open http://{shown_host}:{args.port}", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
