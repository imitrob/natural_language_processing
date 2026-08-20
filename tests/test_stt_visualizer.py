import json
import threading
from http.client import HTTPConnection
from types import SimpleNamespace

import pytest

from natural_language_processing.stt_visualizer import (
    PredictionState,
    VisualizationServer,
    _message_stamp,
    prediction_payload,
)


def test_prediction_payload_sorts_alternatives_and_excludes_winner():
    words_json = json.dumps([{
        "word": "pick",
        "alts": {"push": 0.12, "pick": 0.72, "peek": 0.31},
    }])

    payload = prediction_payload("pick", words_json, stamp=12.5)

    assert payload == {
        "text": "pick",
        "stamp": 12.5,
        "words": [{
            "word": "pick",
            "confidence": 0.72,
            "alternatives": [
                {"word": "peek", "probability": 0.31},
                {"word": "push", "probability": 0.12},
            ],
        }],
    }


def test_prediction_payload_falls_back_to_text_when_metadata_is_bad():
    payload = prediction_payload("pick the cup", "not json")

    assert [word["word"] for word in payload["words"]] == ["pick", "the", "cup"]
    assert all(word["alternatives"] == [] for word in payload["words"])


def test_prediction_payload_does_not_truncate_text_for_one_bad_word_record():
    words_json = json.dumps([
        {"word": "pick", "alts": {"pick": 0.8}},
        {"word": None, "alts": {}},
        {"word": "cup", "alts": {"cup": 0.7}},
    ])

    payload = prediction_payload("pick the cup", words_json)

    assert [word["word"] for word in payload["words"]] == ["pick", "the", "cup"]


def test_prediction_payload_clamps_invalid_display_probabilities():
    words_json = json.dumps([{
        "word": "cup",
        "alts": {"cup": 1.4, "cap": -0.2, "cop": "not-a-number"},
    }])

    word = prediction_payload("cup", words_json)["words"][0]

    assert word["confidence"] == 1.0
    assert [alternative["probability"] for alternative in word["alternatives"]] == [0.0, 0.0]


def test_prediction_state_gives_each_publication_a_revision():
    state = PredictionState()

    revision = state.publish({"text": "stop"})

    assert revision == 1
    assert state.wait_after(0, timeout=0) == (1, {"text": "stop", "listening": False})
    assert state.resume_revision(1) == 1
    assert state.resume_revision(999) == 0


def test_prediction_state_keeps_the_prediction_when_listening_toggles():
    """A burst of prediction-then-idle must not coalesce the words away."""
    state = PredictionState()
    state.set_listening(True)

    assert state.wait_after(0, timeout=0) == (1, {"listening": True})

    state.publish({"text": "stop"})
    state.set_listening(False)

    # One coalesced event, still carrying the accepted words.
    assert state.wait_after(1, timeout=0) == (3, {"text": "stop", "listening": False})


def test_listening_clears_before_the_first_prediction():
    """Speech ignored for missing the wake phrase leaves nothing to show, but
    the page must still be told the microphone went idle."""
    state = PredictionState()
    state.set_listening(True)
    state.set_listening(False)

    assert state.wait_after(1, timeout=0) == (2, {"listening": False})

def test_message_stamp_uses_ros_seconds_and_nanoseconds():
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=250_000_000)))

    assert _message_stamp(message) == 12.25


def test_web_server_serves_page_health_and_prediction_event():
    state = PredictionState()
    try:
        server = VisualizationServer(("127.0.0.1", 0), b"<html>speech</html>", state)
    except PermissionError:
        pytest.skip("test environment does not permit local sockets")
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    connection = HTTPConnection(*server.server_address, timeout=2)

    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"status": "ok"}

        connection.request("GET", "/events")
        response = connection.getresponse()
        assert response.getheader("Content-Type") == "text/event-stream"
        state.publish({"text": "pick cup"})
        assert response.readline() == b"id: 1\n"
        assert json.loads(response.readline().removeprefix(b"data: ")) == {
            "text": "pick cup",
            "listening": False,
        }
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
