import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from natural_language_processing.speech_to_text.auto_stt import (
    AutoSpeechToTextNode,
    UtteranceSegmenter,
    wake_command,
)
from natural_language_processing.speech_to_text.stt_node import SpeechToTextNode


def test_wake_command_is_case_and_punctuation_tolerant():
    assert wake_command("Hey, Robot! Pick up the cup.") == "pick up the cup"
    assert wake_command("robot, pick up the cup") is None
    assert wake_command("They robotically move") is None


def test_segmenter_keeps_preroll_and_ends_after_silence(monkeypatch):
    monkeypatch.setattr("natural_language_processing.speech_to_text.auto_stt.PRE_ROLL_SECONDS", 0.3)
    monkeypatch.setattr("natural_language_processing.speech_to_text.auto_stt.END_SILENCE_SECONDS", 0.7)
    monkeypatch.setattr("natural_language_processing.speech_to_text.auto_stt.MIN_SPEECH_SECONDS", 0.25)
    segmenter = UtteranceSegmenter(sample_rate=1_000)
    silence = np.zeros(100, dtype=np.int16)
    speech = np.ones(100, dtype=np.int16)

    segmenter.push(silence, False, 0.1)
    segmenter.push(silence, False, 0.2)
    _, onset, started = segmenter.push(speech, True, 0.3)
    assert started and onset == pytest.approx(0.2)
    segmenter.push(speech, True, 0.4)
    segmenter.push(speech, True, 0.5)

    utterance = None
    for step in range(6, 13):
        utterance, _, _ = segmenter.push(silence, False, step / 10)

    assert utterance is not None
    assert len(utterance) == 1_200
    assert np.all(utterance[:200] == 0)


def test_pcm_service_passes_samples_without_a_file():
    captured = {}
    node = SimpleNamespace()

    def transcribe(audio, sample_rate):
        captured["audio"] = audio
        captured["sample_rate"] = sample_rate
        return "pick the cup"

    node.transcribe_audio = transcribe
    request = SimpleNamespace(audio=[-32768, 0, 32767], sample_rate=16_000)
    response = SimpleNamespace(text="")

    SpeechToTextNode.transcribe_audio_callback(node, request, response)

    assert response.text == "pick the cup"
    assert captured["audio"].dtype == np.int16
    assert captured["audio"].tolist() == request.audio
    assert captured["sample_rate"] == 16_000


def test_auto_worker_publishes_wake_command_without_a_file():
    published = []
    utterances = queue.Queue()
    utterances.put((np.zeros(16_000, dtype=np.int16), 12.5))
    utterances.put(None)
    node = SimpleNamespace(
        _stop=threading.Event(),
        _utterance_queue=utterances,
        transcribe_audio=lambda _audio, _rate: "Hey, Robot! Stop.",
        publisher=SimpleNamespace(publish=published.append),
    )

    AutoSpeechToTextNode._transcribe_utterances(node)

    assert len(published) == 1
    assert published[0].all_text == "stop"
    assert published[0].header.stamp.sec == 12
    assert published[0].header.stamp.nanosec == 500_000_000
