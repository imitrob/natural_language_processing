import json
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from natural_language_processing.speech_to_text import wake_word_listener
from natural_language_processing.speech_to_text.wake_word_listener import (
    AutoSpeechToTextNode,
    UtteranceSegmenter,
    wake_command_words,
)
from natural_language_processing.speech_to_text.stt_node import SpeechToTextNode


def _words(*spoken):
    """Word entries as whisper_model.normalize_word already leaves them."""
    return [{"start": i / 10, "end": i / 10, "word": w, "alts": {}}
            for i, w in enumerate(spoken)]


def test_wake_phrase_must_be_the_first_words_spoken():
    assert [w["word"] for w in wake_command_words(_words("hey", "robot", "pick", "up"))] \
        == ["pick", "up"]
    assert wake_command_words(_words("robot", "pick", "up")) is None
    assert wake_command_words(_words("they", "robotically", "move")) is None


def test_segmenter_keeps_preroll_and_ends_after_silence(monkeypatch):
    monkeypatch.setattr("natural_language_processing.speech_to_text.wake_word_listener.PRE_ROLL_SECONDS", 0.3)
    monkeypatch.setattr("natural_language_processing.speech_to_text.wake_word_listener.END_SILENCE_SECONDS", 0.7)
    monkeypatch.setattr("natural_language_processing.speech_to_text.wake_word_listener.MIN_SPEECH_SECONDS", 0.25)
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

    def transcribe(audio, sample_rate, stamp=0.0):
        captured["audio"], captured["sample_rate"] = audio, sample_rate
        return [{"start": 1.0, "end": 1.2, "word": "pick", "alts": {"pick": 0.8}}]

    node = SimpleNamespace(transcribe_audio_words=transcribe)
    request = SimpleNamespace(audio=[-32768, 0, 32767], sample_rate=16_000, stamp=0.0)
    response = SimpleNamespace(text="", words_json="")

    SpeechToTextNode.transcribe_audio_callback(node, request, response)

    assert response.text == "pick"
    assert json.loads(response.words_json)[0]["alts"] == {"pick": 0.8}
    assert captured["audio"].dtype == np.int16
    assert captured["audio"].tolist() == request.audio
    assert captured["sample_rate"] == 16_000




def _devices(*names):
    return [{"name": name, "max_input_channels": 1} for name in names]


def test_default_picks_the_headset(monkeypatch):
    monkeypatch.delenv("AUDIO_DEVICE", raising=False)
    monkeypatch.setattr(wake_word_listener.sd, "query_devices", lambda *a, **k: _devices("built-in", "Jabra Speak 710"))

    assert wake_word_listener.resolve_audio_device(None)[0] == 1


def test_default_falls_back_when_the_headset_is_absent(monkeypatch):
    monkeypatch.delenv("AUDIO_DEVICE", raising=False)
    monkeypatch.setattr(wake_word_listener.sd, "query_devices",
                        lambda *a, **k: _devices("built-in") if not k else {"name": "built-in"})

    assert wake_word_listener.resolve_audio_device(None) == (None, {"name": "built-in"})


def test_an_asked_for_device_must_exist(monkeypatch):
    monkeypatch.setattr(wake_word_listener.sd, "query_devices", lambda *a, **k: _devices("built-in"))

    with pytest.raises(ValueError):
        wake_word_listener.resolve_audio_device("Shure")


def test_wake_words_are_sliced_off_the_word_stream():
    """words_json must stay index-aligned with all_text, so both are built
    from the same sliced list."""
    words = [{"start": 1.0, "end": 1.1, "word": "hey", "alts": {"hey": 0.9}},
             {"start": 1.2, "end": 1.3, "word": "robot", "alts": {"robot": 0.9}},
             {"start": 1.5, "end": 1.7, "word": "pick", "alts": {"pick": 0.5, "push": 0.2}}]
    command = wake_command_words(words)

    assert [w["word"] for w in command] == ["pick"]
    assert command[0]["start"] == 1.5
    assert command[0]["alts"] == {"pick": 0.5, "push": 0.2}
    assert wake_command_words(words[2:]) is None  # no wake phrase


def test_auto_worker_publishes_real_stamps_and_alternatives():
    published = []
    utterances = queue.Queue()
    utterances.put((np.zeros(16_000, dtype=np.int16), 12.5))
    utterances.put(None)

    def transcribe_audio_words(_audio, _rate, stamp=0.0):
        return [{"start": stamp, "end": stamp + 0.1, "word": "hey", "alts": {"hey": 0.9}},
                {"start": stamp + 0.2, "end": stamp + 0.3, "word": "robot", "alts": {}},
                {"start": stamp + 0.9, "end": stamp + 1.2, "word": "stop",
                 "alts": {"stop": 0.61, "stomp": 0.04}}]

    node = SimpleNamespace(
        _stop=threading.Event(),
        _utterance_queue=utterances,
        transcribe_audio_words=transcribe_audio_words,
        publisher=SimpleNamespace(publish=published.append),
    )

    AutoSpeechToTextNode._transcribe_utterances(node)

    assert len(published) == 1
    assert published[0].all_text == "stop"
    words = json.loads(published[0].words_json)
    assert [w["word"] for w in words] == published[0].all_text.split()
    # Absolute: the utterance began at 12.5 and "stop" 0.9 s into it.
    assert words[0]["start"] == pytest.approx(13.4)
    assert words[0]["alts"] == {"stop": 0.61, "stomp": 0.04}




def test_text_and_words_are_built_from_the_same_stream():
    from natural_language_processing.speech_to_text.stt_node import fill_response

    response = SimpleNamespace(text="", words_json="")
    fill_response(response, [{"start": 1.0, "end": 1.2, "word": "pick", "alts": {"pick": 0.8}},
                             {"start": 1.4, "end": 1.6, "word": "cup", "alts": {}}])
    assert response.text == "pick cup"
    assert [w["word"] for w in json.loads(response.words_json)] == response.text.split()
