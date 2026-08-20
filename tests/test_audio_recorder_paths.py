"""The recording path travels to the speech-to-text server as a string, so it
has to be absolute and it has to differ between recordings."""
from pathlib import Path

from natural_language_processing.speech_to_text import audio_recorder
from natural_language_processing.speech_to_text.audio_recorder import AudioRecorder


class _FakeProcess:
    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs

    def terminate(self):
        pass

    def wait(self):
        pass


def _record(monkeypatch, tmp_path, output_file=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audio_recorder.subprocess, "Popen", _FakeProcess)
    recorder = AudioRecorder()
    recorder.start_recording(output_file)
    return recorder


def test_recording_path_is_absolute(monkeypatch, tmp_path):
    recorder = _record(monkeypatch, tmp_path)

    assert Path(recorder.output_file).is_absolute()
    assert recorder.output_file in recorder.process.argv


def test_each_recording_gets_its_own_file(monkeypatch, tmp_path):
    first = _record(monkeypatch, tmp_path).output_file
    second = _record(monkeypatch, tmp_path).output_file

    assert first != second


def test_an_explicit_path_is_kept(monkeypatch, tmp_path):
    target = tmp_path / "asked_for.wav"

    recorder = _record(monkeypatch, tmp_path, output_file=target)

    assert recorder.output_file == str(target)


def test_the_recorder_does_not_steal_the_terminals_stdin(monkeypatch, tmp_path):
    """record_voice() calls input() again right after start_recording(); an
    inherited stdin is consumed by arecord and that input() raises EOFError."""
    recorder = _record(monkeypatch, tmp_path)

    assert recorder.process.kwargs["stdin"] is audio_recorder.subprocess.DEVNULL
