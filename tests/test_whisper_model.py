from types import SimpleNamespace

import numpy as np
import torch

from natural_language_processing.speech_to_text import whisper_model


def test_model_loader_uses_supported_torch_dtype_keyword(monkeypatch):
    class Model:
        def to(self, _device):
            return self

    def load_model(_model_id, *, torch_dtype, low_cpu_mem_usage, use_safetensors):
        assert torch_dtype is torch.float16
        assert low_cpu_mem_usage
        assert use_safetensors
        return Model()

    monkeypatch.setattr(
        whisper_model.AutoProcessor, "from_pretrained", staticmethod(lambda _model_id: object())
    )
    monkeypatch.setattr(
        whisper_model.WhisperForConditionalGeneration,
        "from_pretrained",
        staticmethod(load_model),
    )

    whisper_model.SpeechToTextModel(device="cpu")


def test_generate_reuses_attention_free_encoder_output():
    released = False
    encoder_calls = 0
    encoded = object()

    class Output(dict):
        def __del__(self):
            nonlocal released
            released = True

    class Processor:
        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(
                input_features=torch.zeros(1, 80, 3000),
                attention_mask=torch.ones(1, 3000),
            )

        def decode(self, _tokens):
            return "<|endoftext|>"

    class Model:
        def generate(self, *_args, **kwargs):
            assert encoder_calls == 1
            assert kwargs["encoder_outputs"] is encoded
            return Output(
                sequences=torch.tensor([[1]]),
                token_timestamps=torch.tensor([[0.0]]),
                scores=(),
            )

        def get_encoder(self):
            def encode(_features):
                nonlocal encoder_calls
                encoder_calls += 1
                return encoded

            return encode

    model = SimpleNamespace(
        processor=Processor(), model=Model(), device="cpu", torch_dtype=torch.float32
    )

    assert whisper_model.SpeechToTextModel._generate(
        model, np.zeros(16, dtype=np.float32), alternatives=True
    ) == []
    assert encoder_calls == 1
    assert released
