# tests/test_basic_phrase.py
import unittest
import pathlib
import urllib.request
import string

import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

MODEL_NAME = "nvidia/stt_en_fastconformer_hybrid_large_pc"
AUDIO_URL  = "https://datasets-server.huggingface.co/assets/voidful/librispeech_tts/--/default/test.clean/0/audio/audio.wav"
AUDIO_F    = pathlib.Path(__file__).parent / "audio.wav"

# ----------------------------------------------------------------------
def _download_sample():
    if not AUDIO_F.exists():
        AUDIO_F.write_bytes(urllib.request.urlopen(AUDIO_URL).read())

def _simple_normalize(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    
    return text.text.lower().translate(table).strip()

# ----------------------------------------------------------------------
class TestHelloOneTwoThree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _download_sample()
        cls.model = (
            EncDecRNNTBPEModel
            .from_pretrained(MODEL_NAME)
            .cuda()
            .eval()
        )

    def test_basic_phrase(self):
        hyp = self.model.transcribe([str(AUDIO_F)])[0]
        hyp = _simple_normalize(hyp)
        print("\n[DEBUG] hypothesis:", hyp)
        
        for word in ['concord', 'returned', 'to', 'its', 'place', 'amidst', 'the','tents']:
            self.assertIn(word, hyp)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    unittest.main()
