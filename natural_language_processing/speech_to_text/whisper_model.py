"""Whisper speech-to-text with real per-word timestamps and word alternatives.

One encoder pass plus one generate() call returns the transcription, the DTW token
timestamps and the per-step logits, so timing and alternatives cost no extra
decoding pass. The pipeline() API cannot return either, which is why this calls
the model directly. The encoder is run here rather than inside generate() so that
return_token_timestamps does not drag it off SDPA -- see _generate.

Alternatives come from the top-k of a word's *first* token. On its own that top-k
is a list of subword fragments ('Sp', 'Cas'), so each alternative is teacher-forced
and greedily continued to the word boundary, turning ' Sp' into ' Spongebob'. That
roughly doubles a short utterance on a 3080 (0.15 s -> 0.37 s for 3.3 s of audio):
each continuation step is its own un-cached decoder forward, so the cost is kernel
launches, not arithmetic.
ponytail: serial per-alternative decoding. Batch the TOP_K continuations into one
forward if this ever matters -- it is ~25 forwards per utterance today.

Probabilities are whisper's own and are NOT renormalised: the mass missing from
the top-k is the model's own uncertainty about words outside the list, and a
consumer that renormalises would be discarding it.
"""
import numpy as np
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

TOP_K = 5
MIN_ALT_PROB = 0.01
# Whisper's encoder takes a fixed 30 s window. Both callers are well inside it
# (AudioRecorder records 5 s, wake_word_listener caps an utterance at 15 s).
# ponytail: single window, add chunked long-form only if a caller ever exceeds it.
MAX_AUDIO_SECONDS = 30.0
# A greedy continuation covers the longest word whisper tokenises in pieces.
MAX_CONTINUATION_TOKENS = 6

# Whisper prefixes a new word with a space; a piece without one continues the
# previous word. Special tokens are '<|...|>' and belong to no word.
_SPECIAL_PREFIX = "<|"


def normalize_word(word: str) -> str:
    """Lowercase, strip surrounding punctuation. '' if nothing is left.

    The topic already carries lowercase punctuation-free text (wake_command
    normalizes it the same way), so word labels must match or the two stop
    lining up index by index."""
    return word.strip().strip(".,!?;:'\"()[]-").lower()


def merge_alternatives(alternatives):
    """Sum colliding labels, drop the tail, keep descending order.

    Whisper spends much of its top-k on casing variants -- 'sponge' 0.608 and
    'Sponge' 0.384 are one candidate seen twice, and summing them to 0.992 is
    what makes the remaining mass mean something."""
    merged = {}
    for word, prob in alternatives:
        word = normalize_word(word)
        if word:
            merged[word] = merged.get(word, 0.0) + prob
    return {w: round(p, 4) for w, p in
            sorted(merged.items(), key=lambda kv: -kv[1]) if p >= MIN_ALT_PROB}


class SpeechToTextModel():
    def __init__(self,
                 model_id = "openai/whisper-large-v3-turbo", # 13 seconds on laptop
                 device = "cuda:0", # or "cpu"
                 torch_dtype = torch.float16 # torch.float32
                ):
        super(SpeechToTextModel, self).__init__()

        self.device = device
        self.torch_dtype = torch_dtype
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)

    def delete(self):
        self.model.to("cpu")
        del self.model

    def __call__(self, file: str = ""):
        """ Convenience function"""
        return self.transcribe_to_text(file)

    # Transcription
    def _load(self, file: str):
        """Any WAV the recorder writes as mono float32 at whisper's 16 kHz.

        AudioRecorder's `-f cd` is 44.1 kHz stereo, so both a downmix and a
        resample are needed; soundfile and scipy are already dependencies."""
        import soundfile as sf
        from scipy.signal import resample_poly
        audio, sample_rate = sf.read(file, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sample_rate != 16_000:
            audio = resample_poly(audio, 16_000, sample_rate)
        return audio.astype(np.float32)

    # generate() is internally no-grad, but the alternatives pass calls the
    # model directly -- without this the encoder graph (32 layers x 1500 frames
    # of attention activations) is retained for the whole call: ~6 GB of VRAM.
    @torch.inference_mode()
    def _generate(self, audio, sample_rate: int = 16_000, alternatives: bool = False):
        """Transcribe float32 mono audio to a list of word dicts.

        Returns [{"start", "end", "word", "alts"}, ...] with times relative to
        the start of the audio. `alts` is empty unless `alternatives` is set."""
        seconds = len(audio) / sample_rate
        if seconds > MAX_AUDIO_SECONDS:
            print(f"Audio is {seconds:.1f}s, only the first {MAX_AUDIO_SECONDS:.0f}s "
                  f"are transcribed (whisper's window)", flush=True)
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt",
                                return_attention_mask=True)
        features = inputs.input_features.to(self.device, self.torch_dtype)
        # DTW needs to know how much of the fixed 30 s window is real audio, and
        # `num_frames` is the only thing it reads for that -- an attention_mask is
        # ignored (_set_num_frames pops num_frames and nothing else). Without it
        # DTW warps over all 1500 encoder positions and every word in a 3.3 s clip
        # comes back at t=29.98, pinned to the far end of the padding.
        num_frames = int(inputs.attention_mask.sum())

        # Encode once, outside generate(), and hand the result in. return_token_timestamps
        # switches the whole model to output_attentions=True, and SDPA cannot return
        # attention weights, so every attention falls back to the eager implementation.
        # For the 32-layer encoder that means materialising 20 heads x 1500 x 1500
        # scores per layer: 2.5 s and a 7 GB peak on a 3080, to produce attentions DTW
        # never looks at -- it only needs the decoder's cross-attention. Pre-encoding
        # keeps the encoder on SDPA and leaves eager to the 4-layer decoder.
        encoder_outputs = self.model.get_encoder()(features)
        output = self.model.generate(
            features, encoder_outputs=encoder_outputs, num_frames=num_frames,
            language="en", task="transcribe",
            return_token_timestamps=True, output_scores=True, return_dict_in_generate=True,
        )
        token_ids = output["sequences"][0].tolist()
        timestamps = output["token_timestamps"][0]
        scores = output["scores"]
        # DTW leaves all cross-attentions in output; free them before re-encoding.
        # del output
        # generate() scores only the tokens it produced, so the forced prefix
        # (<|startoftranscript|><|en|>...) has no score to line up with.
        score_offset = len(token_ids) - len(scores)

        # The alternatives pass reuses the same encoder output as generate() did;
        # re-encoding per alternative costs more than the whole base pass.

        words, current = [], None
        for index, token in enumerate(token_ids):
            piece = self.processor.decode([token])
            if piece.startswith(_SPECIAL_PREFIX):
                continue
            stamp = float(timestamps[index])
            if piece.startswith(" ") or current is None:
                current = {"start": stamp, "end": stamp, "word": piece,
                           "index": index}
                words.append(current)
            else:
                current["word"] += piece
                current["end"] = stamp

        for word in words:
            word["alts"] = (self._word_alternatives(token_ids, word["index"], scores,
                                                    score_offset, encoder_outputs)
                            if alternatives else {})
            word["word"] = normalize_word(word.pop("word"))
            del word["index"]
        return [w for w in words if w["word"]]

    def _word_alternatives(self, token_ids, index, scores, score_offset, encoder_outputs):
        """Top-k of this word's first token, each continued into a whole word."""
        score_index = index - score_offset
        if not 0 <= score_index < len(scores):
            return {}
        probs = torch.softmax(scores[score_index][0].float(), dim=-1)
        top_probs, top_tokens = torch.topk(probs, TOP_K)
        prefix = token_ids[:index]
        return merge_alternatives(
            (self._continue_word(prefix, int(token), encoder_outputs), float(prob))
            for prob, token in zip(top_probs, top_tokens))

    def _continue_word(self, prefix, first_token, encoder_outputs):
        """Force `first_token` after `prefix`, then greedily finish the word.

        Whisper's top-k on a multi-token word is fragments, so the alternative
        has to be decoded out before it names anything: ' Sp' -> ' Spongebob'."""
        sequence = list(prefix) + [first_token]
        for _ in range(MAX_CONTINUATION_TOKENS):
            logits = self.model(
                decoder_input_ids=torch.tensor([sequence], device=self.device),
                encoder_outputs=encoder_outputs).logits[0, -1]
            next_token = int(logits.argmax())
            piece = self.processor.decode([next_token])
            if piece.startswith((" ", _SPECIAL_PREFIX)):
                break
            sequence.append(next_token)
        return self.processor.decode(sequence[len(prefix):])

    # Model API: __call__(file) -> text, plus the stamped/probabilistic variants
    def transcribe_to_text(self, file: str):
        assert isinstance(file, str)
        r = " ".join(w["word"] for w in self._generate(self._load(file)))
        print("whisper out: ", r)
        return r

    def transcribe_audio_words(self, audio, sample_rate: int = 16_000,
                               stamp: float = 0.0, alternatives: bool = True):
        """Word dicts for signed 16-bit PCM, timestamps offset by `stamp`."""
        samples = np.asarray(audio, dtype=np.int16)
        if samples.size == 0:
            return []
        words = self._generate(samples.astype(np.float32) / 32768.0, sample_rate,
                               alternatives=alternatives)
        return stamp_words(words, stamp)

    def transcribe_to_words(self, file: str, stamp: float = 0.0,
                            alternatives: bool = True):
        """Word dicts for a file, timestamps offset by `stamp`."""
        return stamp_words(self._generate(self._load(file), alternatives=alternatives),
                           stamp)



def stamp_words(words, stamp: float = 0.0):
    """Shift word times onto the wall clock the recording started on.

    Downstream matches gestures to words by comparing these numbers directly
    (tell_show.anchor_slots), so they have to share the gestures' clock."""
    return [{**word, "start": word["start"] + stamp, "end": word["end"] + stamp}
            for word in words]


if __name__ == "__main__":
    # Self-check without a model: the word/alternative post-processing is the
    # part with branches, and it runs on the CPU.
    assert normalize_word(" Spongewipe, ") == "spongewipe"
    assert normalize_word(" ... ") == ""
    # Casing variants are one candidate seen twice.
    assert merge_alternatives([(" sponge", 0.608), (" Sponge", 0.384),
                               (" Sp", 0.004)]) == {"sponge": 0.992}
    # Below the floor, and empty labels, are dropped.
    assert merge_alternatives([(" pick", 0.9), (" push", 0.005), (".", 0.05)]) == {"pick": 0.9}
    # Raw, not renormalised: the missing mass stays missing.
    assert abs(sum(merge_alternatives([(" pick", 0.35), (" push", 0.10)]).values()) - 0.45) < 1e-9
    assert stamp_words([{"start": 0.4, "end": 0.6, "word": "pick"}], 100.0) == \
        [{"start": 100.4, "end": 100.6, "word": "pick"}]
    # The VRAM fix is autograd being off inside _generate. A stub `self` proves
    # the decorator is still there without loading 1.6 GB of weights.
    class _Stop(Exception):
        pass

    class _ModeProbe:
        @property
        def processor(self):
            raise _Stop(torch.is_inference_mode_enabled())

    try:
        SpeechToTextModel._generate(_ModeProbe(), np.zeros(16, dtype=np.float32))
    except _Stop as stop:
        assert stop.args[0], "_generate lost @torch.inference_mode(): VRAM will blow up"
    print("whisper_model post-processing checks ok")
