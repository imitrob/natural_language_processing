
import whisperx, torch
import gc 
import os
import soundfile as sf
from naive_merger.utils import cc

class SpeechToTextModel():
    UNSUCCESSFUL = [{"start": 0.0, "end": 0.0, "words": [{"word": ".", "start": 0.0, "end": 0.01, "score": "0.0"}]}]
    def __init__(self,
                 model_id = "tiny", # "large-v2",
                 device = "cuda", # or "cpu"
                 batch_size = 16, # reduce if low on GPU mem
                 compute_type = "float32",
                ):
        super(SpeechToTextModel, self).__init__()

        # 1. Transcribe with original whisper (batched)
        self.model = whisperx.load_model(model_id, device, compute_type=compute_type)

        self.batch_size = batch_size
        self.device = device

    def delete(self):
        self.model.to("cpu")
        del self.model

    def callback(self, msg):
        self.pub.publish(data=self(msg.data))

    def __call__(self, file: str = "TestSound"):
        return self.raw_transcribe(file)[0]['text']

    def is_audio_too_short(self, file: str, duration_thr: float = 0.5): # duration_thr [s]
        audio, sample_rate = sf.read(file)
        duration = len(audio) / sample_rate
        if duration < duration_thr:
            return True
        else:
            return False

    def remove_audio(self, file: str):
        os.remove(file)

    def raw_transcribe(self, file: str):
        audio = whisperx.load_audio(file)
        try:
            result = self.model.transcribe(audio, batch_size=self.batch_size)
            
        except ValueError:
            print(f"{cc.W}You probably pressed twice, try again! (waveform not in right format, is being manipulated){cc.E}", flush=True)
            return self.UNSUCCESSFUL

        if len(result['segments']) == 0:
            return self.UNSUCCESSFUL

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del self.model

        # 2. Align whisper output
        aligned_results = []
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        return aligned_result["segments"]


    def transcribe_to_text(self, file: str):
        raise NotImplementedError()

    def transcribe_to_stamped(self, file: str, stamp: int = 0):
        """Returns pairs (timestamp, word) of sentence
            Input: "Pick a blue box"
            Output: [
                [0.0, "Pick"],
                [0.1, "a"],
                [0.2, "blue"],
                [0.3, "box"],
            ]
        """        
        whisperx_dict = self.raw_transcribe(file)
        return self.whisperx_result_to_timestamped_words(whisperx_dict, stamp)

    def whisperx_result_to_timestamped_words(self, whisperx_dict: dict, start_timestamp: float):
        timestamped_words = []
        for sentence in whisperx_dict:
            for word in sentence['words']:
                print("word", word)
                timestamped_words.append([float(start_timestamp) + word["end"], word['word']])
        return timestamped_words

    def transcribe_to_probstamped(self, file: str, stamp: int = 0):
        whisperx_dict = self.raw_transcribe(file)

        timestamped_words = []
        for sentence in whisperx_dict:
            for word in sentence['words']:
                timestamped_words.append([float(stamp) + word["end"], {word['word']: 1.0}])
        return timestamped_words

def main():
    import time
    from natural_language_processing.speech_to_text.audio_recorder import AudioRecorder
    rec = AudioRecorder()
    rec.start_recording()
    time.sleep(5)
    file, stamp = rec.stop_recording()
    
    stt = SpeechToTextModel()
    t0 = time.time()
    text = stt(file)
    print(f"Run (1/3): {text} time: {time.time()-t0}")

    t0 = time.time()
    text = stt.transcribe_to_stamped(file)
    print(f"Run (2/3): {text} time: {time.time()-t0}")

    t0 = time.time()
    text = stt.transcribe_to_probstamped(file)
    print(f"Run (3/3): {text} time: {time.time()-t0}")

if __name__ == "__main__":
    main()