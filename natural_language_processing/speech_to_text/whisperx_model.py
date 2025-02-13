
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

    def stamped_transcribe(self, stamped_filename: dict):
        """Returns pairs (timestamp, word) of sentence
            Input: "Pick a blue box"
            Output: [
                [0.0, "Pick"],
                [0.1, "a"],
                [0.2, "blue"],
                [0.3, "box"],
            ]
        """        
        whisperx_dict = self.raw_transcribe(stamped_filename['file'])
        return self.whisperx_result_to_timestamped_words(whisperx_dict, stamped_filename["timestamp"])

    def whisperx_result_to_timestamped_words(self, whisperx_dict: dict, start_timestamp: float):
        timestamped_words = []
        for sentence in whisperx_dict:
            for word in sentence['words']:
                timestamped_words.append([float(start_timestamp) + word["end"], word['word']])
        return timestamped_words

def main():
    stt = SpeechToTextModel()
    output = stt("/home/imitlearn/lfd_ws/output.wav")
    print("1. ", output)
    output = stt.stamped_transcribe({"file": "/home/imitlearn/lfd_ws/output.wav", "timestamp": 1700000})
    print("2. ", output)

if __name__ == "__main__":
    main()