
import whisperx, torch
import gc 
import os
# import soundfile as sf
from naive_merger.utils import cc

class SpeechToTextModel():
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
        self.pub.publish(data=self.forward(msg.data))

    def forward(self, file: str = "TestSound"):
        return self.raw_forward(file)['text']

    def raw_forward(self, file: str):
        # audio, sample_rate = sf.read(file)
        # duration = len(audio) / sample_rate
        # if duration < 0.5:
        #     return {"start": 0.0, "end": 0.0, "words": []}
        audio = whisperx.load_audio(file)
        try:
            result = self.model.transcribe(audio, batch_size=self.batch_size)
        except ValueError:
            print(f"{cc.W}You probably pressed twice, try again! (waveform not in right format, is being manipulated){cc.E}", flush=True)
            return [{"start": 0.0, "end": 0.0, "words": [{"word": ".", "start": 0.0, "end": 0.01, "score": "0.0"}]}]
        # os.remove(file)

        if len(result['segments']) == 0:
            return [{"start": 0.0, "end": 0.0, "words": [{"word": ".", "start": 0.0, "end": 0.01, "score": "0.0"}]}]
        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del self.model

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        return result["segments"]

    def forward_timestamped(self, stamped_filename: dict):
        whisperx_dict = self.raw_forward(stamped_filename['file'])
        #[{"start": 0.028, "end": 1.289, "text": " It's the imagination.", 
        # "words": [{"word": "It's", "start": 0.028, "end": 0.128, "score": 0.218}, 
        # {"word": "the", "start": 0.148, "end": 0.248, "score": 0.683}, 
        # {"word": "imagination.", "start": 0.268, "end": 0.929, "score": 0.873}]},
        # ...
        # ]
        
        timestamped_words = []
        for sentence in whisperx_dict:
            for word in sentence['words']:
                timestamped_words.append([float(stamped_filename["timestamp"]) + word["end"], word['word']])
        return timestamped_words

def main():
    stt = SpeechToTextModel()
    stt.forward("/home/petr/Recordings/Clip")

if __name__ == "__main__":
    main()