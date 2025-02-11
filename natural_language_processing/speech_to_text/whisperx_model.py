
import whisperx, torch
import gc 

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
        audio = whisperx.load_audio(file)
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        
        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del self.model

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        return result["segments"]

    def forward_timestamped(self, file: str):
        whisperx_dict = self.raw_forward(file)
        #[{"start": 0.028, "end": 1.289, "text": " It's the imagination.", 
        # "words": [{"word": "It's", "start": 0.028, "end": 0.128, "score": 0.218}, 
        # {"word": "the", "start": 0.148, "end": 0.248, "score": 0.683}, 
        # {"word": "imagination.", "start": 0.268, "end": 0.929, "score": 0.873}]},
        # ...
        # ]
        
        timestamped_words = []
        for sentence in whisperx_dict:
            for word in sentence['words']:
                timestamped_words.append([word["end"], word['word']])
        return timestamped_words

def main():
    stt = SpeechToTextModel()
    stt.forward("/home/petr/Recordings/Clip")

if __name__ == "__main__":
    main()