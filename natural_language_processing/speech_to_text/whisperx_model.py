
import whisperx
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
        audio = whisperx.load_audio(file)
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        print(result["segments"]) # before alignment

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        print(result["segments"]) # after alignment
    
        return result["segments"]


def main():
    stt = SpeechToTextModel()
    stt.forward("/home/petr/Recordings/Clip")

if __name__ == "__main__":
    main()