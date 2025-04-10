
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

class SpeechToTextModel():
    def __init__(self,
                 model_id = "openai/whisper-large-v3-turbo", # 13 seconds on laptop
                 device = "cuda:0", # or "cpu"
                 torch_dtype = torch.float16 # torch.float32
                ):
        super(SpeechToTextModel, self).__init__()

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

    def delete(self):
        del self.model

    def callback(self, msg):
        self.pub.publish(data=self(msg.data))

    def __call__(self, file: str = ""):
        """ Convenience function"""
        return self.transcribe_to_text(file)
    
    def transcribe_to_text(self, file: str):
        assert isinstance(file, str)
        r = self.pipe(file)["text"]
        print("whisper out: ", r)
        return r

    def transcribe_to_stamped(self, file: str, stamp: float = 0.0):
        l = self.transcribe_to_text(file).split(" ")
        ret = []
        for n,w in enumerate(l):
            ret.append([stamp+n*0.2, w])
        return ret

    def transcribe_to_probstamped(self, file: str, stamp: float = 0.0):
        l = self.transcribe_to_text(file).split(" ")
        ret = []
        for n,w in enumerate(l):
            ret.append([stamp+n*0.2, {w: 1.0}])
        return ret


if __name__ == "__main__":
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
    text = stt.transcribe_to_stamped(file, stamp)
    print(f"Run (2/3): {text} time: {time.time()-t0}")
    
    t0 = time.time()
    text = stt.transcribe_to_probstamped(file, stamp)
    print(f"Run (3/3): {text} time: {time.time()-t0}")
    



