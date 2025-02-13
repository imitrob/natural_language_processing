import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from nltk.corpus import words

DISCARD_PROBABILITY_THR = 0.08
DISCARD_NON_ENG_VOCAB = True
DISCARD_NONVALID = True

# Download words if not already downloaded
if DISCARD_NON_ENG_VOCAB:
    import nltk
    nltk.download('words')
    word_list = set(words.words())

def is_english_word(word):
    if word[0] == " ": word = word[1:]
    if word[-1] == " ": word = word[:-1]
    return word.lower() in word_list


def is_valid_whisper_token(processor, word):
    tokenized = processor.tokenizer(word, add_special_tokens=False).input_ids
    return len(tokenized) > 0

def discard_nonvalid_words(processor, output):
    for ts,alternatives in output:
        for k in list(alternatives.keys()):
            if DISCARD_NONVALID and (not is_valid_whisper_token(processor, k)): 
                alternatives.pop(k)
            elif alternatives[k] < DISCARD_PROBABILITY_THR:
                alternatives.pop(k)
            elif DISCARD_NON_ENG_VOCAB and (not is_english_word(k)):
                alternatives.pop(k)
    return output

def discard_empty_words(output):
    return [[t,i] for t,i in output if len(i)>0]


def process_audio(processor, audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    return inputs.input_features

def get_word_alternatives(scores, token_id, processor, top_k=5):
    logits = scores[token_id]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_tokens = torch.topk(probs, top_k)
    
    alternatives = {}
    for prob, token in zip(top_probs[0].tolist(), top_tokens[0].tolist()):
        word = processor.decode([token])
        if word.strip() and not word.startswith("<|"):
            alternatives[word] = round(prob, 2)
    return alternatives

def decode_tokens_with_alternatives(processor, token_ids, scores, top_k=5):
    """Decode tokens and extract top-k alternatives for each token."""
    alternatives = []
    for token_id, score in zip(token_ids, scores):
        if score is None:
            continue  # Skip if no score is available
        
        # Get logits for the current token
        logits = score # Extract logits from the score tensor
        probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
        
        # Get top-k tokens and their probabilities
        top_probs, top_tokens = torch.topk(probs, top_k)
        
        # Decode tokens to words
        word_alternatives = {}
        for prob, token in zip(top_probs.tolist(), top_tokens.tolist()):
            word = processor.decode([token], skip_special_tokens=True)
            if word.strip():  # Filter out empty or special tokens
                word_alternatives[word] = round(prob, 2)
        
        alternatives.append(word_alternatives)
    return alternatives


def merge_subwords(output):
    """Merge subword fragments into complete words."""
    merged_output = []
    i = 0
    while i < len(output):
        timestamp, alternatives = output[i]
        current_word_list = list(alternatives.keys())
        if len(current_word_list) == 0:
            merged_output.append([timestamp, alternatives])
            i += 1
            continue
        current_word = current_word_list[0]  # Get the most likely word
        
        # Check if the current word is a subword fragment
        if i + 1 < len(output):
            next_timestamp, next_alternatives = output[i + 1]
            next_word_list = list(next_alternatives.keys())
            if len(next_word_list) == 0: 
                merged_output.append([timestamp, alternatives])
                i += 1
                continue
            next_word = next_word_list[0]
            
            # Merge if the next word is a continuation of the current word
            if not next_word.startswith(" ") and not next_word in {".", ",", "!", "?"}:
                merged_word = current_word + next_word
                merged_prob = {
                    k1 + k2: round(v1 * v2, 2)  # Combine probabilities
                    for k1, v1 in alternatives.items()
                    for k2, v2 in next_alternatives.items()
                } | alternatives
                merged_output.append([timestamp, merged_prob])
                i += 1  # Skip the next fragment
                continue
        
        # If no merge, add the current word as-is
        merged_output.append([timestamp, alternatives])
        i += 1
    
    return merged_output


class SpeechToTextModel():
    UNSUCCESSFUL = [{"start": 0.0, "end": 0.0, "words": [{"word": ".", "start": 0.0, "end": 0.01, "score": "0.0"}]}]
    def __init__(self,
                 model_id = "openai/whisper-small", # "large-v2",
                 device = "cuda", # or "cpu"
                ):
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)
        
    def __call__(self, audio_path: str):
        input_features = process_audio(self.processor, audio_path).to(self.device)
        # Generate transcription with timestamps and scores
        result = self.model.generate(
            input_features,
            return_timestamps=True,
            output_scores=True,
            return_dict_in_generate=True,
            task="transcribe"
        )

        segments = result["segments"]
        output = []
        for segments_ in segments:
            for segment in segments_:
                scores = segment['result']['scores']
                start_time = segment["start"].item()  # Convert tensor to float
                end_time = segment["end"].item()
                token_ids = segment["tokens"].tolist()  # Convert tensor to list of token IDs
                
                # Extract scores for the current segment
                segment_scores = scores[segment["idxs"][0]:segment["idxs"][1]]
                
                # Decode tokens and get alternatives
                word_alternatives = decode_tokens_with_alternatives(self.processor, token_ids, segment_scores)
                
                # Pair timestamps with word alternatives
                for i, (token_id, alternatives) in enumerate(zip(token_ids, word_alternatives)):
                    word = self.processor.decode([token_id], skip_special_tokens=True)
                    if word.strip():  # Filter out empty or special tokens
                        timestamp = start_time + (end_time - start_time) * (i / len(token_ids))
                        output.append([round(timestamp, 2), alternatives])
        
        merged_output = merge_subwords(output) # checkout connections to next words
        merged_output = merge_subwords(merged_output) # second round
        merged_output = discard_nonvalid_words(self.processor, merged_output) # check if in alphabet and above thr
        merged_output = discard_empty_words(merged_output)
        return merged_output


if __name__ == "__main__":
    pstt = SpeechToTextModel()
    output = pstt("/home/imitlearn/lfd_ws/output.wav")
    print(output)