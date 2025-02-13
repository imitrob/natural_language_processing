from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# from natural_language_processing.sentence_instruct_transformer.role_config.tiago import ROLE_DESCRIPTION
from natural_language_processing.sentence_instruct_transformer.role_config.gesturenlu import ROLE_DESCRIPTION


class SentenceProcessor():
    def __init__(self, model_name: str = "SultanR/SmolTulu-1.7b-Reinforced"):
        """Good models for instruct:
            model_name = Qwen/Qwen2.5-0.5B-Instruct (1GB VRAM)
            model_name = SultanR/SmolTulu-1.7b-Reinforced (3.3GB VRAM)

        Args:
            model_name (str, optional): _description_. Defaults to "SultanR/SmolTulu-1.7b-Reinforced".
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, 
                prompt: str,
                role_description: str = ROLE_DESCRIPTION,
                max_new_tokens: int = 50, 
                temperature: float = 0.0, 
                top_p: float = 1.0,
                repetition_penalty: float = 1.1,
            ) -> dict:
        """ Returns as "parsed" instruct dict in format:
            Dict[str, str]: keys are always ("target_action", "target_object", "target_storage")
        """
        response = self.raw_predict(prompt, role_description, max_new_tokens, temperature, top_p, repetition_penalty)
        print(response, flush=True)

        response = response.replace(", ", ",")
        response = response.replace("'", "")
        response = response.replace("a can", "can")
        response_list = response.split(",")
        r = {}
        k_prev = ""
        for i in range(len(response_list)):
            s = self.remove_article(response_list[i]) # get rid of a, the..
            
            k, v = self.sort_types(s) # get rid of "action: ..."
            if k_prev == k: # order from model is: object, object2 right after each other; color, color2
                r[k+"2"] = v
            else:
                r[k] = v
            k_prev = k
        return r

    def raw_predict(self, 
                    prompt: str, 
                    role_description: str = ROLE_DESCRIPTION,
                    max_new_tokens: int = 50, 
                    temperature: float = 0.0, 
                    top_p: float = 1.0,
                    repetition_penalty: float = 1.1,
                    ) -> str:
        """ Returns string output from LM. """
        messages = [
            {
            "role": "system",
            "content": role_description,
            },
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)


        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,  # Allow space for full format
            temperature=temperature,
            top_p=top_p,  # Use full distribution
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False  # Force greedy decoding
        )
        # Decode only the new tokens
        response = self.tokenizer.decode(
            generated_ids[0][model_inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )
    
        # Post-process to ensure format
        return response.split("\n")[0].strip()

    def predict_with_probs(self, 
                           prompt: list, 
                           user_input: str, 
                           role_description: str = "", 
                           max_new_tokens: int = 50
                           ):
        """Generate text using LM while incorporating word probabilities from ASR output.
        
        >>> output = model.predict_with_probs(
                prompt=[
                    [0.0, {"Pick": 1.0, "Kick": 0.2}],
                    [0.1, {"a": 0.9, "the": 0.1}],
                    [0.2, {"blue": 0.8, "green": 0.2}],
                    [0.3, {"box": 0.7, "blocks": 0.3}]
                ],
                user_input="Transcribe the following command:",
                max_new_tokens=4
            )
        """
        # Prepare chat template
        messages = [
            {"role": "system", "content": role_description},
            {"role": "user", "content": user_input}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Get only the word probability entries (ignore timestamps)
        word_entries = [entry[1] for entry in prompt]
        
        # Prepare logit biases for each generation step
        logit_biases = []
        for i in range(min(len(word_entries), max_new_tokens)):
            
            word_probs = word_entries[i] if i < len(word_entries) else {}
            bias = {}
            
            if word_probs:
                total_prob = sum(word_probs.values())
                for word, prob in word_probs.items():
                    tokenized = self.tokenizer.tokenize(word)
                    if len(tokenized) == 1:  # Only single-token words
                        token_id = self.tokenizer.convert_tokens_to_ids(tokenized[0])
                        if token_id != self.tokenizer.unk_token_id:
                            normalized_prob = prob / total_prob
                            bias[token_id] = torch.log(torch.tensor(normalized_prob)).item()
            logit_biases.append(bias)
        
        # Debug: Print logit biases
        for i, bias in enumerate(logit_biases):
            print(f"Step {i}: {bias}")

        print("LogitBiasProcessor(logit_biases)")
        print(LogitBiasProcessor(logit_biases))
        # Generate text with logit biases
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            logits_processor=[LogitBiasProcessor(logit_biases)],
            do_sample=True,  # Use sampling to respect probabilities
            temperature=0.5,  # Adjust temperature as needed
            top_p=1.0,  # Use full distribution
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and return
        response = self.tokenizer.decode(
            generated_ids[0][model_inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        return response.strip()
    
    def remove_article(self, str):
        if str[0:2] == "a ":
            str = str.replace("a ", "")
        if str[0:4] == "the ":
            str = str.replace("the ", "")
        return str

    COLORS = ["green", "blue", "red", "pink"]
    def remove_color(self, str):
        ''' Sometimes, model puts color to object, this is a workaround '''
        for color in self.COLORS:
            if color in str:
                str = str.replace(color+" ", "") # "blue box" -> "box"
                str = str.replace(color, "") # "blue" -> "", does nothing if not found
        return str
    
    RELATIONS = ["into"]
    def remove_relation(self, str):
        ''' Sometimes, model puts relation into an action, this is a workaround '''
        for relation in self.RELATIONS:
            if relation in str:
                str = str.replace(" "+relation, "") # "blue box" -> "box"
                str = str.replace(relation, "") # "blue" -> "", does nothing if not found
        return str

    def sort_types(self, str):
        if "action: " in str:
            str = str.split("action: ")[-1]
            str = self.remove_relation(str)
            return "target_action", str
        if "object: " in str:
            str = str.split("object: ")[-1]
            str = self.remove_color(str)
            return "target_object", str
        if "color: " in str:
            str = str.split("color: ")[-1]
            return "target_object_color", str
        if "relationship: " in str:
            str = str.split("relationship: ")[-1]
            return "relationship", str
        raise Exception(f"Either 'action:', 'object:', 'color: ' or 'relationship': in string {str}")

class LogitBiasProcessor:
    """Custom logits processor to apply logit biases."""
    def __init__(self, logit_biases):
        self.logit_biases = logit_biases
        self.step = 0
    
    def __call__(self, input_ids, scores):
        if self.step < len(self.logit_biases):
            bias = self.logit_biases[self.step]
            for token_id, value in bias.items():
                scores[0, token_id] += value
            print(f"Applied bias at step {self.step}: {bias}")  # Debug
        self.step += 1
        return scores

def main():
    sp = SentenceProcessor()
    print(f"Result: {sp.raw_predict('Pick a green book.')}")

    out = sp.predict_with_probs(
        prompt=[
            [0.0, {"Pick": 1.0, "Kick": 0.2}],
            [0.1, {"a": 0.9, "the": 0.1}],
            [0.2, {"blue": 0.8, "green": 0.2}],
            [0.3, {"box": 0.7, "blocks": 0.3}]
        ],
        user_input="",
        max_new_tokens=100
    )
    print(f"Result: {out}")


    try:
        while True:
            prompt = input("Enter: ")
            # print(f"Sample prompt: {prompt}")
            print(f"Result: {sp.raw_predict(prompt)}")
    except KeyboardInterrupt:
        exit()

if __name__ == "__main__":
    main()
