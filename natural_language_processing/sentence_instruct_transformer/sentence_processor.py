from transformers import AutoModelForCausalLM, AutoTokenizer
import torch




ROLE_DESCRIPTION = """
You are an assistant that extracts the action, objects, spatial relationships, and their colors from user sentences. Output the result as: action, first object, second object, spatial relationship, first object color, second object color. Always specify the spatial relationship, such as 'on top', 'into', or 'near'. If there is no second object, color, or spatial relationship, return null for those fields. Colors are adjectives and should never be classified as objects.

actions are: pick, put, pour, place, stack, unglue.
objects are: cup, book, water, ball, cup, laptop, cleaner, tape, cube, drawer 
colors are: green, red, yellow, blue, pink

Here are some examples:


   Input: 'Put the book on top of the drawer.'
   Output: 'action: put, object: book, object: drawer, relationship: on top, color: null, color: null'

   Input: 'Put the cup into the drawer.'
   Output: 'action: put into, object: cup, object: drawer, relationship: null, color: null, color: null'

   Input: 'Place the ball near the chair.'
   Output: 'action: place, object: ball, object: chair, relationship: near, color: null, color: null'

   Input: 'Pick up the red ball.'
   Output: 'action: pick up, object: ball, object: null, relationship: null, color: null, color: red'
   
   Input: 'Stack the boxes on top of each other.'
   Output: 'action: stack, object: boxes, object: each other, relationship: on top, color: null, color: null'

   Input: 'Pick up the blue cup.'
   Output: 'action: pick, object: cup, object: null, relationship: up, color: blue, color: null'

   Input: 'Put the red book on the table.'
   Output: 'action: put, object: book, object: table, relationship: on, color: red, color: null'
   
   Input: 'Pour the water into the green bowl.'
   Output: 'action: pour, object: water, object: bowl, relationship: into, color: null, color: green'

   Input: 'Place the yellow ball in the basket.'
   Output: 'action: place, object: ball, object: basket, relationship: in, color: yellow, color: null'

   Input: 'Put the cup in the drawer.'
   Output: 'action: put, object: cup, object: drawer, relationship: in, color: null, color: null'
   
   Input: 'Pour the water into the bowl.'
   Output: 'action: pour, object: water, object: bowl, relationship: into, color: null, color: null'
   
   Input: 'Pick up the red book.'
   Output: 'action: pick, object: book, object: null, relationship: up, color: red, color: null'

   Input: 'Place the laptop on the desk.'
   Output: 'action: place, object: laptop, object: desk, relationship: on, color: null, color: null'

   Input: 'stack cleaner to crackers'
   Output: 'action: stack, object: cleaner, object: crackers, relationship: to, color: null, color: null'

   Input: 'unglue tape from box'
   Output: 'action: unglue, object: tape, object: box, relationship: from, color: null, color: null'

   Input: 'put a cube into the drawer'
   Output: 'action: put, object: cube, object: drawer, relationship: into, color: null, color: null'

   Input: 'pick the red cube'
   Output: 'action: pick, object: cube, object: null, relationship: null, color: red, color: null'

   Input: 'stack the cleaner to the cup'
   Output: 'action: stack, object: cleaner, object: cup, relationship: to, color: null, color: null'

   Input: 'Put it on top of the drawer'
   Output: 'action: put, object: drawer, object: null, relationship: on, color: null, color: null'

   Input: 'Put it into the drawer'
   Output: 'action: put into, object: drawer, object: null, relationship: into, color: null, color: null'

   For every sentence, always identify the action, objects, spatial relationship, and their colors. If any information is missing, return 'null' for that field.
"""


class SentenceProcessor():
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)




    def predict(self, prompt: str):
        """ Returns instruct dict in format:
            Dict[str, str]: keys are always ("target_action", "target_object", "target_storage")
        """
        response = self.raw_predict(prompt)
        print(response)

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

    def raw_predict(self, prompt: str) -> str:
        """ Returns string output from LM. """
        messages = [
            {
            "role": "system",
            "content": ROLE_DESCRIPTION,
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
            max_new_tokens=50,
            temperature = 0.3,
            top_p = 0.9,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

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
        raise Exception()

def main():
    sp = SentenceProcessor()
    prompt = "Pick for me the blue cup in the middle of the room."
    print(f"Sample prompt: {prompt}")
    print(f"Result: {sp.predict(prompt)}")

if __name__ == "__main__":
    main()
