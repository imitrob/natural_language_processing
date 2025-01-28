
# Natural language processing

Package requires hri_msgs, right now part of modality_merging package.

Press enter to start voice record, when button is released, the record is processed:
1. Speech to text node (*speech_to_text* folder)
2. Text to command (*sentence_instruct_transformer* folder)

See `nl_node.py` for more details.

## Install

Install packages:
```
conda env create -f environment.yml
cd <your_ws>/src/natural_language_processing
git clone https://huggingface.co/hexgrad/Kokoro-82M natural_language_processing/text_to_speech/Kokoro
```

## Usage

```
ros2 run natural_language_processing nl_node
```

