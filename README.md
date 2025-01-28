
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
```

## Usage

```
ros2 run natural_language_processing nl_node
```

## FAQ:

- If recording not working: Try to copy the alsa lib to the miniconda
`mkdir ~/miniconda3/envs/<conda env>/lib/alsa-lib/`
`sudo cp /usr/lib/x86_64-linux-gnu/alsa-lib/* ~/miniconda3/envs/<conda env>/lib/alsa-lib/`
