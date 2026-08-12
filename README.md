
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

Always-listening wake interaction keeps audio in memory and publishes the
command after hearing one utterance beginning with "hey robot":

```
ros2 run natural_language_processing stt_node --interaction auto --audio-device Jabra
ros2 run multi_modal_reasoning multi_modal_reasoning --name_user demo --interaction auto
```

`--audio-device` accepts a PortAudio index or a name substring. It falls back
to `AUDIO_DEVICE`, then the system default input. Wake phrase, VAD threshold,
pre-roll, silence timeout, and duration limits are at the top of
`speech_to_text/auto_stt.py`.

## FAQ:

- If recording not working: Try to copy the alsa lib to the miniconda
`mkdir ~/miniconda3/envs/<conda env>/lib/alsa-lib/`
`sudo cp /usr/lib/x86_64-linux-gnu/alsa-lib/* ~/miniconda3/envs/<conda env>/lib/alsa-lib/`
