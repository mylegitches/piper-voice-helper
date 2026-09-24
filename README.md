# Piper Voice Helper

Record yourself and train a [Piper text to speech](https://github.com/OHF-Voice/piper1-gpl) voice from the same web UI.

Based on [Piper Recording Studio](https://github.com/rhasspy/piper-recording-studio) by Michael Hansen (MIT), with an added **Train Voice** button.

![Screen shot](etc/screenshot.jpg)

[![Sponsored by Nabu Casa](etc/nabu_casa_sponsored.png)](https://nabucasa.com)


## Tutorial

See a [video tutorial](https://www.youtube.com/watch?v=Z1pptxLT_3I) by [Thorsten Müller](https://www.thorsten-voice.de/)


## Docker

``` sh
docker run -it -p 8000:8000 -v '/path/to/output:/app/output' rhasspy/piper-recording-studio
```

Visit http://localhost:8000 to select a language and start recording.

Add `--help` to see more options.


### Building

``` sh
docker build . -t rhasspy/piper-recording-studio
```


## Installing without Docker

``` sh
git clone https://github.com/rhasspy/piper-recording-studio.git
cd piper-recording-studio/

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```


## Running without Docker

``` sh
python3 -m piper_recording_studio
```

Visit http://localhost:8000 to select a language and start recording.

Prompts are in the `prompts/` directory with the following format:

* Language directories are named `<language name>_<language code>`
* Each `.txt` in a language directory contains lines with:
    * `<id>\t<text>` or
    * `text` (id is automatically assigned based on line number)

Output audio is written to `output/`

See `--debug` for more options.


## Exporting

Install ffmpeg:

``` sh
sudo apt-get install ffmpeg
```

Install exporting dependencies:

``` sh
python3 -m pip install -r requirements_export.txt
```

Export recordings for a language to a Piper-compatible dataset (LJSpeech format):

``` sh
python3 -m export_dataset output/<language>/ /path/to/dataset
```

Requires a non-Docker install. If you used Docker to record your dataset, you may need to adjust the permissions of the output directory:

``` sh
sudo chown -R "$(id -u):$(id -u)" output/
```

See `--help` for more options. You may need to adjust the silence detection parameters to correctly remove button clicks and keypresses.


## Training

Training uses [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl), with a workflow modeled on
[TextyMcSpeechy](https://github.com/domesticatedviking/TextyMcSpeechy): fine-tune a pretrained checkpoint,
listen to checkpoints while training runs, and stop when the voice sounds right.

Install piper1-gpl training once (needs `build-essential cmake ninja-build`; a CUDA GPU is strongly recommended):

``` sh
script/setup_training
```

This creates `.piper1-gpl/`, which the studio uses automatically (override with `--train-python /path/to/python3`).

Pick a language on the home page (or finish recording) and click **Train Voice**:

* **Starting checkpoint**: medium-quality [pretrained checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints) from TextyMcSpeechy's per-language lists (listen at [piper-samples](https://rhasspy.github.io/piper-samples/)), a custom path/URL, "continue from this voice's latest checkpoint", or none (from scratch). Downloads are cached in `output/_training/checkpoints/`.
* **Epochs**: `0` trains until you press **Stop**; otherwise the number of epochs past the starting checkpoint.
* **Export & test latest checkpoint**: works during or after training. It writes `voice/epoch_<N>/<lang>-<name>-medium.onnx` + `.onnx.json` (with the fields Home Assistant expects) and speaks the test sentence so you can play it in the page.

Training runs `export_dataset` on your recordings first (install `ffmpeg` and `requirements_export.txt`). Work files and `train.log` go to `output/_training/<language>/`. One job runs at a time. Training is not available in the Docker image.

Compatibility notes: the MOS checkpoint callback is disabled so training works offline, and ONNX export uses torch's TorchScript exporter because the default dynamo exporter in torch 2.9+ fails on Piper models.


## Multi-User Mode

``` sh
python3 -m piper_recording_studio --multi-user
```

Now a "login code" will be required to record. A directory `output/user_<code>/<language>` must exist for each user and language.
