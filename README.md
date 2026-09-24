# Piper Voice Helper

A web app that turns your voice into a [Piper](https://github.com/OHF-Voice/piper1-gpl) text-to-speech voice for Home Assistant. No coding, no command line after setup, everything runs locally.

**Record** sentences in the browser → click **Train** → **listen** to it → **download** for Home Assistant.

---

## Quick start

Requirements: Docker, ~15 GB of disk, and an NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (strongly recommended; CPU training takes days).

```bash
git clone https://github.com/mylegitches/piper-voice-helper.git
cd piper-voice-helper
docker compose up --build
```

Open **http://localhost:8000**.

No GPU? `docker compose -f docker-compose.cpu.yml up --build` (recording and testing are fine, training is very slow).

## Using it

1. **Voice**: create a voice with a name, language and whether it should sound female or male (this picks a similar pretrained voice to start from).
2. **Record**: read the sentences shown. Keys: `R` record/stop, `P` play back, `S` save & next, `K` skip. 50 recordings is the minimum, 300+ sounds much better. Already have recordings? Upload a zip (`metadata.csv` with `file|text` lines plus audio, or audio files with matching `.txt` transcripts).
3. **Train**: pick how long and click **Train**:

   | Preset | Time | |
   |---|---|---|
   | Quick test | 30 min | rough, to check everything works |
   | **Good** (default) | 3 hours | |
   | Best | 8 hours | |
   | Until I stop it | no limit | press **Stop** when it sounds right |

   **Advanced settings** has the starting voice, hours, max epochs, batch size (auto-picked from GPU memory), device and sample rate. **Train more** continues where the last run stopped.
4. **Test & install**: type anything and click **Speak**. During training, **Export latest version now** lets you hear progress. Then click **Download for Home Assistant**:
   1. In Home Assistant open **Settings → Add-ons → Piper → Open Web UI** and upload the `.onnx` and `.onnx.json` files (or copy them to `/share/piper`).
   2. Restart the Piper add-on if the voice doesn't show up.
   3. In **Settings → Voice assistants**, pick your voice.

Everything is stored in `./data/` (voices, recordings, training runs, downloaded base voices).

## Home Assistant compatibility

Voices are trained with piper1-gpl, the current Piper. Home Assistant's Piper add-on runs `wyoming-piper`, which uses the same `piper-tts` package (1.8+), so the old `rhasspy/piper` is not needed. Files are named `<lang>-<name>-medium.onnx` and the `.onnx.json` has the fields Home Assistant expects.

## Without Docker

Needs Python 3.10+, `git`, `ffmpeg`, `build-essential`, `cmake` and `ninja-build`:

```bash
script/setup   # creates .venv with the app and piper1-gpl training
script/run     # http://localhost:8000
```

## How it works

```
recordings → trim silence (Silero VAD) → dataset → piper1-gpl fine-tune → ONNX + .onnx.json
```

- **FastAPI** serves a single page and streams training progress with server-sent events.
- Training fine-tunes a medium-quality checkpoint from [piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints), chosen from [TextyMcSpeechy](https://github.com/domesticatedviking/TextyMcSpeechy)'s per-language lists (multi-speaker checkpoints are excluded).
- Training stops at the time limit (Lightning `max_time`), the epoch limit, or **Stop**. The latest checkpoint is then exported automatically.
- One training job at a time; a second request gets HTTP 409.
- Two workarounds for current piper1-gpl: the MOS checkpoint callback is disabled so training works offline, and ONNX export uses torch's TorchScript exporter because the dynamo exporter (default since torch 2.9) fails on Piper models.

## Credits

- [Piper Recording Studio](https://github.com/rhasspy/piper-recording-studio) by Michael Hansen (MIT): prompts, recorder approach and dataset export (`export_dataset/`, `prompts/`, see `LICENSE.md`)
- [TextyMcSpeechy](https://github.com/domesticatedviking/TextyMcSpeechy) by Erik Bjorgan (MIT): pretrained checkpoint lists and training workflow (`app/checkpoints/`)
- [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) (GPL-3.0): training and synthesis, installed at build time
- UI modeled on [easy-wakeword-trainer](https://github.com/mylegitches/easy-wakeword-trainer)
