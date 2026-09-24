"""Voices on disk: recordings, prompts to read, and dataset upload.

data/voices/<name>/
  voice.json            name, language, espeak voice, gender
  recordings/<group>/<id>.<audio ext> + <id>.txt
  training/             see training.py
  exports/epoch_<N>/    <lang>-<name>-medium.onnx + .onnx.json
"""

import csv
import io
import json
import re
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AUDIO_EXTENSIONS = (".webm", ".ogg", ".wav", ".flac", ".mp3", ".m4a")
VOICE_NAME = re.compile(r"^[a-z0-9_]{1,40}$")

# Prompt language code (lower case) -> espeak-ng voice, where the language prefix
# alone isn't the right choice.
_ESPEAK_VOICES = {
    "en-us": "en-us",
    "en-ca": "en-us",
    # espeak-ng's "en" is British English ("en-gb" isn't a voice name)
    "en-gb": "en",
    "en-au": "en",
    "en-ie": "en",
    "en-in": "en",
    "pt-br": "pt-br",
    "pt-pt": "pt",
    "es-mx": "es-419",
    "zh-cn": "cmn",
    "zh-tw": "cmn",
    "zh-hk": "yue",
}


def default_espeak_voice(language: str) -> str:
    """Guess an espeak-ng voice from a prompt language code like en-US."""
    code = language.lower()
    return _ESPEAK_VOICES.get(code, code.split("-", maxsplit=1)[0])


@dataclass
class Prompt:
    """Sentence for the user to read."""

    group: str
    id: str
    text: str


def load_prompts(prompts_dir: Path) -> Tuple[Dict[str, str], Dict[str, List[Prompt]]]:
    """Language names and prompts from prompts/<name>_<code>/*.txt."""
    languages: Dict[str, str] = {}
    prompts: Dict[str, List[Prompt]] = {}
    for language_dir in sorted(prompts_dir.iterdir()):
        if not language_dir.is_dir():
            continue

        name, code = language_dir.name.rsplit("_", maxsplit=1)
        if code == "test":
            continue

        languages[code] = name
        language_prompts = prompts.setdefault(code, [])
        for prompt_path in sorted(language_dir.glob("*.txt")):
            with open(prompt_path, "r", encoding="utf-8") as prompt_file:
                reader = csv.reader(prompt_file, delimiter="\t")
                for i, row in enumerate(reader):
                    if not row:
                        continue

                    prompt_id = str(i) if len(row) == 1 else row[0]
                    language_prompts.append(
                        Prompt(group=prompt_path.stem, id=prompt_id, text=row[-1])
                    )

    return languages, prompts


@dataclass
class Voice:
    """A voice being recorded and trained."""

    name: str
    language: str
    espeak_voice: str
    gender: str = "female"
    microphone: str = ""
    """Microphone the first recording was made with (keeps clips consistent)."""

    created: float = field(default_factory=time.time)
    root: Path = field(default=Path(), repr=False)

    @property
    def recordings_dir(self) -> Path:
        return self.root / "recordings"

    @property
    def model_stem(self) -> str:
        """Piper naming convention, e.g. en_US-my_voice-medium."""
        return f"{self.language.replace('-', '_')}-{self.name}-medium"

    def to_json(self) -> Dict:
        data = asdict(self)
        data.pop("root")
        return data

    def save_meta(self) -> None:
        (self.root / "voice.json").write_text(
            json.dumps(self.to_json(), indent=2), encoding="utf-8"
        )

    def remember_microphone(self, label: str) -> None:
        label = label.strip()[:200]
        if label and not self.microphone:
            self.microphone = label
            self.save_meta()

    def num_recorded(self) -> int:
        if not self.recordings_dir.is_dir():
            return 0

        return sum(
            1
            for text_path in self.recordings_dir.rglob("*.txt")
            if any(text_path.with_suffix(ext).exists() for ext in AUDIO_EXTENSIONS)
        )

    def next_prompt(self, prompts: List[Prompt], skip: int = 0) -> Optional[Prompt]:
        """First unrecorded prompt, after skipping `skip` of them."""
        for prompt in prompts:
            if (self.recordings_dir / prompt.group / f"{prompt.id}.txt").exists():
                continue

            if skip <= 0:
                return prompt

            skip -= 1

        return None

    def save_recording(
        self, group: str, prompt_id: str, text: str, audio: bytes, extension: str
    ) -> None:
        for part in (group, prompt_id):
            if not re.fullmatch(r"[\w.-]+", part) or part.startswith("."):
                raise ValueError(f"Invalid prompt: {group}/{prompt_id}")

        if extension not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio type: {extension}")

        group_dir = self.recordings_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for old_ext in AUDIO_EXTENSIONS:
            (group_dir / f"{prompt_id}{old_ext}").unlink(missing_ok=True)

        (group_dir / f"{prompt_id}{extension}").write_bytes(audio)
        (group_dir / f"{prompt_id}.txt").write_text(text, encoding="utf-8")

    def import_zip(self, data: bytes) -> int:
        """Import recordings from a zip.

        Either LJSpeech style (metadata.csv with id|text plus audio files) or
        audio files next to .txt transcripts with the same name.
        """
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            audio: Dict[str, zipfile.ZipInfo] = {}
            texts: Dict[str, zipfile.ZipInfo] = {}
            metadata: Optional[zipfile.ZipInfo] = None
            for info in archive.infolist():
                path = Path(info.filename)
                if (
                    info.is_dir()
                    or path.name.startswith(".")
                    or "__MACOSX" in path.parts
                ):
                    continue

                suffix = path.suffix.lower()
                if suffix in AUDIO_EXTENSIONS:
                    audio.setdefault(path.stem, info)
                elif path.name.lower() == "metadata.csv":
                    metadata = info
                elif suffix == ".txt":
                    texts.setdefault(path.stem, info)

            pairs: List[Tuple[str, str]] = []
            if metadata is not None:
                reader = csv.reader(
                    io.StringIO(archive.read(metadata).decode("utf-8-sig")),
                    delimiter="|",
                    quoting=csv.QUOTE_NONE,
                )
                for row in reader:
                    if len(row) >= 2:
                        pairs.append((Path(row[0]).stem, row[1].strip()))
            else:
                for stem, info in texts.items():
                    pairs.append((stem, archive.read(info).decode("utf-8-sig").strip()))

            imported = 0
            for stem, text in pairs:
                info = audio.get(stem)
                if (info is None) or (not text):
                    continue

                safe_id = re.sub(r"[^\w.-]", "_", stem).lstrip(".") or "clip"
                self.save_recording(
                    "upload",
                    safe_id,
                    text,
                    archive.read(info),
                    Path(info.filename).suffix.lower(),
                )
                imported += 1

        if imported == 0:
            raise ValueError(
                "No recordings found. Expected metadata.csv (id|text) with audio "
                "files, or audio files with matching .txt transcripts."
            )

        return imported


class VoiceStore:
    """All voices under data/voices."""

    def __init__(self, voices_dir: Path) -> None:
        self.voices_dir = voices_dir
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> List[Voice]:
        voices = []
        for meta_path in self.voices_dir.glob("*/voice.json"):
            voice = self._load(meta_path.parent)
            if voice is not None:
                voices.append(voice)

        return sorted(voices, key=lambda v: v.created, reverse=True)

    def get(self, name: str) -> Voice:
        voice = self._load(self.voices_dir / name) if VOICE_NAME.match(name) else None
        if voice is None:
            raise KeyError(f"No voice named {name}")

        return voice

    def create(self, name: str, language: str, gender: str) -> Voice:
        if not VOICE_NAME.match(name):
            raise ValueError(
                "Name must be 1-40 lower case letters, numbers or underscores"
            )

        if gender not in ("female", "male"):
            raise ValueError("Gender must be female or male")

        root = self.voices_dir / name
        if root.exists():
            raise ValueError(f"A voice named {name} already exists")

        voice = Voice(
            name=name,
            language=language,
            espeak_voice=default_espeak_voice(language),
            gender=gender,
            root=root,
        )
        root.mkdir(parents=True)
        voice.save_meta()
        return voice

    def delete(self, voice: Voice) -> None:
        shutil.rmtree(voice.root)

    def _load(self, root: Path) -> Optional[Voice]:
        meta_path = root / "voice.json"
        if not meta_path.is_file():
            return None

        data = json.loads(meta_path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(Voice)} - {"root"}
        return Voice(root=root, **{k: v for k, v in data.items() if k in known})
