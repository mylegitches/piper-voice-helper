"""Background Piper voice training with piper1-gpl.

Workflow borrowed from TextyMcSpeechy: fine-tune from a pretrained checkpoint,
train for a while (or until stopped), and export + listen to the latest
checkpoint at any time. Finished voices work with Home Assistant's Piper.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import sys
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .voices import AUDIO_EXTENSIONS, Voice

_LOGGER = logging.getLogger(__name__)
_DIR = Path(__file__).parent
_REPO_DIR = _DIR.parent

ACCELERATORS = ("auto", "gpu", "cpu")
LATEST_CHECKPOINT = "latest"
"""Special checkpoint value: continue from this voice's newest checkpoint."""

PRESETS = {
    "quick": {"label": "Quick test", "hours": 0.5, "hint": "~30 min, rough"},
    "good": {"label": "Good", "hours": 3, "hint": "~3 hours"},
    "best": {"label": "Best", "hours": 8, "hint": "~8 hours"},
    "unlimited": {"label": "Until I stop it", "hours": 0, "hint": "no time limit"},
}
DEFAULT_PRESET = "good"

_CHECKPOINT_EPOCH_SCRIPT = """
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(ckpt.get("epoch", 0))
"""

_DEVICE_SCRIPT = """
import json, torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(json.dumps({"gpu": p.name, "memory_gb": round(p.total_memory / 2**30, 1)}))
else:
    print(json.dumps({"gpu": None, "memory_gb": 0}))
"""

# Runs "python3 -m piper.train" without the val_mos checkpoint callback.
# The MOS predictor is downloaded on first use; when that fails (offline) the
# callback aborts training after the first validation. We export by listening
# to the latest checkpoint instead, so it isn't needed.
_TRAIN_SCRIPT = """
import piper.train.__main__ as train_main
callbacks = getattr(train_main, "_DEFAULT_CALLBACKS", [])
callbacks[:] = [c for c in callbacks if getattr(c, "monitor", None) != "val_mos"]
train_main.main()
"""

# Runs "python3 -m piper.train.export_onnx" with the TorchScript-based exporter.
# Since torch 2.9 torch.onnx.export defaults to dynamo=True, which fails on
# Piper's model (piper1-gpl allows any torch 2.x).
_EXPORT_SCRIPT = """
import inspect, torch
_export = torch.onnx.export
def export(*args, **kwargs):
    if "dynamo" in inspect.signature(_export).parameters:
        kwargs.setdefault("dynamo", False)
    return _export(*args, **kwargs)
torch.onnx.export = export
from piper.train.export_onnx import main
main()
"""


def find_train_python() -> str:
    """Python with piper1-gpl training: $PIPER_PYTHON, script/setup's venv, or ours."""
    if os.environ.get("PIPER_PYTHON"):
        return os.environ["PIPER_PYTHON"]

    setup_python = _REPO_DIR / ".venv" / "bin" / "python3"
    if setup_python.exists():
        return str(setup_python)

    return sys.executable


# -----------------------------------------------------------------------------


# Multi-speaker datasets in the lists can't seed a single-speaker voice
_MULTI_SPEAKER = re.compile(
    r"^(arctic|l2arctic|libritts|libritts_r|vctk|aru|semaine|mls(_.*)?|thorsten_emotional)$"
)


@dataclass
class PretrainedCheckpoint:
    """Entry from TextyMcSpeechy's pretrained checkpoint lists."""

    group: str
    """espeak voice of the .conf file it came from (or "generic")."""

    name: str
    gender: str
    url: str


def load_checkpoint_catalog(
    checkpoints_dir: Path = _DIR / "checkpoints",
) -> Dict[str, List[PretrainedCheckpoint]]:
    """Load medium quality checkpoints from TextyMcSpeechy .conf files."""
    catalog: Dict[str, List[PretrainedCheckpoint]] = {}
    for conf_path in sorted(checkpoints_dir.glob("*.conf")):
        group = conf_path.stem
        for line in conf_path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^DEFAULT_([MF])_MED_URL=\"?([^\"\s]+)\"?", line.strip())
            if not match:
                continue

            gender, url = match.groups()
            # .../resolve/main/<lang>/<locale>/<voice>/<quality>/<file>
            parts = urllib.parse.urlparse(url).path.split("/")
            if len(parts) < 5 or _MULTI_SPEAKER.match(parts[-3]):
                continue

            name = f"{parts[-3]} ({parts[-4]})"
            entries = catalog.setdefault(group, [])
            if not any(e.url == url for e in entries):
                entries.append(
                    PretrainedCheckpoint(
                        group=group,
                        name=name,
                        gender="male" if gender == "M" else "female",
                        url=url,
                    )
                )

    return catalog


def catalog_groups(language: str, espeak_voice: str) -> List[str]:
    """Checkpoint list names that match a voice, best first."""
    candidates = [language.lower(), espeak_voice, espeak_voice.split("-")[0]]
    return list(dict.fromkeys(candidates))


def suggest_checkpoint(
    catalog: Dict[str, List[PretrainedCheckpoint]],
    language: str,
    espeak_voice: str,
    gender: str,
) -> Optional[PretrainedCheckpoint]:
    """Best starting checkpoint for a language and voice type."""
    entries: List[PretrainedCheckpoint] = []
    for group in catalog_groups(language, espeak_voice) + ["generic"]:
        entries = catalog.get(group, [])
        if entries:
            break

    if not entries:
        return None

    # Same gender first; lessac is what Piper's own docs fine-tune from
    return sorted(entries, key=lambda e: (e.gender != gender, "lessac" not in e.name))[
        0
    ]


# -----------------------------------------------------------------------------


@dataclass
class TrainingSettings:
    """Training options. Everything has a sensible default."""

    preset: str = DEFAULT_PRESET
    hours: float = PRESETS[DEFAULT_PRESET]["hours"]
    """Wall-clock limit in hours (0 = none)."""

    epochs: int = 0
    """Epochs to train past the starting checkpoint (0 = no limit)."""

    checkpoint: str = ""
    """Path/URL of a medium quality checkpoint, "latest", or empty (scratch)."""

    batch_size: int = 0
    """0 = pick from GPU memory."""

    accelerator: str = "auto"
    sample_rate: int = 22050

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TrainingSettings":
        names = {f.name for f in fields(TrainingSettings)}
        settings = TrainingSettings(**{k: v for k, v in data.items() if k in names})
        if settings.accelerator not in ACCELERATORS:
            raise ValueError(f"Invalid device: {settings.accelerator}")

        if settings.sample_rate not in (16000, 22050):
            raise ValueError("Sample rate must be 16000 or 22050")

        settings.hours = max(0.0, float(settings.hours))
        settings.epochs = max(0, int(settings.epochs))
        settings.batch_size = max(0, int(settings.batch_size))
        settings.checkpoint = str(settings.checkpoint).strip()
        return settings


@dataclass
class Workspace:
    """Training state for one voice."""

    voice: Voice
    settings: Optional[TrainingSettings] = None
    state: str = "idle"  # idle, running, succeeded, failed, stopped
    stage: str = ""  # prepare, download, train, export
    started: Optional[float] = None
    finished: Optional[float] = None
    epoch: Optional[int] = None
    exporting: bool = False
    error: str = ""
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=2000))
    log_count: int = 0
    """Total lines ever logged (for incremental streaming)."""

    proc: Optional[asyncio.subprocess.Process] = None

    @property
    def work_dir(self) -> Path:
        return self.voice.root / "training"

    @property
    def dataset_dir(self) -> Path:
        return self.work_dir / "dataset"

    @property
    def train_dir(self) -> Path:
        return self.work_dir / "train"

    @property
    def config_path(self) -> Path:
        return self.train_dir / "config.json"

    @property
    def exports_dir(self) -> Path:
        return self.voice.root / "exports"

    @property
    def running(self) -> bool:
        return self.state == "running"

    def latest_checkpoint(self) -> Optional[Path]:
        """Newest checkpoint from the most recent training run."""
        checkpoints = list(
            (self.train_dir / "lightning_logs").glob("*/checkpoints/*.ckpt")
        )
        if not checkpoints:
            return None

        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    def exports(self) -> List[Dict[str, Any]]:
        """Exported voices, newest epoch first."""
        exports = []
        for export_dir in self.exports_dir.glob("epoch_*"):
            onnx_files = list(export_dir.glob("*.onnx"))
            if not onnx_files or not Path(f"{onnx_files[0]}.json").exists():
                continue

            exports.append(
                {
                    "epoch": int(export_dir.name.split("_", 1)[1]),
                    "dir": export_dir.name,
                    "model": onnx_files[0].name,
                    "created": onnx_files[0].stat().st_mtime,
                }
            )

        return sorted(exports, key=lambda e: e["epoch"], reverse=True)

    def status(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "stage": self.stage,
            "started": self.started,
            "finished": self.finished,
            "now": time.time(),
            "epoch": self.epoch,
            "exporting": self.exporting,
            "error": self.error,
            "hasCheckpoint": self.latest_checkpoint() is not None,
            "exports": self.exports(),
            "settings": asdict(self.settings) if self.settings else None,
        }

    def lines_since(self, count: int) -> List[str]:
        """Log lines after the first `count` ever logged."""
        new = self.log_count - count
        if new <= 0:
            return []

        return list(self.log)[-min(new, len(self.log)) :]


class TrainingManager:
    """Runs at most one training job at a time."""

    def __init__(self, python: str) -> None:
        self.python = python
        self.workspaces: Dict[str, Workspace] = {}
        self.device: Dict[str, Any] = {"gpu": None, "memory_gb": 0, "ok": False}

    async def detect_device(self) -> None:
        """Check that training is installed and find the GPU."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.python,
                "-c",
                "import piper.train\n" + _DEVICE_SCRIPT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                self.device = {**json.loads(stdout.decode()), "ok": True}
            else:
                self.device["problem"] = stderr.decode(errors="replace")[-500:]
        except Exception as err:
            self.device["problem"] = str(err)

        if not self.device["ok"]:
            _LOGGER.warning("Piper training not available: %s", self.device)

    def default_batch_size(self, accelerator: str) -> int:
        memory = self.device.get("memory_gb") or 0
        if (accelerator == "cpu") or (not memory):
            return 8

        if memory >= 20:
            return 32

        return 16 if memory >= 10 else 8

    def get(self, voice: Voice) -> Workspace:
        workspace = self.workspaces.get(voice.name)
        if workspace is None:
            workspace = Workspace(voice=voice)
            log_path = workspace.work_dir / "train.log"
            if log_path.is_file():
                with open(log_path, "r", encoding="utf-8", errors="replace") as log:
                    for line in log:
                        workspace.log.append(line.rstrip("\n"))
                        workspace.log_count += 1

            self.workspaces[voice.name] = workspace

        return workspace

    def forget(self, voice: Voice) -> None:
        workspace = self.workspaces.get(voice.name)
        if (workspace is not None) and (workspace.running or workspace.exporting):
            raise RuntimeError("Voice is busy")

        self.workspaces.pop(voice.name, None)

    def busy_voice(self) -> Optional[str]:
        for name, workspace in self.workspaces.items():
            if workspace.running:
                return name

        return None

    def start(self, workspace: Workspace, settings: TrainingSettings) -> None:
        busy = self.busy_voice()
        if busy is not None:
            raise RuntimeError(f"Already training {busy}")

        if workspace.exporting:
            raise RuntimeError("Wait for the export to finish")

        if not self.device["ok"]:
            raise RuntimeError(
                "Piper training is not installed: "
                + self.device.get("problem", "unknown problem")
            )

        workspace.work_dir.mkdir(parents=True, exist_ok=True)
        workspace.settings = settings
        workspace.state = "running"
        workspace.stage = "prepare"
        workspace.started = time.time()
        workspace.finished = None
        workspace.epoch = None
        workspace.error = ""
        asyncio.create_task(self._train(workspace))

    async def stop(self, workspace: Workspace) -> None:
        if not workspace.running:
            return

        workspace.state = "stopped"
        self._log(workspace, "Stopping...")
        proc = workspace.proc
        if (proc is not None) and (proc.returncode is None):
            try:
                # Let Lightning shut down cleanly first
                os.killpg(proc.pid, signal.SIGINT)
                await asyncio.wait_for(proc.wait(), timeout=30)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def start_export(self, workspace: Workspace) -> None:
        if workspace.exporting:
            raise RuntimeError("Already exporting")

        if workspace.latest_checkpoint() is None:
            raise RuntimeError("No checkpoint to export yet")

        workspace.exporting = True
        asyncio.create_task(self._export_guarded(workspace))

    async def speak(self, workspace: Workspace, export_dir: str, text: str) -> bytes:
        """Synthesize text with an exported voice."""
        model_paths = list((workspace.exports_dir / export_dir).glob("*.onnx"))
        if not model_paths:
            raise FileNotFoundError("Voice not exported")

        proc = await asyncio.create_subprocess_exec(
            self.python,
            "-m",
            "piper",
            "-m",
            str(model_paths[0]),
            "--output-file",
            "-",  # WAV to stdout
            "--",
            text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode(errors="replace")[-500:])

        return stdout

    # -------------------------------------------------------------------------

    async def _train(self, ws: Workspace) -> None:
        assert ws.settings is not None
        settings = ws.settings
        voice = ws.voice

        try:
            # 1. Recordings -> dataset (wav/ + metadata.csv)
            if ws.dataset_dir.exists():
                shutil.rmtree(ws.dataset_dir)

            command = [
                sys.executable,
                "-m",
                "export_dataset",
                str(voice.recordings_dir),
                str(ws.dataset_dir),
            ]
            for extension in AUDIO_EXTENSIONS:
                command.extend(["--audio-glob", f"*{extension}"])

            await self._exec(ws, command, cwd=_REPO_DIR)
            self._check_running(ws)

            metadata_path = ws.dataset_dir / "metadata.csv"
            num_utterances = sum(
                1
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if num_utterances < 10:
                raise RuntimeError(
                    f"Only {num_utterances} usable recording(s); record at least 10"
                )

            self._log(ws, f"Prepared {num_utterances} recordings")

            # 2. Starting checkpoint
            checkpoint_path: Optional[Path] = None
            if settings.checkpoint == LATEST_CHECKPOINT:
                checkpoint_path = ws.latest_checkpoint()
                if checkpoint_path is None:
                    raise RuntimeError("No previous checkpoint to continue from")
            elif settings.checkpoint:
                ws.stage = "download"
                checkpoint_path = await self._get_checkpoint(ws, settings.checkpoint)

            self._check_running(ws)
            max_epochs = -1
            if settings.epochs > 0:
                base_epoch = 0
                if checkpoint_path is not None:
                    base_epoch = await self._checkpoint_epoch(ws, checkpoint_path)

                # Epochs continue counting from the checkpoint
                max_epochs = base_epoch + settings.epochs

            batch_size = settings.batch_size or self.default_batch_size(
                settings.accelerator
            )
            # Keep a few batches per epoch on small datasets
            batch_size = max(1, min(batch_size, num_utterances // 4))

            # 3. Train
            ws.stage = "train"
            ws.train_dir.mkdir(parents=True, exist_ok=True)
            command = [
                self.python,
                "-c",
                _TRAIN_SCRIPT,
                "fit",
                "--model.mos_metric",
                "none",
                "--data.voice_name",
                voice.name,
                "--data.csv_path",
                str(metadata_path),
                "--data.audio_dir",
                str(ws.dataset_dir / "wav"),
                "--model.sample_rate",
                str(settings.sample_rate),
                "--data.espeak_voice",
                voice.espeak_voice,
                "--data.cache_dir",
                str(
                    ws.work_dir
                    / "cache"
                    / f"{voice.espeak_voice}_{settings.sample_rate}"
                ),
                "--data.config_path",
                str(ws.config_path),
                "--data.batch_size",
                str(batch_size),
                "--trainer.max_epochs",
                str(max_epochs),
                "--trainer.accelerator",
                settings.accelerator,
                "--trainer.default_root_dir",
                str(ws.train_dir),
            ]
            if settings.hours > 0:
                minutes = max(1, round(settings.hours * 60))
                command.extend(
                    [
                        "--trainer.max_time",
                        f"{minutes // 1440:02}:{minutes // 60 % 24:02}:{minutes % 60:02}:00",
                    ]
                )

            if checkpoint_path is not None:
                command.extend(["--ckpt_path", str(checkpoint_path)])

            self._log(
                ws,
                f"Training with batch size {batch_size}"
                + (f" for {settings.hours:g} hour(s)" if settings.hours > 0 else "")
                + (f" until epoch {max_epochs}" if max_epochs > 0 else ""),
            )
            await self._exec(ws, command)
            ws.state = "succeeded"
        except Exception as err:
            if ws.running:
                _LOGGER.exception("Training failed")
                ws.state = "failed"
                ws.error = str(err)
                self._log(ws, f"ERROR: {err}")
        finally:
            ws.proc = None
            ws.finished = time.time()

        # Leave a usable voice behind whenever training ran
        if (
            (ws.state in ("succeeded", "stopped"))
            and (ws.latest_checkpoint() is not None)
            and (not ws.exporting)
        ):
            ws.exporting = True
            await self._export_guarded(ws)

        ws.stage = ""

    def _check_running(self, ws: Workspace) -> None:
        if not ws.running:
            raise RuntimeError("Stopped")

    async def _export_guarded(self, ws: Workspace) -> None:
        stage = ws.stage
        ws.stage = "export"
        try:
            await self._export(ws)
        except Exception as err:
            _LOGGER.exception("Export failed")
            ws.error = f"Export failed: {err}"
            self._log(ws, f"ERROR: export failed: {err}")
        finally:
            ws.exporting = False
            ws.stage = stage if ws.running else ""

    async def _export(self, ws: Workspace) -> None:
        """Export newest checkpoint to onnx for Piper / Home Assistant."""
        voice = ws.voice
        checkpoint_path = ws.latest_checkpoint()
        if checkpoint_path is None:
            raise RuntimeError("No checkpoint to export")

        epoch = await self._checkpoint_epoch(ws, checkpoint_path)
        export_dir = ws.exports_dir / f"epoch_{epoch}"
        tmp_dir = ws.exports_dir / f".epoch_{epoch}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        tmp_dir.mkdir(parents=True)

        # Piper naming convention, e.g. en_US-my_voice-medium.onnx
        onnx_path = tmp_dir / f"{voice.model_stem}.onnx"
        self._log(ws, f"Exporting epoch {epoch}")
        await self._exec(
            ws,
            [
                self.python,
                "-c",
                _EXPORT_SCRIPT,
                "--checkpoint",
                str(checkpoint_path),
                "--output-file",
                str(onnx_path),
            ],
            track=False,
        )

        # Fields Home Assistant expects (as in TextyMcSpeechy's exporter)
        config = json.loads(ws.config_path.read_text(encoding="utf-8"))
        config["dataset"] = voice.name
        config.setdefault("audio", {})["quality"] = "medium"
        config.setdefault("language", {})["code"] = voice.language.replace("-", "_")
        Path(f"{onnx_path}.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if export_dir.exists():
            shutil.rmtree(export_dir)

        tmp_dir.rename(export_dir)
        self._log(ws, f"Voice ready: epoch {epoch}")

    async def _checkpoint_epoch(self, ws: Workspace, checkpoint_path: Path) -> int:
        """Read the epoch stored in a checkpoint (falls back to its file name)."""
        proc = await asyncio.create_subprocess_exec(
            self.python,
            "-c",
            _CHECKPOINT_EPOCH_SCRIPT,
            str(checkpoint_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            try:
                return int(stdout.decode().strip().splitlines()[-1])
            except (ValueError, IndexError):
                pass

        match = re.search(r"epoch=(\d+)", checkpoint_path.name)
        if match:
            return int(match.group(1))

        self._log(
            ws,
            f"Could not read epoch from {checkpoint_path}: "
            + stderr.decode(errors="replace").strip()[-200:],
        )
        return 0

    async def _exec(
        self,
        ws: Workspace,
        command: List[str],
        cwd: Optional[Path] = None,
        track: bool = True,
    ) -> None:
        """Run a command, streaming its output into the log.

        Only tracked processes (training) are interrupted by stop().
        """
        self._log(
            ws, "$ " + " ".join(c if "\n" not in c else "<script>" for c in command)
        )
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # so stop() can signal the whole group
        )
        if track:
            ws.proc = proc

        assert proc.stdout is not None

        # Progress bars use \r, so split on that too.
        buffer = b""
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break

            buffer += chunk
            *lines, buffer = re.split(rb"[\r\n]", buffer)
            for line in lines:
                if line.strip():
                    self._log(ws, line.decode("utf-8", errors="replace"))

        if buffer.strip():
            self._log(ws, buffer.decode("utf-8", errors="replace"))

        return_code = await proc.wait()
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}")

    async def _get_checkpoint(self, ws: Workspace, checkpoint: str) -> Path:
        if not re.match(r"^https?://", checkpoint):
            path = Path(checkpoint).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path}")

            return path

        # Shared download cache; keep the voice name since some files are
        # just "<voice>-<n>.ckpt".
        url_path = urllib.parse.urlparse(checkpoint).path
        parts = [urllib.parse.unquote(p) for p in url_path.split("/") if p]
        path = ws.voice.root.parent.parent / "checkpoints" / "_".join(parts[-3:])
        if path.is_file():
            self._log(ws, f"Using cached base voice {path.name}")
            return path

        self._log(ws, f"Downloading base voice {checkpoint}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".part")

        def download() -> None:
            with urllib.request.urlopen(checkpoint) as response, open(
                tmp_path, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file, 1024 * 1024)

        await asyncio.to_thread(download)
        tmp_path.rename(path)
        self._log(ws, f"Saved base voice to {path}")
        return path

    def _log(self, ws: Workspace, line: str) -> None:
        ws.log.append(line)
        ws.log_count += 1
        match = re.match(r"^Epoch (\d+):", line)
        if match:
            ws.epoch = int(match.group(1))

        try:
            ws.work_dir.mkdir(parents=True, exist_ok=True)
            with open(ws.work_dir / "train.log", "a", encoding="utf-8") as log_file:
                print(line, file=log_file)
        except OSError:
            pass
