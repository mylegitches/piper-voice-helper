"""Background Piper voice training with piper1-gpl.

Workflow borrowed from TextyMcSpeechy: fine-tune from a pretrained checkpoint,
keep training until stopped (or an epoch limit), and export + listen to the
latest checkpoint at any time.
"""
import asyncio
import json
import logging
import os
import re
import shutil
import signal
import sys
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)
_DIR = Path(__file__).parent
_REPO_DIR = _DIR.parent

ACCELERATORS = ("auto", "gpu", "cpu")
LATEST_CHECKPOINT = "latest"
"""Special checkpoint value: continue from this voice's newest checkpoint."""

# Prompt language code (lower case) -> espeak-ng voice, where the language prefix
# alone isn't the right choice.
_ESPEAK_VOICES = {
    "en-us": "en-us",
    "en-ca": "en-us",
    "en-gb": "en-gb",
    "en-au": "en-gb",
    "en-ie": "en-gb",
    "en-in": "en-gb",
    "pt-br": "pt-br",
    "pt-pt": "pt",
    "es-mx": "es-419",
    "zh-cn": "cmn",
    "zh-tw": "cmn",
    "zh-hk": "yue",
}

_CHECKPOINT_EPOCH_SCRIPT = """
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(ckpt.get("epoch", 0))
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


def default_espeak_voice(language: str) -> str:
    """Guess an espeak-ng voice from a prompt language code like en-US."""
    code = language.lower()
    return _ESPEAK_VOICES.get(code, code.split("-", maxsplit=1)[0])


def find_train_python() -> str:
    """Python from script/setup_training if present, else this interpreter."""
    setup_python = _REPO_DIR / ".piper1-gpl" / ".venv" / "bin" / "python3"
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


# -----------------------------------------------------------------------------


@dataclass
class TrainingSettings:
    """Settings chosen on the training page."""

    voice_name: str
    espeak_voice: str
    checkpoint: str = ""
    """Path or URL to a medium quality checkpoint, "latest", or empty (scratch)."""

    sample_rate: int = 22050
    batch_size: int = 16
    epochs: int = 0
    """Epochs to train past the starting checkpoint (0 = until stopped)."""

    accelerator: str = "auto"
    test_text: str = "The quick brown fox jumped over the lazy dogs."

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TrainingSettings":
        names = {f.name for f in fields(TrainingSettings)}
        return TrainingSettings(**{k: v for k, v in data.items() if k in names})


@dataclass
class Workspace:
    """Training state and files for one user/language (output/_training/<lang>)."""

    language: str
    work_dir: Path
    settings: Optional[TrainingSettings] = None
    state: str = "idle"  # idle, running, succeeded, failed, stopped
    step: str = ""
    exporting: bool = False
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=1000))
    proc: Optional[asyncio.subprocess.Process] = None
    task: Optional["asyncio.Task[None]"] = None

    @property
    def settings_path(self) -> Path:
        return self.work_dir / "settings.json"

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
    def voice_dir(self) -> Path:
        return self.work_dir / "voice"

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
        for export_dir in self.voice_dir.glob("epoch_*"):
            files = sorted(
                p.name for p in export_dir.iterdir() if p.suffix in (".onnx", ".json")
            )
            if not any(f.endswith(".onnx") for f in files):
                continue

            exports.append(
                {
                    "epoch": int(export_dir.name.split("_", 1)[1]),
                    "dir": export_dir.name,
                    "files": files,
                    "wav": "test.wav" if (export_dir / "test.wav").exists() else None,
                }
            )

        return sorted(exports, key=lambda e: e["epoch"], reverse=True)

    def to_json(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "state": self.state,
            "step": self.step,
            "exporting": self.exporting,
            "hasCheckpoint": self.latest_checkpoint() is not None,
            "exports": self.exports(),
            "settings": asdict(self.settings) if self.settings else None,
            "log": "\n".join(self.log),
        }


class TrainingManager:
    """Runs at most one training job at a time using piper1-gpl."""

    def __init__(self, python: str) -> None:
        self.python = python
        self.workspaces: Dict[Path, Workspace] = {}

    def get(self, language: str, work_dir: Path) -> Workspace:
        workspace = self.workspaces.get(work_dir)
        if workspace is None:
            workspace = Workspace(language=language, work_dir=work_dir)
            if workspace.settings_path.is_file():
                workspace.settings = TrainingSettings.from_dict(
                    json.loads(workspace.settings_path.read_text(encoding="utf-8"))
                )

            log_path = work_dir / "train.log"
            if log_path.is_file():
                with open(log_path, "r", encoding="utf-8", errors="replace") as log:
                    workspace.log.extend(line.rstrip("\n") for line in log)

            self.workspaces[work_dir] = workspace

        return workspace

    def start(
        self, workspace: Workspace, recordings_dir: Path, settings: TrainingSettings
    ) -> None:
        for other in self.workspaces.values():
            if other.state == "running":
                raise RuntimeError(
                    f"A training job is already running for {other.language}"
                )

        if workspace.exporting:
            raise RuntimeError("Wait for the export to finish")

        workspace.work_dir.mkdir(parents=True, exist_ok=True)
        workspace.settings = settings
        workspace.settings_path.write_text(
            json.dumps(asdict(settings), indent=2), encoding="utf-8"
        )
        workspace.state = "running"
        workspace.log.clear()
        workspace.task = asyncio.create_task(self._train(workspace, recordings_dir))

    async def stop(self, workspace: Workspace) -> None:
        if workspace.state != "running":
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

        if workspace.task is not None:
            workspace.task.cancel()

    def start_export(self, workspace: Workspace) -> None:
        if workspace.exporting:
            raise RuntimeError("Already exporting")

        if workspace.latest_checkpoint() is None:
            raise RuntimeError("No checkpoint to export yet")

        if workspace.settings is None:
            raise RuntimeError("No training settings found")

        workspace.exporting = True
        asyncio.create_task(self._export_guarded(workspace))

    # -------------------------------------------------------------------------

    async def _train(self, ws: Workspace, recordings_dir: Path) -> None:
        assert ws.settings is not None
        settings = ws.settings

        try:
            # 1. Recordings -> dataset (wav/ + metadata.csv)
            ws.step = "Exporting dataset"
            if ws.dataset_dir.exists():
                shutil.rmtree(ws.dataset_dir)

            await self._exec(
                ws,
                [
                    sys.executable,
                    "-m",
                    "export_dataset",
                    str(recordings_dir),
                    str(ws.dataset_dir),
                ],
                cwd=_REPO_DIR,
            )
            metadata_path = ws.dataset_dir / "metadata.csv"
            num_utterances = sum(
                1
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if num_utterances < 1:
                raise RuntimeError("No recordings were exported")

            self._log(ws, f"Exported {num_utterances} utterance(s)")

            # 2. Starting checkpoint
            checkpoint_path: Optional[Path] = None
            if settings.checkpoint == LATEST_CHECKPOINT:
                checkpoint_path = ws.latest_checkpoint()
                if checkpoint_path is None:
                    raise RuntimeError("No previous checkpoint to continue from")
            elif settings.checkpoint:
                ws.step = "Getting base checkpoint"
                checkpoint_path = await self._get_checkpoint(ws, settings.checkpoint)

            max_epochs = -1
            if settings.epochs > 0:
                base_epoch = 0
                if checkpoint_path is not None:
                    base_epoch = await self._checkpoint_epoch(ws, checkpoint_path)

                # Epochs continue counting from the checkpoint
                max_epochs = base_epoch + settings.epochs

            # 3. Train
            ws.step = "Training"
            ws.train_dir.mkdir(parents=True, exist_ok=True)
            command = [
                self.python,
                "-c",
                _TRAIN_SCRIPT,
                "fit",
                "--model.mos_metric",
                "none",
                "--data.voice_name",
                settings.voice_name,
                "--data.csv_path",
                str(metadata_path),
                "--data.audio_dir",
                str(ws.dataset_dir / "wav"),
                "--model.sample_rate",
                str(settings.sample_rate),
                "--data.espeak_voice",
                settings.espeak_voice,
                "--data.cache_dir",
                str(
                    ws.work_dir
                    / "cache"
                    / f"{settings.espeak_voice}_{settings.sample_rate}"
                ),
                "--data.config_path",
                str(ws.config_path),
                "--data.batch_size",
                str(settings.batch_size),
                "--trainer.max_epochs",
                str(max_epochs),
                "--trainer.accelerator",
                settings.accelerator,
                "--trainer.default_root_dir",
                str(ws.train_dir),
            ]
            if checkpoint_path is not None:
                command.extend(["--ckpt_path", str(checkpoint_path)])

            self._log(
                ws,
                "Training until stopped"
                if max_epochs < 0
                else f"Training until epoch {max_epochs}",
            )
            await self._exec(ws, command)

            ws.state = "succeeded"
            ws.step = "Done"
        except asyncio.CancelledError:
            ws.state = "stopped"
            self._log(ws, "Stopped")
        except Exception as err:
            _LOGGER.exception("Training failed")
            if ws.state == "running":
                ws.state = "failed"
                self._log(ws, f"ERROR: {err}")
        finally:
            ws.proc = None
            ws.step = "" if ws.state != "succeeded" else ws.step

        # Always leave a usable voice behind when training finishes on its own
        if (ws.state == "succeeded") and (not ws.exporting):
            ws.exporting = True
            await self._export_guarded(ws)

    async def _export_guarded(self, ws: Workspace) -> None:
        try:
            await self._export(ws)
        except Exception as err:
            _LOGGER.exception("Export failed")
            self._log(ws, f"ERROR: export failed: {err}")
        finally:
            ws.exporting = False

    async def _export(self, ws: Workspace) -> None:
        """Export newest checkpoint to onnx and synthesize a test sentence."""
        assert ws.settings is not None
        checkpoint_path = ws.latest_checkpoint()
        if checkpoint_path is None:
            raise RuntimeError("No checkpoint to export")

        epoch = await self._checkpoint_epoch(ws, checkpoint_path)
        export_dir = ws.voice_dir / f"epoch_{epoch}"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Piper naming convention, e.g. en_US-my_voice-medium.onnx
        lang_code = ws.language.replace("-", "_")
        onnx_path = export_dir / f"{lang_code}-{ws.settings.voice_name}-medium.onnx"
        self._log(ws, f"[export] Exporting epoch {epoch} from {checkpoint_path}")
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

        # Fields Home Assistant expects (same fix-ups as TextyMcSpeechy's exporter)
        config = json.loads(ws.config_path.read_text(encoding="utf-8"))
        config["dataset"] = ws.settings.voice_name
        config.setdefault("audio", {})["quality"] = "medium"
        config.setdefault("language", {})["code"] = lang_code
        Path(f"{onnx_path}.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if ws.settings.test_text.strip():
            await self._exec(
                ws,
                [
                    self.python,
                    "-m",
                    "piper",
                    "-m",
                    str(onnx_path),
                    "-f",
                    str(export_dir / "test.wav"),
                    "--",
                    ws.settings.test_text.strip(),
                ],
                track=False,
            )

        self._log(ws, f"[export] Voice written to {export_dir}")

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
        self._log(ws, "$ " + " ".join(command))
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

        # Cache downloads next to the per-language work dirs.
        # Keep the voice name since some files are just "<voice>-<n>.ckpt".
        url_path = urllib.parse.urlparse(checkpoint).path
        parts = [urllib.parse.unquote(p) for p in url_path.split("/") if p]
        name = "_".join(parts[-3:])
        path = ws.work_dir.parent / "checkpoints" / name
        if path.is_file():
            self._log(ws, f"Using cached checkpoint {path}")
            return path

        self._log(ws, f"Downloading {checkpoint}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".part")

        def download() -> None:
            with urllib.request.urlopen(checkpoint) as response, open(
                tmp_path, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file, 1024 * 1024)

        await asyncio.to_thread(download)
        tmp_path.rename(path)
        self._log(ws, f"Saved checkpoint to {path}")
        return path

    def _log(self, ws: Workspace, line: str) -> None:
        ws.log.append(line)
        try:
            with open(ws.work_dir / "train.log", "a", encoding="utf-8") as log_file:
                print(line, file=log_file)
        except OSError:
            pass
