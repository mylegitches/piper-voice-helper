"""Background training jobs: export dataset -> train Piper voice -> export onnx."""
import asyncio
import logging
import os
import re
import shutil
import signal
import sys
import time
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)
_DIR = Path(__file__).parent
_REPO_DIR = _DIR.parent

BACKENDS = ("piper1", "legacy")
ACCELERATORS = ("auto", "gpu", "cpu")

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


def default_espeak_voice(language: str) -> str:
    """Guess an espeak-ng voice from a prompt language code like en-US."""
    code = language.lower()
    return _ESPEAK_VOICES.get(code, code.split("-", maxsplit=1)[0])


@dataclass
class TrainingSettings:
    """Settings chosen on the training page."""

    voice_name: str
    espeak_voice: str
    checkpoint: str = ""
    """Path or URL to a (medium quality) Piper checkpoint to fine-tune from."""

    sample_rate: int = 22050
    batch_size: int = 32
    epochs: int = 1000
    """Epochs to train, on top of the checkpoint's own epoch count."""

    accelerator: str = "auto"


@dataclass
class TrainingJob:
    """State of one training run for a user/language."""

    language: str
    work_dir: Path
    settings: TrainingSettings
    state: str = "running"  # running, succeeded, failed, stopped
    step: str = ""
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    outputs: List[str] = field(default_factory=list)
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=1000))
    proc: Optional[asyncio.subprocess.Process] = None
    task: Optional["asyncio.Task[None]"] = None

    @property
    def voice_dir(self) -> Path:
        return self.work_dir / "voice"

    def to_json(self) -> Dict:
        return {
            "language": self.language,
            "state": self.state,
            "step": self.step,
            "started": self.started,
            "finished": self.finished,
            "outputs": self.outputs,
            "settings": asdict(self.settings),
            "log": "\n".join(self.log),
        }


class TrainingManager:
    """Runs at most one training job at a time."""

    def __init__(self, python: str, backend: str) -> None:
        self.python = python
        self.backend = backend
        self.jobs: Dict[Path, TrainingJob] = {}

    @property
    def running_job(self) -> Optional[TrainingJob]:
        for job in self.jobs.values():
            if job.state == "running":
                return job

        return None

    def get_job(self, work_dir: Path) -> Optional[TrainingJob]:
        return self.jobs.get(work_dir)

    def start(
        self,
        language: str,
        recordings_dir: Path,
        work_dir: Path,
        settings: TrainingSettings,
    ) -> TrainingJob:
        running = self.running_job
        if running is not None:
            raise RuntimeError(
                f"A training job is already running for {running.language}"
            )

        job = TrainingJob(language=language, work_dir=work_dir, settings=settings)
        self.jobs[work_dir] = job
        job.task = asyncio.create_task(self._run(job, recordings_dir))
        return job

    async def stop(self, work_dir: Path) -> None:
        job = self.jobs.get(work_dir)
        if (job is None) or (job.state != "running"):
            return

        job.state = "stopped"
        self._log(job, "Stopping...")
        if (job.proc is not None) and (job.proc.returncode is None):
            try:
                os.killpg(job.proc.pid, signal.SIGINT)
                await asyncio.wait_for(job.proc.wait(), timeout=30)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    os.killpg(job.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        if job.task is not None:
            job.task.cancel()

    # -------------------------------------------------------------------------

    async def _run(self, job: TrainingJob, recordings_dir: Path) -> None:
        settings = job.settings
        work_dir = job.work_dir
        dataset_dir = work_dir / "dataset"
        train_dir = work_dir / "train"
        config_path = train_dir / "config.json"

        try:
            work_dir.mkdir(parents=True, exist_ok=True)

            # 1. Recordings -> LJSpeech dataset (wav/ + metadata.csv)
            job.step = "Exporting dataset"
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)

            await self._exec(
                job,
                [
                    sys.executable,
                    "-m",
                    "export_dataset",
                    str(recordings_dir),
                    str(dataset_dir),
                ],
                cwd=_REPO_DIR,
            )
            metadata_path = dataset_dir / "metadata.csv"
            num_utterances = sum(
                1
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if num_utterances < 1:
                raise RuntimeError("No recordings were exported")

            self._log(job, f"Exported {num_utterances} utterance(s)")

            # 2. Base checkpoint
            checkpoint_path: Optional[Path] = None
            if settings.checkpoint:
                job.step = "Getting base checkpoint"
                checkpoint_path = await self._get_checkpoint(job, settings.checkpoint)

            # Epoch counts continue from the checkpoint, so offset max epochs.
            base_epoch = 0
            if checkpoint_path is not None:
                match = re.search(r"epoch=(\d+)", checkpoint_path.name)
                if match:
                    base_epoch = int(match.group(1))

            max_epochs = base_epoch + settings.epochs
            train_dir.mkdir(parents=True, exist_ok=True)

            # 3. Train
            if self.backend == "legacy":
                job.step = "Preprocessing"
                await self._exec(
                    job,
                    [
                        self.python,
                        "-m",
                        "piper_train.preprocess",
                        "--language",
                        settings.espeak_voice,
                        "--input-dir",
                        str(dataset_dir),
                        "--output-dir",
                        str(train_dir),
                        "--dataset-format",
                        "ljspeech",
                        "--single-speaker",
                        "--sample-rate",
                        str(settings.sample_rate),
                    ],
                )

                job.step = "Training"
                command = [
                    self.python,
                    "-m",
                    "piper_train",
                    "--dataset-dir",
                    str(train_dir),
                    "--accelerator",
                    settings.accelerator,
                    "--devices",
                    "1",
                    "--batch-size",
                    str(settings.batch_size),
                    "--validation-split",
                    "0.0",
                    "--num-test-examples",
                    "0",
                    "--max_epochs",
                    str(max_epochs),
                    "--checkpoint-epochs",
                    "1",
                    "--precision",
                    "32",
                    "--quality",
                    "medium",
                ]
                if checkpoint_path is not None:
                    command.extend(["--resume_from_checkpoint", str(checkpoint_path)])
            else:
                job.step = "Training"
                command = [
                    self.python,
                    "-m",
                    "piper.train",
                    "fit",
                    "--data.voice_name",
                    settings.voice_name,
                    "--data.csv_path",
                    str(metadata_path),
                    "--data.audio_dir",
                    str(dataset_dir / "wav"),
                    "--model.sample_rate",
                    str(settings.sample_rate),
                    "--data.espeak_voice",
                    settings.espeak_voice,
                    "--data.cache_dir",
                    str(work_dir / "cache"),
                    "--data.config_path",
                    str(config_path),
                    "--data.batch_size",
                    str(settings.batch_size),
                    "--trainer.max_epochs",
                    str(max_epochs),
                    "--trainer.accelerator",
                    settings.accelerator,
                    "--trainer.default_root_dir",
                    str(train_dir),
                ]
                if checkpoint_path is not None:
                    command.extend(["--ckpt_path", str(checkpoint_path)])

            self._log(job, f"Training until epoch {max_epochs}")
            await self._exec(job, command)

            # 4. Export latest checkpoint to onnx
            job.step = "Exporting voice"
            checkpoints = sorted(
                (train_dir / "lightning_logs").rglob("*.ckpt"),
                key=lambda p: p.stat().st_mtime,
            )
            if not checkpoints:
                raise RuntimeError("Training produced no checkpoints")

            job.voice_dir.mkdir(parents=True, exist_ok=True)
            voice_stem = (
                f"{job.language.replace('-', '_')}-{settings.voice_name}-medium"
            )
            onnx_path = job.voice_dir / f"{voice_stem}.onnx"
            if self.backend == "legacy":
                command = [
                    self.python,
                    "-m",
                    "piper_train.export_onnx",
                    str(checkpoints[-1]),
                    str(onnx_path),
                ]
            else:
                command = [
                    self.python,
                    "-m",
                    "piper.train.export_onnx",
                    "--checkpoint",
                    str(checkpoints[-1]),
                    "--output-file",
                    str(onnx_path),
                ]

            await self._exec(job, command)
            shutil.copy(config_path, onnx_path.with_suffix(".onnx.json"))

            job.outputs = [onnx_path.name, f"{onnx_path.name}.json"]
            job.state = "succeeded"
            job.step = "Done"
            self._log(job, f"Voice written to {job.voice_dir}")
        except asyncio.CancelledError:
            job.state = "stopped"
            self._log(job, "Stopped")
        except Exception as err:
            _LOGGER.exception("Training failed")
            if job.state == "running":
                job.state = "failed"
                self._log(job, f"ERROR: {err}")
        finally:
            job.proc = None
            job.finished = time.time()

    async def _exec(
        self, job: TrainingJob, command: List[str], cwd: Optional[Path] = None
    ) -> None:
        """Run a command, streaming its output into the job log."""
        self._log(job, "$ " + " ".join(command))
        job.proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # so stop() can signal the whole group
        )
        assert job.proc.stdout is not None

        # Progress bars use \r, so split on that too.
        buffer = b""
        while True:
            chunk = await job.proc.stdout.read(4096)
            if not chunk:
                break

            buffer += chunk
            *lines, buffer = re.split(rb"[\r\n]", buffer)
            for line in lines:
                if line.strip():
                    self._log(job, line.decode("utf-8", errors="replace"))

        if buffer.strip():
            self._log(job, buffer.decode("utf-8", errors="replace"))

        return_code = await job.proc.wait()
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}")

    async def _get_checkpoint(self, job: TrainingJob, checkpoint: str) -> Path:
        if not re.match(r"^https?://", checkpoint):
            path = Path(checkpoint).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path}")

            return path

        # Cache downloads next to the per-language work dirs
        cache_dir = job.work_dir.parent / "checkpoints"
        name = urllib.request.url2pathname(checkpoint.split("?")[0].rsplit("/", 1)[-1])
        path = cache_dir / name
        if path.is_file():
            self._log(job, f"Using cached checkpoint {path}")
            return path

        self._log(job, f"Downloading {checkpoint}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".part")

        def download() -> None:
            with urllib.request.urlopen(checkpoint) as response, open(
                tmp_path, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file, 1024 * 1024)

        await asyncio.to_thread(download)
        tmp_path.rename(path)
        self._log(job, f"Saved checkpoint to {path}")
        return path

    def _log(self, job: TrainingJob, line: str) -> None:
        job.log.append(line)
        try:
            with open(job.work_dir / "train.log", "a", encoding="utf-8") as log_file:
                print(line, file=log_file)
        except OSError:
            pass
