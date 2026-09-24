import argparse
import asyncio
import csv
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import hypercorn
from quart import (
    Quart,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from .training import (
    ACCELERATORS,
    BACKENDS,
    TrainingManager,
    TrainingSettings,
    default_espeak_voice,
)

_LOGGER = logging.getLogger(__name__)
_DIR = Path(__file__).parent
_DEFAULT_CHECKPOINT = (
    "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
    "en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt"
)


@dataclass
class Prompt:
    """Single prompt for the user to read."""

    group: str
    id: str
    text: str


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    #
    parser.add_argument(
        "--prompts",
        help="Path to prompts directory",
        action="append",
        default=[_DIR.parent / "prompts"],
    )
    parser.add_argument(
        "--output",
        help="Path to output directory",
        default=_DIR.parent / "output",
    )
    #
    parser.add_argument(
        "--multi-user",
        action="store_true",
        help="Require login code and user output directory to exist",
    )
    parser.add_argument("--cc0", action="store_true", help="Show public domain notice")
    #
    parser.add_argument(
        "--train-python",
        default=sys.executable,
        help="Python interpreter with Piper training installed (default: this one)",
    )
    parser.add_argument(
        "--train-backend",
        choices=BACKENDS,
        default="piper1",
        help="piper1 = OHF-Voice/piper1-gpl (piper.train), legacy = rhasspy/piper (piper_train)",
    )
    parser.add_argument(
        "--default-checkpoint",
        default=_DEFAULT_CHECKPOINT,
        help="Checkpoint path/URL pre-filled on the training page",
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    prompts_dirs = [Path(p) for p in args.prompts]
    output_dir = Path(args.output)
    css_dir = _DIR / "css"
    js_dir = _DIR / "js"
    img_dir = _DIR / "img"
    webfonts_dir = _DIR / "webfonts"

    prompts, languages = load_prompts(prompts_dirs)
    language_names = {code: name for name, code in languages.items()}

    app = Quart("piper", template_folder=str(_DIR / "templates"))
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 mb
    app.secret_key = str(uuid4())

    trainer = TrainingManager(python=args.train_python, backend=args.train_backend)

    def get_user_dir(user_id: Optional[str], language: str) -> Path:
        """Directory holding a user's recordings (output/ or output/user_<id>)."""
        if not args.multi_user:
            return output_dir

        user_dir = output_dir / f"user_{user_id}"
        if (not user_id) or (not (user_dir / language).is_dir()):
            _LOGGER.warning("No user/language directory: %s", user_dir / language)
            raise RuntimeError("Invalid login code")

        return user_dir

    def get_language(language: str) -> str:
        if language not in prompts:
            raise ValueError(f"Unknown language: {language}")

        return language

    @app.route("/")
    @app.route("/index.html")
    async def api_index() -> str:
        """Main page"""
        return await render_template(
            "index.html",
            languages=sorted(languages.items()),
            multi_user=args.multi_user,
            cc0=args.cc0,
        )

    @app.route("/done.html")
    async def api_done() -> str:
        return await render_template(
            "done.html",
            language=request.args.get("language", ""),
            user_id=request.args.get("userId", ""),
        )

    @app.route("/record")
    async def api_record() -> str:
        """Record audio for a text prompt"""
        language = request.args["language"]

        if args.multi_user:
            user_id = request.args.get("userId")
            audio_dir = output_dir / f"user_{user_id}"
            user_dir = audio_dir / language
            if not user_dir.is_dir():
                _LOGGER.warning("No user/language directory: %s", user_dir)
                user_id = None
        else:
            user_id = None
            audio_dir = output_dir

        next_prompt, num_complete, num_items = get_next_prompt(
            prompts,
            audio_dir,
            language,
        )
        if next_prompt is None:
            return await render_template(
                "done.html", language=language, user_id=user_id or ""
            )

        complete_percent = 100 * (num_complete / num_items if num_items > 0 else 1)
        return await render_template(
            "record.html",
            language=language,
            prompt_group=next_prompt.group,
            prompt_id=next_prompt.id,
            text=next_prompt.text,
            num_complete=num_complete,
            num_items=num_items,
            complete_percent=complete_percent,
            multi_user=args.multi_user,
            user_id=user_id,
        )

    @app.route("/submit", methods=["POST"])
    async def api_submit() -> Response:
        """Submit audio for a text prompt"""
        form = await request.form
        language = form["language"]
        prompt_group = form["promptGroup"]
        prompt_id = form["promptId"]
        prompt_text = form["text"]
        audio_format = form["format"]

        files = await request.files
        assert "audio" in files, "No audio"

        suffix = ".webm"
        if "wav" in audio_format:
            suffix = ".wav"

        if args.multi_user:
            user_id = form["userId"]
            user_dir = output_dir / f"user_{user_id}"
            if not user_dir.is_dir():
                _LOGGER.warning("No user/language directory: %s", user_dir)
                raise ValueError("Invalid login code")

            audio_dir = user_dir
        else:
            audio_dir = output_dir

        # Save audio and transcription
        audio_path = audio_dir / language / prompt_group / f"{prompt_id}{suffix}"
        _LOGGER.debug("Saving to %s", audio_path)

        audio_path.parent.mkdir(parents=True, exist_ok=True)
        await files["audio"].save(audio_path)

        text_path = audio_path.parent / f"{prompt_id}.txt"
        text_path.write_text(prompt_text, encoding="utf-8")

        # Get next prompt
        next_prompt, num_complete, num_items = get_next_prompt(
            prompts,
            audio_dir,
            language,
        )
        if next_prompt is None:
            return jsonify({"done": True})

        complete_percent = 100 * (num_complete / num_items if num_items > 0 else 1)
        return jsonify(
            {
                "done": False,
                "promptGroup": next_prompt.group,
                "promptId": next_prompt.id,
                "promptText": next_prompt.text,
                "numComplete": num_complete,
                "numItems": num_items,
                "completePercent": complete_percent,
            }
        )

    @app.route("/upload")
    async def api_upload() -> str:
        """Upload an existing dataset"""
        language = request.args["language"]

        if args.multi_user:
            user_id = request.args.get("userId")
            audio_dir = output_dir / f"user_{user_id}"
            user_dir = audio_dir / language
            if not user_dir.is_dir():
                _LOGGER.warning("No user/language directory: %s", user_dir)
                raise RuntimeError("Invalid login code")
        else:
            user_id = None
            audio_dir = output_dir

        return await render_template(
            "upload.html",
            language=language,
            multi_user=args.multi_user,
            user_id=user_id,
        )

    @app.route("/dataset", methods=["POST"])
    async def api_dataset() -> str:
        """Upload an existing dataset"""
        form = await request.form
        language = form["language"]

        if args.multi_user:
            user_id = form.get("userId")
            audio_dir = output_dir / f"user_{user_id}"
            user_dir = audio_dir / language
            if not user_dir.is_dir():
                _LOGGER.warning("No user/language directory: %s", user_dir)
                raise RuntimeError("Invalid login code")
        else:
            user_id = None
            audio_dir = output_dir

        files = await request.files
        dataset_file = files["dataset"]
        upload_path = user_dir / "_uploads" / Path(dataset_file.filename).name
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        await dataset_file.save(upload_path)
        _LOGGER.debug("Saved dataset to %s", upload_path)

        return await render_template("done.html")

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    @app.route("/train")
    async def api_train() -> str:
        """Training page for a language"""
        language = get_language(request.args["language"])
        user_id = request.args.get("userId")
        user_dir = get_user_dir(user_id, language)
        recordings_dir = user_dir / language
        num_recorded = (
            sum(1 for _ in recordings_dir.rglob("*.txt"))
            if recordings_dir.is_dir()
            else 0
        )

        job = trainer.get_job(user_dir / "_training" / language)
        settings = (
            job.settings
            if job is not None
            else TrainingSettings(
                voice_name="my_voice",
                espeak_voice=default_espeak_voice(language),
                checkpoint=args.default_checkpoint,
            )
        )

        return await render_template(
            "train.html",
            language=language,
            language_name=language_names.get(language, language),
            user_id=user_id or "",
            num_recorded=num_recorded,
            num_items=len(prompts[language]),
            settings=settings,
            accelerators=ACCELERATORS,
            backend=args.train_backend,
        )

    @app.route("/train/start", methods=["POST"])
    async def api_train_start() -> Response:
        form = await request.form
        language = get_language(form["language"])
        user_dir = get_user_dir(form.get("userId"), language)

        voice_name = form["voiceName"].strip()
        if not re.fullmatch(r"[A-Za-z0-9_]+", voice_name):
            raise ValueError("Voice name may only contain letters, numbers, and _")

        accelerator = form.get("accelerator", "auto")
        if accelerator not in ACCELERATORS:
            raise ValueError(f"Invalid accelerator: {accelerator}")

        settings = TrainingSettings(
            voice_name=voice_name,
            espeak_voice=form["espeakVoice"].strip(),
            checkpoint=form.get("checkpoint", "").strip(),
            sample_rate=int(form.get("sampleRate", 22050)),
            batch_size=int(form.get("batchSize", 32)),
            epochs=int(form.get("epochs", 1000)),
            accelerator=accelerator,
        )
        job = trainer.start(
            language,
            recordings_dir=user_dir / language,
            work_dir=user_dir / "_training" / language,
            settings=settings,
        )
        return jsonify(job.to_json())

    @app.route("/train/status")
    async def api_train_status() -> Response:
        language = get_language(request.args["language"])
        user_dir = get_user_dir(request.args.get("userId"), language)
        job = trainer.get_job(user_dir / "_training" / language)
        if job is None:
            return jsonify({"state": "idle"})

        return jsonify(job.to_json())

    @app.route("/train/stop", methods=["POST"])
    async def api_train_stop() -> Response:
        form = await request.form
        language = get_language(form["language"])
        user_dir = get_user_dir(form.get("userId"), language)
        await trainer.stop(user_dir / "_training" / language)
        return jsonify({"ok": True})

    @app.route("/train/download")
    async def api_train_download() -> Response:
        language = get_language(request.args["language"])
        user_dir = get_user_dir(request.args.get("userId"), language)
        voice_dir = user_dir / "_training" / language / "voice"
        return await send_from_directory(
            voice_dir, request.args["file"], as_attachment=True
        )

    @app.errorhandler(Exception)
    async def handle_error(err) -> Tuple[str, int]:
        """Return error as text."""
        _LOGGER.exception(err)
        return (f"{err.__class__.__name__}: {err}", 500)

    @app.route("/css/<path:filename>", methods=["GET"])
    async def css(filename) -> Response:
        """CSS static endpoint."""
        return await send_from_directory(css_dir, filename)

    @app.route("/js/<path:filename>", methods=["GET"])
    async def js(filename) -> Response:
        """Javascript static endpoint."""
        return await send_from_directory(js_dir, filename)

    @app.route("/img/<path:filename>", methods=["GET"])
    async def img(filename) -> Response:
        """Image static endpoint."""
        return await send_from_directory(img_dir, filename)

    @app.route("/webfonts/<path:filename>", methods=["GET"])
    async def webfonts(filename) -> Response:
        """Webfonts static endpoint."""
        return await send_from_directory(webfonts_dir, filename)

    # Run web server
    hyp_config = hypercorn.config.Config()
    hyp_config.bind = [f"{args.host}:{args.port}"]

    asyncio.run(hypercorn.asyncio.serve(app, hyp_config))


# -----------------------------------------------------------------------------


def load_prompts(
    prompts_dirs: List[Path],
) -> Tuple[Dict[str, List[Prompt]], Dict[str, str]]:
    prompts = defaultdict(list)
    languages = {}

    for prompts_dir in prompts_dirs:
        for language_dir in prompts_dir.iterdir():
            if not language_dir.is_dir():
                continue

            name, code = language_dir.name.rsplit("_", maxsplit=1)
            languages[name] = code
            for prompt_path in language_dir.glob("*.txt"):
                _LOGGER.debug("Loading prompts from %s", prompt_path)
                prompt_group = prompt_path.stem
                with open(prompt_path, "r", encoding="utf-8") as prompt_file:
                    reader = csv.reader(prompt_file, delimiter="\t")
                    for i, row in enumerate(reader):
                        if len(row) == 1:
                            prompt_id = str(i)
                        else:
                            prompt_id = row[0]

                        prompts[code].append(
                            Prompt(group=prompt_group, id=prompt_id, text=row[-1])
                        )

    return prompts, languages


def get_next_prompt(
    prompts: Dict[str, List[Prompt]],
    output_dir: Path,
    language: str,
):
    language_prompts = prompts[language]
    language_dir = output_dir / language
    incomplete_prompts = []
    for prompt in language_prompts:
        text_path = language_dir / prompt.group / f"{prompt.id}.txt"
        if not text_path.exists():
            incomplete_prompts.append(prompt)

    num_items = len(language_prompts)
    num_complete = num_items - len(incomplete_prompts)

    if incomplete_prompts:
        next_prompt = incomplete_prompts[0]
    else:
        next_prompt = None

    return next_prompt, num_complete, num_items


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
