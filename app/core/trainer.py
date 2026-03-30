"""
GPT-SoVITS training pipeline — drives GPT-SoVITS's own scripts via subprocess,
exactly as webui.py does, so the pipeline is always in sync with the upstream code.

Pipeline:
  1. Slice audio into 3-10s segments (pydub)
  2. Transcribe with Whisper → list_file.txt
  3. BERT feature extraction   (1-get-text.py)
  4. HuBERT + 32kHz WAV        (2-get-hubert-wav32k.py)
  5. Semantic tokens            (3-get-semantic.py)
  6. Fine-tune GPT (s1)         (s1_train.py)
  7. Fine-tune SoVITS (s2)      (s2_train.py)
  8. Collect output models → storage/models/{voice_id}/
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Callable, List, Optional

import yaml

logger = logging.getLogger(__name__)


class GPTSoVITSTrainer:
    """
    Drives GPT-SoVITS fine-tuning through subprocess calls to the official
    training scripts, passing parameters via environment variables and config files.
    """

    def __init__(
        self,
        gptsovits_dir: str = "GPT-SoVITS",
        pretrained_dir: str = "storage/pretrained_models/GPT-SoVITS",
        output_dir: str = "storage/models",
    ) -> None:
        self.gptsovits_dir = Path(gptsovits_dir).resolve()
        self.gptsovits_inner = self.gptsovits_dir / "GPT_SoVITS"
        self.pretrained_dir = Path(pretrained_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.python = sys.executable

    # ── Guards ──────────────────────────────────────────────────────────────

    def _check_ready(self) -> None:
        if not self.gptsovits_dir.exists():
            raise RuntimeError(
                f"GPT-SoVITS directory not found: {self.gptsovits_dir}\n"
                "Run: setup\\clone_gptsovits.bat"
            )
        for name in ["pretrained_s1.ckpt", "pretrained_s2G.pth", "pretrained_s2D.pth"]:
            p = self.pretrained_dir / name
            if not p.exists():
                raise RuntimeError(
                    f"Pretrained model not found: {p}\n"
                    "Run: conda run -n voice-cloning python setup/download_models.py"
                )
        for name in ["chinese-hubert-base", "chinese-roberta-wwm-ext-large"]:
            p = self.pretrained_dir / name / "config.json"
            if not p.exists():
                raise RuntimeError(
                    f"Pretrained model not found: {self.pretrained_dir / name}\n"
                    "Run: conda run -n voice-cloning python setup/download_models.py"
                )

    # ── Step 1: audio slicing ────────────────────────────────────────────────

    def _slice_audio(
        self,
        audio_files: List[str],
        work_dir: Path,
        speaker_name: str,
        cb: Optional[Callable],
    ) -> List[Path]:
        try:
            from pydub import AudioSegment
            from pydub.silence import split_on_silence
        except ImportError:
            raise RuntimeError("pydub is required. pip install pydub")

        sliced_dir = work_dir / "sliced"
        sliced_dir.mkdir(parents=True, exist_ok=True)
        segments: List[Path] = []
        idx = 0

        for audio_file in audio_files:
            try:
                audio = AudioSegment.from_file(audio_file)
                audio = audio.set_channels(1).set_frame_rate(32000)
                chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40, keep_silence=200)
                if not chunks:
                    chunks = [audio[i:i+8000] for i in range(0, len(audio), 8000)]
                for chunk in chunks:
                    dur = len(chunk) / 1000.0
                    if dur < 3.0:
                        continue
                    if dur > 10.0:
                        for sub in [chunk[i:i+8000] for i in range(0, len(chunk), 8000)]:
                            if len(sub) / 1000.0 < 3.0:
                                continue
                            p = sliced_dir / f"{speaker_name}_{idx:04d}.wav"
                            sub.export(str(p), format="wav")
                            segments.append(p)
                            idx += 1
                    else:
                        p = sliced_dir / f"{speaker_name}_{idx:04d}.wav"
                        chunk.export(str(p), format="wav")
                        segments.append(p)
                        idx += 1
                    if idx >= 50:
                        break
            except Exception as e:
                logger.warning(f"Slice failed {audio_file}: {e}")

        if not segments:
            raise RuntimeError("No usable 3-10s segments found. Provide 1-3 min of clean audio.")

        logger.info(f"Sliced {len(audio_files)} files → {len(segments)} segments")
        return segments

    # ── Step 2: Whisper transcription ─────────────────────────────────────

    def _transcribe(
        self,
        segments: List[Path],
        language: str,
        speaker: str,
        work_dir: Path,
        whisper_model_size: str = "medium",
    ) -> Path:
        try:
            import whisper
        except ImportError:
            raise RuntimeError("openai-whisper is required. pip install openai-whisper")

        logger.info(f"Loading Whisper '{whisper_model_size}'...")
        model = whisper.load_model(whisper_model_size)

        lang_code = language if language in ("zh", "en", "ja", "ko") else "zh"
        lines = []
        ref_wav: Optional[Path] = None

        for seg in segments:
            try:
                result = model.transcribe(str(seg), language=lang_code, fp16=False)
                text = result["text"].strip()
                if not text:
                    continue
                text = text.replace("|", "，")
                lines.append(f"{seg}|{speaker}|{lang_code}|{text}")
                if ref_wav is None:
                    ref_wav = seg
            except Exception as e:
                logger.warning(f"Transcription failed {seg.name}: {e}")

        if not lines:
            raise RuntimeError("Transcription produced no output. Check audio quality.")

        list_file = work_dir / "list_file.txt"
        list_file.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Transcribed {len(lines)} segments → {list_file}")

        # Copy best reference.wav
        if ref_wav:
            shutil.copy2(str(ref_wav), str(work_dir / "reference.wav"))

        return list_file

    # ── Subprocess runner ────────────────────────────────────────────────────

    def _run_script(
        self,
        script_rel: str,       # relative path inside GPT-SoVITS root, e.g. "GPT_SoVITS/prepare_datasets/1-get-text.py"
        extra_env: dict,
        cb: Optional[Callable],
        stage_name: str,
    ) -> None:
        env = {**os.environ, **extra_env}
        script_abs = str(self.gptsovits_dir / script_rel)
        cmd = [self.python, script_abs]

        logger.info(f"Running {stage_name} ...")
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.gptsovits_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                logger.debug(f"[{stage_name}] {line}")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"{stage_name} failed (exit code {proc.returncode})")

    # ── Step 3: BERT features (1-get-text.py) ───────────────────────────────

    def _extract_bert(self, list_file: Path, opt_dir: Path, exp_name: str, cb: Optional[Callable]) -> None:
        env = {
            "inp_text": str(list_file),
            "inp_wav_dir": str(list_file.parent / "sliced"),
            "exp_name": exp_name,
            "opt_dir": str(opt_dir),
            "bert_pretrained_dir": str(self.pretrained_dir / "chinese-roberta-wwm-ext-large"),
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": "0",
            "is_half": "True",
            "version": "v2",
        }
        self._run_script("GPT_SoVITS/prepare_datasets/1-get-text.py", env, cb, "BERT feature extraction")

        # Merge part files (only one part here)
        txt0 = opt_dir / "2-name2text-0.txt"
        txt = opt_dir / "2-name2text.txt"
        if txt0.exists() and not txt.exists():
            txt0.rename(txt)
        if not txt.exists():
            raise RuntimeError("2-name2text.txt not found after BERT extraction.")

    # ── Step 4: HuBERT + 32kHz WAV (2-get-hubert-wav32k.py) ────────────────

    def _extract_hubert(self, list_file: Path, opt_dir: Path, exp_name: str, cb: Optional[Callable]) -> None:
        env = {
            "inp_text": str(list_file),
            "inp_wav_dir": str(list_file.parent / "sliced"),
            "exp_name": exp_name,
            "opt_dir": str(opt_dir),
            "cnhubert_base_dir": str(self.pretrained_dir / "chinese-hubert-base"),
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": "0",
            "is_half": "True",
        }
        self._run_script("GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py", env, cb, "HuBERT feature extraction")

    # ── Step 5: Semantic tokens (3-get-semantic.py) ──────────────────────────

    def _extract_semantic(self, list_file: Path, opt_dir: Path, exp_name: str, cb: Optional[Callable]) -> None:
        s2config = self.gptsovits_dir / "GPT_SoVITS" / "configs" / "s2.json"
        env = {
            "inp_text": str(list_file),
            "exp_name": exp_name,
            "opt_dir": str(opt_dir),
            "pretrained_s2G": str(self.pretrained_dir / "pretrained_s2G.pth"),
            "s2config_path": str(s2config),
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": "0",
            "is_half": "True",
        }
        self._run_script("GPT_SoVITS/prepare_datasets/3-get-semantic.py", env, cb, "Semantic token extraction")

        # Merge part files
        sem0 = opt_dir / "6-name2semantic-0.tsv"
        sem = opt_dir / "6-name2semantic.tsv"
        if sem0.exists() and not sem.exists():
            sem0.rename(sem)
        if not sem.exists():
            raise RuntimeError("6-name2semantic.tsv not found after semantic extraction.")

    # ── Step 6: Train GPT / s1 ───────────────────────────────────────────────

    def _train_gpt(
        self,
        opt_dir: Path,
        voice_dir: Path,
        exp_name: str,
        epochs: int,
        cb: Optional[Callable],
    ) -> Path:
        """Run s1_train.py and return path to the best .ckpt file."""
        base_cfg_path = self.gptsovits_dir / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml"
        with open(base_cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        logs_dir = opt_dir / "logs_s1_v2"
        logs_dir.mkdir(parents=True, exist_ok=True)
        voice_dir.mkdir(parents=True, exist_ok=True)

        cfg["pretrained_s1"] = str(self.pretrained_dir / "pretrained_s1.ckpt")
        cfg["train"]["batch_size"] = 4
        cfg["train"]["epochs"] = epochs
        cfg["train"]["save_every_n_epoch"] = max(1, epochs // 5)
        cfg["train"]["if_save_latest"] = True
        cfg["train"]["if_save_every_weights"] = True
        cfg["train"]["if_dpo"] = False
        cfg["train"]["half_weights_save_dir"] = str(voice_dir)
        cfg["train"]["exp_name"] = exp_name
        cfg["train"]["precision"] = "16-mixed"
        cfg["train_semantic_path"] = str(opt_dir / "6-name2semantic.tsv")
        cfg["train_phoneme_path"] = str(opt_dir / "2-name2text.txt")
        cfg["output_dir"] = str(logs_dir)

        tmp_cfg = opt_dir / "tmp_s1.yaml"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

        env = {**os.environ, "_CUDA_VISIBLE_DEVICES": "0", "hz": "25hz"}
        cmd = [self.python, str(self.gptsovits_dir / "GPT_SoVITS" / "s1_train.py"),
               "--config_file", str(tmp_cfg)]

        logger.info("Training GPT model (s1)...")
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.gptsovits_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                logger.debug(f"[s1_train] {line}")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"s1_train.py failed (exit code {proc.returncode})")

        # Find the latest .ckpt saved to voice_dir
        ckpts = sorted(voice_dir.glob(f"{exp_name}-e*.ckpt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not ckpts:
            raise RuntimeError(f"No GPT checkpoint found in {voice_dir} after training.")
        return ckpts[0]

    # ── Step 7: Train SoVITS / s2 ────────────────────────────────────────────

    def _train_sovits(
        self,
        opt_dir: Path,
        voice_dir: Path,
        exp_name: str,
        epochs: int,
        cb: Optional[Callable],
    ) -> Path:
        """Run s2_train.py and return path to the best .pth file."""
        base_cfg_path = self.gptsovits_dir / "GPT_SoVITS" / "configs" / "s2.json"
        with open(base_cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)

        logs_s2 = opt_dir / "logs_s2_v2"
        logs_s2.mkdir(parents=True, exist_ok=True)
        voice_dir.mkdir(parents=True, exist_ok=True)

        cfg["train"]["batch_size"] = 8
        cfg["train"]["epochs"] = epochs
        cfg["train"]["pretrained_s2G"] = str(self.pretrained_dir / "pretrained_s2G.pth")
        cfg["train"]["pretrained_s2D"] = str(self.pretrained_dir / "pretrained_s2D.pth")
        cfg["train"]["if_save_latest"] = True
        cfg["train"]["if_save_every_weights"] = True
        cfg["train"]["save_every_epoch"] = max(1, epochs // 4)
        cfg["train"]["gpu_numbers"] = "0"
        cfg["train"]["text_low_lr_rate"] = 0.4
        cfg["train"]["grad_ckpt"] = False
        cfg["train"]["lora_rank"] = 128
        cfg["train"]["fp16_run"] = True
        cfg["data"]["exp_dir"] = str(opt_dir)
        cfg["model"]["version"] = "v2"
        cfg["s2_ckpt_dir"] = str(logs_s2)
        cfg["save_weight_dir"] = str(voice_dir)
        cfg["name"] = exp_name
        cfg["version"] = "v2"

        tmp_cfg = opt_dir / "tmp_s2.json"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        env = {**os.environ, "_CUDA_VISIBLE_DEVICES": "0"}
        cmd = [self.python, str(self.gptsovits_dir / "GPT_SoVITS" / "s2_train.py"),
               "--config", str(tmp_cfg)]

        logger.info("Training SoVITS model (s2)...")
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.gptsovits_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                logger.debug(f"[s2_train] {line}")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"s2_train.py failed (exit code {proc.returncode})")

        # Find the latest .pth saved to voice_dir
        pths = sorted(voice_dir.glob(f"{exp_name}_e*.pth"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not pths:
            raise RuntimeError(f"No SoVITS checkpoint found in {voice_dir} after training.")
        return pths[0]

    # ── Public entry point ────────────────────────────────────────────────────

    def train(
        self,
        voice_id: str,
        voice_name: str,
        audio_files: List[str],
        language: str = "zh",
        epochs_gpt: int = 15,
        epochs_sovits: int = 8,
        whisper_model_size: str = "medium",
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Run the full training pipeline.

        Args:
            voice_id:             UUID for this voice
            voice_name:           Display name
            audio_files:          List of source audio paths
            language:             'zh' or 'en'
            epochs_gpt:           GPT fine-tuning epochs
            epochs_sovits:        SoVITS fine-tuning epochs
            whisper_model_size:   'tiny'/'small'/'medium'/'large'
            progress_callback:    fn(stage, pct, msg)

        Returns:
            metadata dict (to be saved as metadata.json)
        """

        def cb(stage: str, pct: int, msg: str) -> None:
            logger.info(f"[{voice_id}] {pct}% — {msg}")
            if progress_callback:
                progress_callback(stage, pct, msg)

        self._check_ready()

        # Training workspace lives inside the GPT-SoVITS logs/ tree
        opt_dir = self.gptsovits_dir / "logs" / voice_id
        opt_dir.mkdir(parents=True, exist_ok=True)

        # Final model output directory
        voice_dir = self.output_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        speaker = voice_id[:8]  # short safe speaker tag

        # ── 1 Slice ────────────────────────────────────────────────────────
        cb("slice", 5, "Preprocessing / slicing audio...")
        segments = self._slice_audio(audio_files, opt_dir, speaker, cb)
        total_dur = sum(len(__import__("pydub").AudioSegment.from_file(str(s))) / 1000.0
                        for s in segments)

        # ── 2 Transcribe ──────────────────────────────────────────────────
        cb("transcribe", 12, "Transcribing with Whisper...")
        list_file = self._transcribe(segments, language, speaker, opt_dir, whisper_model_size)

        # ── 3 BERT features ───────────────────────────────────────────────
        cb("bert", 22, "Extracting BERT text features...")
        self._extract_bert(list_file, opt_dir, voice_id, cb)

        # ── 4 HuBERT features ─────────────────────────────────────────────
        cb("hubert", 33, "Extracting HuBERT audio features...")
        self._extract_hubert(list_file, opt_dir, voice_id, cb)

        # ── 5 Semantic tokens ─────────────────────────────────────────────
        cb("semantic", 43, "Extracting semantic tokens...")
        self._extract_semantic(list_file, opt_dir, voice_id, cb)

        # ── 6 Train GPT ───────────────────────────────────────────────────
        cb("gpt_train", 50, "Training GPT model...")
        gpt_raw = self._train_gpt(opt_dir, voice_dir, voice_id, epochs_gpt, cb)

        # ── 7 Train SoVITS ────────────────────────────────────────────────
        cb("sovits_train", 78, "Training SoVITS model...")
        sovits_raw = self._train_sovits(opt_dir, voice_dir, voice_id, epochs_sovits, cb)

        # ── 8 Rename to canonical names ───────────────────────────────────
        cb("finalize", 95, "Saving model files...")
        gpt_dest = voice_dir / f"{voice_id}_gpt.ckpt"
        sovits_dest = voice_dir / f"{voice_id}_sovits.pth"

        if gpt_raw != gpt_dest:
            shutil.copy2(str(gpt_raw), str(gpt_dest))
        if sovits_raw != sovits_dest:
            shutil.copy2(str(sovits_raw), str(sovits_dest))

        # Copy reference.wav if available
        ref_src = opt_dir / "reference.wav"
        if ref_src.exists():
            shutil.copy2(str(ref_src), str(voice_dir / "reference.wav"))

        # Build metadata
        from app.core.voice_manager import VoiceManager
        metadata = VoiceManager.build_metadata(
            voice_id=voice_id,
            voice_name=voice_name,
            language=language,
            audio_duration_seconds=round(total_dur, 1),
            training_epochs_gpt=epochs_gpt,
            training_epochs_sovits=epochs_sovits,
        )

        cb("done", 100, f"Training complete! voice_id: {voice_id}")
        logger.info(f"Training complete: {voice_dir}")
        return metadata
