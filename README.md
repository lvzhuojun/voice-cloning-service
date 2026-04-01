# Voice Cloning Service

**[English](README.md) | [中文](README_zh.md)**

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![Framework](https://img.shields.io/badge/framework-GPT--SoVITS%20v2-orange)

A RESTful voice cloning backend based on **GPT-SoVITS v2** real fine-tuning.

Upload 1–3 minutes of reference audio → fine-tune a personal voice model on the server → export portable dual-file models (`.ckpt` + `.pth`) for downstream systems to manage, distribute, and run inference with.

---

## Features

- **Real fine-tuning** — full GPT-SoVITS v2 training pipeline, not zero-shot cloning
- **8-step pipeline** — audio slicing → Whisper transcription → BERT/HuBERT feature extraction → GPT (s1) training → SoVITS (s2) training → model export
- **Portable model output** — each trained voice produces a self-contained directory (`{voice_id}_gpt.ckpt` + `{voice_id}_sovits.pth` + `metadata.json` + `reference.wav`)
- **Full REST API** — training, synthesis preview, model download, voice management
- **Web management UI** — built-in HTML UI at `/`
- **Windows-native** — tested on Windows 11 + RTX GPU + Anaconda

---

## Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10 / 11 |
| Conda | Miniconda or Anaconda |
| Python | 3.10 (managed via conda) |
| GPU | NVIDIA GPU with CUDA support |
| VRAM | ≥ 6 GB recommended |
| Disk | ~10 GB free (pretrained models + training workspace) |

> **GPU note:** The default `setup/install.bat` installs PyTorch 2.7 + CUDA 12.8 (for RTX 40/50 series). Edit the install script to change the CUDA version if needed.

---

## Installation

### Step 1 — Clone this repository

```bash
git clone https://github.com/lvzhuojun/voice-cloning-service.git
cd voice-cloning-service
```

### Step 2 — Clone GPT-SoVITS

```bat
setup\clone_gptsovits.bat
```

This clones the [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) source into the `GPT-SoVITS/` directory (excluded from version control).

### Step 3 — Install dependencies

```bat
setup\install.bat
```

This will:
1. Create the `voice-cloning` conda environment (Python 3.10)
2. Install PyTorch 2.7 + CUDA 12.8
3. Install all pip dependencies from `requirements-pip.txt`
4. Run an environment verification check

### Step 4 — Download pretrained models

```bat
conda run -n voice-cloning python setup/download_models.py
```

Downloads to `storage/pretrained_models/GPT-SoVITS/` (~3–4 GB total):

| File | Description |
|------|-------------|
| `pretrained_s1.ckpt` | Base GPT (AR) model |
| `pretrained_s2G.pth` | Base SoVITS generator |
| `pretrained_s2D.pth` | Base SoVITS discriminator |
| `chinese-hubert-base/` | Audio feature extractor |
| `chinese-roberta-wwm-ext-large/` | BERT text encoder |

> **China mirror:** The downloader uses `https://hf-mirror.com` by default. Set `HF_ENDPOINT` in `.env` to override.

---

## Quick Start

### 1. Start the service

```bat
start.bat
```

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | Web management UI |
| `http://localhost:8000/docs` | Swagger API docs |
| `http://localhost:8000/api/health` | Health check |

### 2. Submit a training job

**From a local folder** (place audio files in `data/samples/<folder_name>/`):

```http
POST /api/train/from-folder
Content-Type: application/json

{
  "folder_name": "my_speaker",
  "voice_name": "My Voice",
  "language": "zh"
}
```

**From uploaded files:**

```http
POST /api/train/from-upload
Content-Type: multipart/form-data

voice_name=My Voice
language=zh
files=<audio files>
```

Supported formats: `WAV / MP3 / M4A / FLAC / OGG`, up to 10 files, max 200 MB each.

### 3. Poll training status

```http
GET /api/train/{task_id}/status
```

Response includes `progress` (0–100) and `voice_id` when `status` is `done`.

### 4. Synthesize a test preview

```http
POST /api/voices/{voice_id}/test
Content-Type: application/json

{
  "text": "Hello, this is a voice cloning test.",
  "speed": 1.0,
  "language": "zh"
}
```

Returns a WAV audio stream.

### 5. Download model files

```http
GET /api/voices/{voice_id}/download/gpt      # GPT checkpoint (.ckpt)
GET /api/voices/{voice_id}/download/sovits   # SoVITS weights (.pth)
GET /api/voices/{voice_id}/download/all      # ZIP archive (all files)
```

---

## Output Format

Each successful training run produces an isolated voice directory:

```
storage/models/{voice_id}/
├── {voice_id}_gpt.ckpt      # GPT text-to-semantic model weights
├── {voice_id}_sovits.pth    # SoVITS vocoder generator weights
├── metadata.json            # Voice metadata (name, language, training params)
└── reference.wav            # Reference audio used for synthesis
```

**`metadata.json` example:**

```json
{
  "voice_id": "1511e200-f24d-4346-8af9-29d253d0dde5",
  "voice_name": "My Voice",
  "language": "zh",
  "created_at": "2026-03-30T15:24:37+00:00",
  "training_epochs_gpt": 15,
  "training_epochs_sovits": 8,
  "gpt_model_file": "1511e200-f24d-4346-8af9-29d253d0dde5_gpt.ckpt",
  "sovits_model_file": "1511e200-f24d-4346-8af9-29d253d0dde5_sovits.pth",
  "base_model_version": "GPT-SoVITS v2"
}
```

### Using exported models in downstream systems

Any system that runs GPT-SoVITS inference can load these files directly:

```python
from TTS_infer_pack.TTS import TTS, TTS_Config

voice_id = "1511e200-f24d-4346-8af9-29d253d0dde5"
config = TTS_Config({
    "custom": {
        "device": "cuda",
        "is_half": True,
        "version": "v2",
        "t2s_weights_path": f"storage/models/{voice_id}/{voice_id}_gpt.ckpt",
        "vits_weights_path": f"storage/models/{voice_id}/{voice_id}_sovits.pth",
        "cnhuhbert_base_path": "storage/pretrained_models/GPT-SoVITS/chinese-hubert-base",
        "bert_base_path": "storage/pretrained_models/GPT-SoVITS/chinese-roberta-wwm-ext-large",
    }
})
tts = TTS(config)
```

---

## API Reference

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Service status, GPU info, voice count, uptime |

### Training

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/train/from-upload` | Submit training job from uploaded files |
| POST | `/api/train/from-folder` | Submit training job from local folder |
| GET | `/api/train/{task_id}/status` | Poll training progress |

### Voice Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/voices/` | List all trained voices |
| GET | `/api/voices/{voice_id}` | Get voice details |
| DELETE | `/api/voices/{voice_id}` | Delete voice and all files |
| POST | `/api/voices/{voice_id}/test` | Synthesize test speech (WAV) |
| GET | `/api/voices/{voice_id}/download/gpt` | Download GPT checkpoint |
| GET | `/api/voices/{voice_id}/download/sovits` | Download SoVITS weights |
| GET | `/api/voices/{voice_id}/download/all` | Download ZIP archive |

Full interactive docs available at `/docs` (Swagger) and `/redoc`.

---

## Configuration

Copy `.env.example` to `.env` and edit as needed:

```env
HOST=0.0.0.0
PORT=8000

# HuggingFace mirror (recommended for China)
HF_ENDPOINT=https://hf-mirror.com

# Training parameters
TRAINING_EPOCHS_GPT=15
TRAINING_EPOCHS_SOVITS=8
WHISPER_MODEL=medium

# Upload limits
MAX_UPLOAD_SIZE_MB=200
```

---

## Project Structure

```
voice-cloning-service/
├── app/
│   ├── api/            # FastAPI route handlers (health, train, voices)
│   ├── core/           # Business logic (trainer, tts_engine, voice_manager)
│   ├── models/         # Pydantic schemas
│   └── utils/          # Logging, file utilities
├── setup/
│   ├── install.bat     # One-click install (Windows)
│   ├── clone_gptsovits.bat / .sh
│   ├── download_models.py
│   └── check_env.py
├── static/             # Web management UI
├── data/samples/       # Local training audio (gitignored)
├── storage/            # Runtime data (gitignored)
│   ├── models/         # Trained voice models
│   ├── pretrained_models/
│   └── uploads/
├── tests/
├── GPT-SoVITS/         # GPT-SoVITS source (cloned separately, gitignored)
├── start.bat
├── stop.bat
└── .env.example
```

---

## Notes

- **Privacy / copyright:** this repository intentionally does **not** track user training audio, trained voice models, pretrained weights, or the separately cloned `GPT-SoVITS/` source tree. Keep those files under `data/`, `storage/`, and `GPT-SoVITS/` only.
- **Chinese is the recommended language.** Training and synthesis for Chinese (`zh`) are fully tested.
- **English synthesis** requires additional packages (`wordsegment`, `g2p_en`, NLTK data). These are listed in `requirements-pip.txt` but may need manual verification on first run.
- **Training task state** is stored in memory. Restarting the service clears all task history (voice models on disk are unaffected).
- **Reference audio** (`reference.wav`) is kept alongside the model files and is required for the built-in synthesis endpoint.

---

## License

This project is released under the [MIT License](LICENSE).

GPT-SoVITS is a separate project with its own license — see `GPT-SoVITS/LICENSE`.
