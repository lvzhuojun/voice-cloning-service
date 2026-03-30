#!/usr/bin/env python3
"""
GPT-SoVITS pretrained model download script.

Downloads all pretrained models required for training and inference:
  - chinese-hubert-base       (audio feature extractor)
  - chinese-roberta-wwm-ext-large  (BERT text encoder)
  - pretrained_s1.ckpt        (base GPT / AR model)
  - pretrained_s2G.pth        (base SoVITS generator)
  - pretrained_s2D.pth        (base SoVITS discriminator)

All models are saved to:  storage/pretrained_models/GPT-SoVITS/

Usage:
    conda run -n voice-cloning python setup/download_models.py
    python setup/download_models.py --model-dir storage/pretrained_models
    python setup/download_models.py --skip-transformers   # skip hubert/roberta if already present
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "storage/pretrained_models")
GPTSOVITS_SUBDIR  = "GPT-SoVITS"

# HuggingFace sources
GPTSOVITS_REPO      = "lj1995/GPT-SoVITS"
HUBERT_REPO         = "TencentGameMate/chinese-hubert-base"
ROBERTA_REPO        = "hfl/chinese-roberta-wwm-ext-large"

# Files to fetch from lj1995/GPT-SoVITS and their local target names
# (source path inside HF repo  →  local filename in pretrained_dir)
GPTSOVITS_FILES = [
    (
        "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "pretrained_s1.ckpt",
    ),
    (
        "gsv-v2final-pretrained/s2G2333k.pth",
        "pretrained_s2G.pth",
    ),
    (
        "gsv-v2final-pretrained/s2D2333k.pth",
        "pretrained_s2D.pth",
    ),
]

# ── ANSI colour helpers ────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

if sys.platform == "win32":
    os.system("")  # enable ANSI on Windows


def ok(msg):   return f"{GREEN}[OK]{RESET}   {msg}"
def warn(msg): return f"{YELLOW}[WARN]{RESET} {msg}"
def err(msg):  return f"{RED}[FAIL]{RESET} {msg}"
def info(msg): return f"{BLUE}[INFO]{RESET} {msg}"


# ── Dependency check ──────────────────────────────────────────────────────────

def check_huggingface_hub() -> bool:
    try:
        import huggingface_hub
        print(ok(f"huggingface_hub {huggingface_hub.__version__}"))
        return True
    except ImportError:
        print(err("huggingface_hub not installed. Run setup/install.bat first."))
        return False


# ── Individual file download ───────────────────────────────────────────────────

def download_file(repo_id: str, repo_filename: str, local_path: Path, force: bool) -> bool:
    """
    Download a single file from a HuggingFace repo and save it to local_path.
    Returns True on success.
    """
    from huggingface_hub import hf_hub_download

    if local_path.exists() and not force:
        print(info(f"  Already exists, skipping: {local_path.name}"))
        return True

    print(info(f"  Downloading {repo_filename} from {repo_id} ..."))
    try:
        tmp = hf_hub_download(
            repo_id=repo_id,
            filename=repo_filename,
            local_dir_use_symlinks=False,
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp, str(local_path))
        print(ok(f"  Saved: {local_path}"))
        return True
    except Exception as e:
        print(err(f"  Failed: {e}"))
        return False


# ── Transformer model directory download ─────────────────────────────────────

def download_transformer_model(repo_id: str, local_dir: Path, force: bool) -> bool:
    """
    Download a full HuggingFace transformer repo (e.g. hubert or roberta) to local_dir.
    Returns True on success.
    """
    from huggingface_hub import snapshot_download

    # Quick completeness check: look for config.json
    config_path = local_dir / "config.json"
    if config_path.exists() and not force:
        print(info(f"  Already exists, skipping: {local_dir.name}/"))
        return True

    print(info(f"  Downloading {repo_id} -> {local_dir} ..."))
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*", "*.ot"],
        )
        print(ok(f"  Saved: {local_dir}/"))
        return True
    except Exception as e:
        print(err(f"  Failed: {e}"))
        return False


# ── Main download orchestration ───────────────────────────────────────────────

def download_all(pretrained_dir: Path, force: bool, skip_transformers: bool) -> bool:
    """
    Download all required pretrained models to pretrained_dir.
    Returns True if all downloads succeeded.
    """
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    # ── 1. GPT / SoVITS checkpoint files ──────────────────────────────────────
    print(f"\n{BOLD}[1/3] GPT-SoVITS base model checkpoints{RESET}")
    print(f"      Source repo: {GPTSOVITS_REPO}")

    for repo_path, local_name in GPTSOVITS_FILES:
        local_path = pretrained_dir / local_name
        ok_flag = download_file(GPTSOVITS_REPO, repo_path, local_path, force)
        if not ok_flag:
            all_ok = False

    # ── 2. Chinese HuBERT base ─────────────────────────────────────────────────
    print(f"\n{BOLD}[2/3] Chinese HuBERT base (audio feature extractor){RESET}")
    if skip_transformers:
        print(info("  --skip-transformers set, skipping."))
    else:
        hubert_dir = pretrained_dir / "chinese-hubert-base"
        ok_flag = download_transformer_model(HUBERT_REPO, hubert_dir, force)
        if not ok_flag:
            all_ok = False

    # ── 3. Chinese RoBERTa wwm ext large ──────────────────────────────────────
    print(f"\n{BOLD}[3/3] Chinese RoBERTa-wwm-ext-large (BERT text encoder){RESET}")
    if skip_transformers:
        print(info("  --skip-transformers set, skipping."))
    else:
        roberta_dir = pretrained_dir / "chinese-roberta-wwm-ext-large"
        ok_flag = download_transformer_model(ROBERTA_REPO, roberta_dir, force)
        if not ok_flag:
            all_ok = False

    return all_ok


def verify(pretrained_dir: Path) -> bool:
    """
    Check that all required model files/dirs exist.
    Returns True if complete.
    """
    required = [
        pretrained_dir / "pretrained_s1.ckpt",
        pretrained_dir / "pretrained_s2G.pth",
        pretrained_dir / "pretrained_s2D.pth",
        pretrained_dir / "chinese-hubert-base" / "config.json",
        pretrained_dir / "chinese-roberta-wwm-ext-large" / "config.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print(warn("Missing files:"))
        for p in missing:
            print(f"  - {p}")
        return False
    return True


def print_stats(pretrained_dir: Path):
    total = sum(
        f.stat().st_size
        for f in pretrained_dir.rglob("*")
        if f.is_file()
    )
    count = sum(1 for f in pretrained_dir.rglob("*") if f.is_file())
    print(info(f"Total: {count} files, {total / 1024**3:.2f} GB"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download GPT-SoVITS pretrained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  conda run -n voice-cloning python setup/download_models.py
  python setup/download_models.py --model-dir /data/pretrained
  python setup/download_models.py --force
  python setup/download_models.py --skip-transformers
        """,
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Root directory for pretrained models (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--skip-transformers",
        action="store_true",
        help="Skip downloading chinese-hubert-base and chinese-roberta-wwm-ext-large",
    )
    return parser.parse_args()


def main() -> int:
    print("")
    print("=" * 60)
    print("  GPT-SoVITS Pretrained Model Download")
    print("=" * 60)
    print(f"  HF endpoint : {HF_ENDPOINT}")

    if not check_huggingface_hub():
        return 1

    args = parse_args()
    pretrained_dir = Path(args.model_dir) / GPTSOVITS_SUBDIR
    print(f"  Target dir  : {pretrained_dir.absolute()}")

    success = download_all(pretrained_dir, force=args.force, skip_transformers=args.skip_transformers)

    print(f"\n{BOLD}Verifying downloads...{RESET}")
    if verify(pretrained_dir) and not args.skip_transformers:
        print_stats(pretrained_dir)
        print("")
        print(f"{GREEN}{BOLD}[OK] All pretrained models downloaded successfully!{RESET}")
        print(info("Next step: start the service with start.bat"))
        return 0
    elif args.skip_transformers and success:
        print(ok("Checkpoint files downloaded. Remember to download transformer models separately."))
        return 0
    else:
        print("")
        print(f"{RED}{BOLD}[FAIL] Some downloads failed. Review the output above.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
