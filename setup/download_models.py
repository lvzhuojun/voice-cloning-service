#!/usr/bin/env python3
"""
Model download script
Downloads the Fun-CosyVoice3-0.5B-2512 pretrained model from HuggingFace (or a mirror).
Supports configuring a mirror via the HF_ENDPOINT environment variable;
hf-mirror.com is recommended for users in China.

Usage:
    python setup/download_models.py
    python setup/download_models.py --model-dir storage/pretrained_models
    HF_ENDPOINT=https://hf-mirror.com python setup/download_models.py
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Set the environment variable before importing huggingface_hub so hf_hub uses the mirror.
# Read from the environment variable; default to hf-mirror.com (China mirror).
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

# ── Color output (ANSI escape codes are ASCII-safe) ───────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

if sys.platform == "win32":
    os.system("")  # Enable Windows ANSI colors


def ok(msg: str) -> str:
    return f"{GREEN}[OK]{RESET} {msg}"


def warn(msg: str) -> str:
    return f"{YELLOW}[WARN]{RESET} {msg}"


def err(msg: str) -> str:
    return f"{RED}[FAIL]{RESET} {msg}"


def info(msg: str) -> str:
    return f"{BLUE}[INFO]{RESET} {msg}"


# ── Model configuration ───────────────────────────────────────────────────────
MODEL_REPO_ID = "FunAudioLLM/CosyVoice2-0.5B"  # HuggingFace repository for CosyVoice3
MODEL_DIR_NAME = "Fun-CosyVoice3-0.5B-2512"      # Local storage directory name

# Key files required for integrity verification
REQUIRED_FILES = [
    "cosyvoice.yaml",
    "campplus.onnx",
]


def check_huggingface_hub() -> bool:
    """
    Check whether huggingface_hub is installed.

    Returns:
        bool: True if installed
    """
    try:
        import huggingface_hub
        print(ok(f"huggingface_hub {huggingface_hub.__version__} is installed"))
        return True
    except ImportError:
        print(err("huggingface_hub is not installed. Please run setup/install.bat first."))
        print(info("Or install manually: pip install huggingface_hub"))
        return False


def download_model(model_dir: str, force: bool = False) -> bool:
    """
    Download the Fun-CosyVoice3-0.5B-2512 model to the specified directory.

    Args:
        model_dir: Model storage root directory (relative or absolute path)
        force: Whether to force re-download even if the model already exists

    Returns:
        bool: True if download succeeded
    """
    from huggingface_hub import snapshot_download

    target_dir = Path(model_dir) / MODEL_DIR_NAME
    target_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print(f"{BOLD}Model download configuration{RESET}")
    print(f"  Repository ID : {MODEL_REPO_ID}")
    print(f"  Target dir    : {target_dir.absolute()}")
    print(f"  HF endpoint   : {HF_ENDPOINT}")

    # Check if already downloaded
    if not force and _is_model_complete(str(target_dir)):
        print(ok("Model already fully downloaded. Skipping (use --force to re-download)."))
        return True

    print("")
    print(f"{BOLD}Starting model download...{RESET}")
    print(info("Download is approximately 2 GB. Please be patient."))
    print(info("Users in China should use hf-mirror.com (already set as default)."))

    try:
        # Use snapshot_download to download the entire repository.
        # huggingface_hub automatically reads the HF_ENDPOINT environment variable.
        local_dir = snapshot_download(
            repo_id=MODEL_REPO_ID,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,  # Windows does not support symlinks; use real files
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(ok(f"Model download complete: {local_dir}"))

    except Exception as e:
        print(err(f"Download failed: {e}"))
        print(info("Trying fallback: downloading files individually..."))

        # Fallback: download files one by one
        try:
            _download_files_individually(str(target_dir))
        except Exception as e2:
            print(err(f"Fallback download also failed: {e2}"))
            print(info("Please check:"))
            print(info("  1. Whether your network connection is working"))
            print(info("  2. Whether HF_ENDPOINT is accessible"))
            print(info(f"  3. Try opening manually: {HF_ENDPOINT}/{MODEL_REPO_ID}"))
            return False

    # Verify integrity
    print("")
    print(f"{BOLD}Verifying file integrity...{RESET}")
    if _is_model_complete(str(target_dir)):
        print(ok("File integrity check passed"))
        _print_model_stats(str(target_dir))
        return True
    else:
        print(warn("Some key files are missing; the download may be incomplete"))
        return False


def _download_files_individually(target_dir: str) -> None:
    """
    Download files one by one (fallback when snapshot_download fails).

    Args:
        target_dir: Local storage directory path
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    print(info("Listing repository files..."))
    try:
        files = list(list_repo_files(MODEL_REPO_ID))
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve file list: {e}")

    # Filter out unwanted files
    skip_patterns = [".msgpack", "flax_model", "tf_model", "rust_model"]
    files = [f for f in files if not any(p in f for p in skip_patterns)]

    print(f"  {len(files)} files to download")

    for i, filename in enumerate(files, 1):
        local_path = Path(target_dir) / filename
        if local_path.exists():
            print(info(f"  [{i}/{len(files)}] Already exists: {filename}"))
            continue

        print(f"  [{i}/{len(files)}] Downloading: {filename}")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )


def _is_model_complete(target_dir: str) -> bool:
    """
    Check whether the model's key files exist.

    Args:
        target_dir: Model directory path

    Returns:
        bool: True if all key files exist
    """
    dir_path = Path(target_dir)
    if not dir_path.exists():
        return False

    for required_file in REQUIRED_FILES:
        if not (dir_path / required_file).exists():
            return False

    return True


def _print_model_stats(model_dir: str) -> None:
    """
    Print file statistics for the model directory.

    Args:
        model_dir: Model directory path
    """
    total_size = 0
    file_count = 0
    dir_path = Path(model_dir)

    for filepath in dir_path.rglob("*"):
        if filepath.is_file():
            total_size += filepath.stat().st_size
            file_count += 1

    size_mb = total_size / (1024 ** 2)
    print(info(f"Model stats: {file_count} files, {size_mb:.1f} MB total"))


def save_download_info(model_dir: str) -> None:
    """
    Save download information to download_info.json for version tracking.

    Args:
        model_dir: Model root directory
    """
    import datetime

    info_path = Path(model_dir) / MODEL_DIR_NAME / "download_info.json"
    download_info = {
        "repo_id": MODEL_REPO_ID,
        "downloaded_at": datetime.datetime.now().isoformat(),
        "hf_endpoint": HF_ENDPOINT,
        "local_dir_name": MODEL_DIR_NAME,
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(download_info, f, ensure_ascii=False, indent=2)

    print(info(f"Download info saved: {info_path}"))


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed argument object
    """
    parser = argparse.ArgumentParser(
        description="Download the Fun-CosyVoice3-0.5B-2512 pretrained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup/download_models.py
  python setup/download_models.py --model-dir /data/models
  python setup/download_models.py --force
  HF_ENDPOINT=https://hf-mirror.com python setup/download_models.py

Mirror notes:
  For users in China, hf-mirror.com is recommended (already the default):
  set HF_ENDPOINT=https://hf-mirror.com
        """
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "storage/pretrained_models"),
        help="Model storage directory (default: storage/pretrained_models or MODEL_DIR env var)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the model already exists"
    )
    return parser.parse_args()


def main() -> int:
    """
    Main function; executes the model download workflow.

    Returns:
        int: 0 for success, 1 for failure
    """
    print("")
    print("=" * 54)
    print("  CosyVoice3 Model Download Script")
    print("=" * 54)
    print(f"  HF endpoint: {HF_ENDPOINT}")
    print(info("To change the mirror, set the HF_ENDPOINT environment variable."))

    args = parse_args()

    # Check dependencies
    if not check_huggingface_hub():
        return 1

    # Execute download
    success = download_model(args.model_dir, force=args.force)

    if success:
        save_download_info(args.model_dir)
        print("")
        print(f"{GREEN}{BOLD}[OK] Model download complete!{RESET}")
        print(info("Next step: copy .env.example to .env, then run start.bat to start the service."))
        return 0
    else:
        print("")
        print(f"{RED}{BOLD}[FAIL] Model download failed. Please review the error messages above.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
