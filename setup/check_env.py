#!/usr/bin/env python3
"""
Environment check script
Detects the Python version, CUDA, GPU, PyTorch, and all core dependencies,
then outputs a friendly report.
Run this script after installation and before starting the service to catch
compatibility issues early.
"""

import sys
import os
import platform
import importlib
import subprocess
from typing import Tuple, Optional

# ── Color output (Windows 10+ supports ANSI colors) ──────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Enable ANSI color support on Windows
if platform.system() == "Windows":
    os.system("")  # Activate ANSI escape codes


def ok(msg: str) -> str:
    return f"{GREEN}✓{RESET} {msg}"


def warn(msg: str) -> str:
    return f"{YELLOW}⚠{RESET} {msg}"


def err(msg: str) -> str:
    return f"{RED}✗{RESET} {msg}"


def info(msg: str) -> str:
    return f"{BLUE}ℹ{RESET} {msg}"


def section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 50}{RESET}")


def check_python() -> bool:
    """
    Check the Python version; requires 3.10.x.

    Returns:
        bool: True if the version meets the requirement, False otherwise
    """
    section("Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"  Current version: {version_str}")
    print(f"  Python path: {sys.executable}")

    if version.major == 3 and version.minor == 10:
        print(ok(f"Python {version_str} meets the requirement (needs 3.10.x)"))
        return True
    elif version.major == 3 and version.minor > 10:
        print(warn(f"Python {version_str} is newer than required (3.10); compatibility issues may occur"))
        return True
    else:
        print(err(f"Python {version_str} does not meet the requirement; needs 3.10.x"))
        return False


def check_pytorch() -> Tuple[bool, Optional[str]]:
    """
    Check the PyTorch installation; requires >= 2.7.

    Returns:
        Tuple[bool, Optional[str]]: (whether check passed, PyTorch version string)
    """
    section("PyTorch Version")
    try:
        import torch
        version = torch.__version__
        print(f"  PyTorch version: {version}")
        print(f"  Install path: {torch.__file__}")

        # Parse version number (supports nightly format like 2.7.0.dev20240301+cu121)
        version_clean = version.split("+")[0].replace(".dev", ".")
        parts = version_clean.split(".")
        major, minor = int(parts[0]), int(parts[1])

        if major > 2 or (major == 2 and minor >= 7):
            print(ok(f"PyTorch {version} meets the requirement (needs >= 2.7, supports RTX 5060 Blackwell sm_120)"))
            return True, version
        else:
            print(err(f"PyTorch {version} is too old; RTX 5060 (Blackwell/sm_120) requires PyTorch >= 2.7"))
            print(info("  Please run: pip install torch>=2.7 --index-url https://download.pytorch.org/whl/nightly/cu121"))
            return False, version
    except ImportError:
        print(err("PyTorch is not installed"))
        return False, None


def check_cuda() -> Tuple[bool, Optional[str]]:
    """
    Check CUDA availability and GPU information.

    Returns:
        Tuple[bool, Optional[str]]: (whether CUDA is available, GPU name)
    """
    section("CUDA & GPU Detection")

    try:
        import torch

        # Check whether CUDA is available
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {'Yes' if cuda_available else 'No'}")

        if not cuda_available:
            print(err("CUDA is not available; the service will run in CPU mode (very slow)"))
            print(info("  Possible reasons:"))
            print(info("    1. The PyTorch version does not support the current GPU (RTX 5060 requires >= 2.7)"))
            print(info("    2. NVIDIA drivers are not correctly installed"))
            print(info("    3. CUDA library paths are not configured in the current environment"))
            return False, None

        # Get the CUDA version PyTorch was compiled with
        cuda_version = torch.version.cuda
        print(f"  PyTorch CUDA version: {cuda_version}")

        # Get GPU information
        device_count = torch.cuda.device_count()
        print(f"  GPU count: {device_count}")

        gpu_name = None
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            cc_major = props.major
            cc_minor = props.minor
            cc = f"{cc_major}.{cc_minor}"
            total_mem_gb = props.total_memory / (1024 ** 3)
            gpu_name = props.name

            print(f"\n  GPU {i}: {props.name}")
            print(f"    Compute Capability: sm_{cc_major}{cc_minor} ({cc})")
            print(f"    VRAM: {total_mem_gb:.1f} GB")
            print(f"    Multiprocessor count: {props.multi_processor_count}")

            # Special note for RTX 5060 Blackwell
            if cc_major >= 12:
                print(ok(f"    Blackwell architecture (sm_{cc_major}{cc_minor}) detected - requires PyTorch >= 2.7"))
            elif cc_major == 8 or cc_major == 9:
                print(ok(f"    Ampere/Ada architecture (sm_{cc_major}{cc_minor}) fully supported"))
            else:
                print(info(f"    Architecture: sm_{cc_major}{cc_minor}"))

        print(ok("CUDA is available and GPU is recognized correctly"))
        return True, gpu_name

    except Exception as e:
        print(err(f"CUDA detection failed: {e}"))
        return False, None


def check_package(package_name: str, import_name: Optional[str] = None, min_version: Optional[str] = None) -> bool:
    """
    Check whether a single Python package is installed.

    Args:
        package_name: PyPI package name (for display)
        import_name: Module name used in import (may differ from package name)
        min_version: Minimum version requirement (string, optional)

    Returns:
        bool: True if the package exists and meets the version requirement
    """
    import_name = import_name or package_name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        print(ok(f"{package_name} {version}"))
        return True
    except ImportError:
        print(err(f"{package_name} is not installed"))
        return False


def check_core_packages() -> bool:
    """
    Check all core dependency packages.

    Returns:
        bool: True if all required packages are present
    """
    section("Core Dependency Check")

    required_packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("pydantic", "pydantic"),
        ("python-multipart", "multipart"),
        ("aiofiles", "aiofiles"),
        ("huggingface_hub", "huggingface_hub"),
        ("tqdm", "tqdm"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
    ]

    optional_packages = [
        ("transformers", "transformers"),
        ("omegaconf", "omegaconf"),
        ("einops", "einops"),
        ("diffusers", "diffusers"),
        ("ffmpeg-python", "ffmpeg"),
    ]

    print("\n  [Required packages]")
    all_ok = True
    for pkg_name, import_name in required_packages:
        result = check_package(pkg_name, import_name)
        if not result:
            all_ok = False

    print("\n  [Optional packages (CosyVoice3-related)]")
    for pkg_name, import_name in optional_packages:
        check_package(pkg_name, import_name)

    return all_ok


def check_ffmpeg_binary() -> bool:
    """
    Check whether the ffmpeg command-line tool is available
    (checked separately from the ffmpeg-python package).

    Returns:
        bool: True if ffmpeg is available
    """
    section("FFmpeg Command-Line Tool")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Extract version number (first line)
            first_line = result.stdout.split("\n")[0]
            print(ok(f"FFmpeg is available: {first_line}"))
            return True
        else:
            print(err("FFmpeg command execution failed"))
            return False
    except FileNotFoundError:
        print(err("ffmpeg command not found. Make sure ffmpeg is installed via conda or is in PATH."))
        return False
    except Exception as e:
        print(err(f"FFmpeg detection error: {e}"))
        return False


def check_model_files() -> None:
    """
    Check whether the pretrained model has been downloaded.
    """
    section("Pretrained Model")

    # Read model directory from environment variable or use default path
    model_dir = os.environ.get("MODEL_DIR", "storage/pretrained_models")
    cosyvoice_dir = os.path.join(model_dir, "Fun-CosyVoice3-0.5B-2512")

    print(f"  Model directory: {os.path.abspath(model_dir)}")

    if not os.path.exists(model_dir):
        print(warn(f"Model directory does not exist: {model_dir}"))
        print(info("  Please run: python setup/download_models.py"))
        return

    if os.path.exists(cosyvoice_dir):
        # Check key files
        key_files = ["config.yaml", "cosyvoice.yaml"]
        found = [f for f in key_files if os.path.exists(os.path.join(cosyvoice_dir, f))]
        if found:
            print(ok(f"Fun-CosyVoice3-0.5B-2512 model has been downloaded"))
            # Count files and total size
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(cosyvoice_dir):
                for file in files:
                    fp = os.path.join(root, file)
                    total_size += os.path.getsize(fp)
                    file_count += 1
            print(info(f"  File count: {file_count}, total size: {total_size / (1024**2):.1f} MB"))
        else:
            print(warn("Model directory exists but key files are missing; consider re-downloading"))
            print(info("  Please run: python setup/download_models.py"))
    else:
        print(warn("Fun-CosyVoice3-0.5B-2512 model has not been downloaded yet"))
        print(info("  Please run: python setup/download_models.py"))


def main() -> int:
    """
    Main function; runs all checks and returns an exit code.

    Returns:
        int: 0 if all checks pass, 1 if any check fails
    """
    print(f"\n{BOLD}{'═' * 52}{RESET}")
    print(f"{BOLD}  Voice Cloning Service - Environment Check Report{RESET}")
    print(f"{BOLD}{'═' * 52}{RESET}")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")

    results = []

    # Run all checks in order
    results.append(("Python Version", check_python()))
    pytorch_ok, _ = check_pytorch()
    results.append(("PyTorch", pytorch_ok))
    cuda_ok, _ = check_cuda()
    results.append(("CUDA/GPU", cuda_ok))
    results.append(("Core Dependencies", check_core_packages()))
    results.append(("FFmpeg", check_ffmpeg_binary()))

    # Model file check (not counted in pass/fail)
    check_model_files()

    # Summary report
    section("Check Summary")
    all_passed = True
    for name, passed in results:
        if passed:
            print(ok(f"{name}"))
        else:
            print(err(f"{name}"))
            all_passed = False

    print()
    if all_passed:
        print(f"{GREEN}{BOLD}✓ All checks passed! Environment is ready.{RESET}")
        print(info("  Next step: python setup/download_models.py (if model not yet downloaded)"))
        print(info("  Start service: start.bat"))
        return 0
    else:
        print(f"{RED}{BOLD}✗ Some checks failed. Please fix the issues indicated above and try again.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
