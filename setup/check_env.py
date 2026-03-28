#!/usr/bin/env python3
"""
Environment check script
Detects Python version, CUDA, GPU, PyTorch, and all core dependencies.
Run after installation and before starting the service.
"""

import sys
import os
import platform
import importlib
import subprocess
from typing import Tuple, Optional


def section(title):
    print("")
    print("--------------------------------------------------")
    print("  " + title)
    print("--------------------------------------------------")


def check_python():
    section("Python Version")
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    print(f"  Version : {ver}")
    print(f"  Path    : {sys.executable}")
    if v.major == 3 and v.minor == 10:
        print(f"  [OK] Python {ver} meets requirement (3.10.x)")
        return True
    elif v.major == 3 and v.minor > 10:
        print(f"  [WARN] Python {ver} is newer than required (3.10); may have issues")
        return True
    else:
        print(f"  [FAIL] Python {ver} does not meet requirement; needs 3.10.x")
        return False


def check_pytorch():
    section("PyTorch Version")
    try:
        import torch
        version = torch.__version__
        print(f"  PyTorch : {version}")
        print(f"  Path    : {torch.__file__}")

        version_clean = version.split("+")[0].replace(".dev", ".")
        parts = version_clean.split(".")
        major, minor = int(parts[0]), int(parts[1])

        if major > 2 or (major == 2 and minor >= 7):
            print(f"  [OK] PyTorch {version} supports RTX 5060 Blackwell (sm_120)")
            return True, version
        else:
            print(f"  [FAIL] PyTorch {version} too old; RTX 5060 requires >= 2.7")
            return False, version
    except ImportError:
        print("  [FAIL] PyTorch is not installed")
        return False, None


def check_cuda():
    section("CUDA & GPU")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available : {'Yes' if cuda_available else 'No'}")

        if not cuda_available:
            print("  [FAIL] CUDA not available; will run on CPU (very slow)")
            print("  Possible causes:")
            print("    1. PyTorch version does not support this GPU (RTX 5060 needs >= 2.7)")
            print("    2. NVIDIA drivers not installed correctly")
            return False, None

        print(f"  PyTorch CUDA   : {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"  GPU count      : {device_count}")

        gpu_name = None
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            cc = f"{props.major}.{props.minor}"
            mem_gb = props.total_memory / (1024 ** 3)
            gpu_name = props.name
            print(f"  GPU {i}          : {props.name}")
            print(f"    Compute Cap  : sm_{props.major}{props.minor} ({cc})")
            print(f"    VRAM         : {mem_gb:.1f} GB")
            if props.major >= 12:
                print(f"    [OK] Blackwell (sm_{props.major}{props.minor}) detected, PyTorch 2.7+ required")
            elif props.major >= 8:
                print(f"    [OK] Ampere/Ada (sm_{props.major}{props.minor}) fully supported")

        print("  [OK] CUDA available and GPU recognized")
        return True, gpu_name
    except Exception as e:
        print(f"  [FAIL] CUDA detection error: {e}")
        return False, None


def check_package(pkg_name, import_name=None):
    import_name = import_name or pkg_name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        print(f"  [OK] {pkg_name} {version}")
        return True
    except ImportError:
        print(f"  [FAIL] {pkg_name} not installed")
        return False


def check_core_packages():
    section("Core Dependencies")

    required = [
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
    optional = [
        ("transformers", "transformers"),
        ("omegaconf", "omegaconf"),
        ("einops", "einops"),
        ("diffusers", "diffusers"),
    ]

    print("  [Required]")
    all_ok = True
    for pkg, imp in required:
        if not check_package(pkg, imp):
            all_ok = False

    print("  [Optional - CosyVoice3]")
    for pkg, imp in optional:
        check_package(pkg, imp)

    return all_ok


def check_ffmpeg():
    section("FFmpeg Binary")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            first_line = result.stdout.split("\n")[0]
            print(f"  [OK] {first_line}")
            return True
        else:
            print("  [FAIL] ffmpeg command failed")
            return False
    except FileNotFoundError:
        print("  [FAIL] ffmpeg not found in PATH")
        return False
    except Exception as e:
        print(f"  [FAIL] ffmpeg check error: {e}")
        return False


def check_model_files():
    section("Pretrained Model")
    model_dir = os.environ.get("MODEL_DIR", "storage/pretrained_models")
    cosyvoice_dir = os.path.join(model_dir, "Fun-CosyVoice3-0.5B-2512")
    print(f"  Model dir : {os.path.abspath(model_dir)}")

    if not os.path.exists(model_dir):
        print("  [WARN] Model directory does not exist")
        print("  Run: python setup/download_models.py")
        return

    if os.path.exists(cosyvoice_dir):
        key_files = ["cosyvoice2.yaml", "campplus.onnx"]
        found = [f for f in key_files if os.path.exists(os.path.join(cosyvoice_dir, f))]
        if found:
            total_size = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _, files in os.walk(cosyvoice_dir)
                for f in files
            )
            print(f"  [OK] Fun-CosyVoice3-0.5B-2512 found ({total_size / 1024**2:.0f} MB)")
        else:
            print("  [WARN] Model directory exists but key files missing; re-download recommended")
    else:
        print("  [WARN] Fun-CosyVoice3-0.5B-2512 not downloaded yet")
        print("  Run: python setup/download_models.py")


def main():
    print("")
    print("==================================================")
    print("  Voice Cloning Service - Environment Check")
    print("==================================================")
    print(f"  Platform : {platform.system()} {platform.release()}")

    results = []
    results.append(("Python",        check_python()))
    ok, _  = check_pytorch();  results.append(("PyTorch", ok))
    ok, _  = check_cuda();     results.append(("CUDA/GPU", ok))
    results.append(("Core packages", check_core_packages()))
    results.append(("FFmpeg",        check_ffmpeg()))
    check_model_files()

    section("Summary")
    all_passed = True
    for name, passed in results:
        status = "[OK]  " if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("")
    if all_passed:
        print("  All checks passed. Environment is ready.")
        return 0
    else:
        print("  Some checks failed. Fix the issues above and retry.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
