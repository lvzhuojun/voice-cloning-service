#!/usr/bin/env python3
"""
环境检测脚本
检测 Python 版本、CUDA、GPU、PyTorch 以及所有核心依赖，输出友好的报告。
在安装完成后、启动服务前运行此脚本，可提前发现兼容性问题。
"""

import sys
import os
import platform
import importlib
import subprocess
from typing import Tuple, Optional

# ── 颜色输出（Windows 10+ 支持 ANSI 颜色）────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

# 在 Windows 上启用 ANSI 颜色支持
if platform.system() == "Windows":
    os.system("")  # 激活 ANSI 转义


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
    检测 Python 版本，要求 3.10.x。

    Returns:
        bool: 版本符合要求返回 True，否则 False
    """
    section("Python 版本")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"  当前版本: {version_str}")
    print(f"  Python 路径: {sys.executable}")

    if version.major == 3 and version.minor == 10:
        print(ok(f"Python {version_str} 符合要求（需要 3.10.x）"))
        return True
    elif version.major == 3 and version.minor > 10:
        print(warn(f"Python {version_str} 高于要求版本（3.10），可能存在兼容性问题"))
        return True
    else:
        print(err(f"Python {version_str} 不符合要求，需要 3.10.x"))
        return False


def check_pytorch() -> Tuple[bool, Optional[str]]:
    """
    检测 PyTorch 安装情况，要求 >= 2.7。

    Returns:
        Tuple[bool, Optional[str]]: (是否通过, PyTorch版本字符串)
    """
    section("PyTorch 版本")
    try:
        import torch
        version = torch.__version__
        print(f"  PyTorch 版本: {version}")
        print(f"  安装路径: {torch.__file__}")

        # 解析版本号（支持 nightly 格式如 2.7.0.dev20240301+cu121）
        version_clean = version.split("+")[0].replace(".dev", ".")
        parts = version_clean.split(".")
        major, minor = int(parts[0]), int(parts[1])

        if major > 2 or (major == 2 and minor >= 7):
            print(ok(f"PyTorch {version} 符合要求（需要 >= 2.7，支持 RTX 5060 Blackwell sm_120）"))
            return True, version
        else:
            print(err(f"PyTorch {version} 版本过低，RTX 5060 (Blackwell/sm_120) 需要 PyTorch >= 2.7"))
            print(info("  请运行: pip install torch>=2.7 --index-url https://download.pytorch.org/whl/nightly/cu121"))
            return False, version
    except ImportError:
        print(err("PyTorch 未安装"))
        return False, None


def check_cuda() -> Tuple[bool, Optional[str]]:
    """
    检测 CUDA 可用性和 GPU 信息。

    Returns:
        Tuple[bool, Optional[str]]: (CUDA是否可用, GPU名称)
    """
    section("CUDA & GPU 检测")

    try:
        import torch

        # 检测 CUDA 是否可用
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA 可用: {'是' if cuda_available else '否'}")

        if not cuda_available:
            print(err("CUDA 不可用，服务将以 CPU 模式运行（性能极低）"))
            print(info("  可能原因："))
            print(info("    1. PyTorch 版本不支持当前 GPU（RTX 5060 需要 >= 2.7）"))
            print(info("    2. NVIDIA 驱动未正确安装"))
            print(info("    3. 当前环境中 CUDA 库路径未配置"))
            return False, None

        # 获取 PyTorch 编译时 CUDA 版本
        cuda_version = torch.version.cuda
        print(f"  PyTorch CUDA 版本: {cuda_version}")

        # 获取 GPU 信息
        device_count = torch.cuda.device_count()
        print(f"  GPU 数量: {device_count}")

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
            print(f"    显存: {total_mem_gb:.1f} GB")
            print(f"    多处理器数量: {props.multi_processor_count}")

            # 针对 RTX 5060 Blackwell 的特殊提示
            if cc_major >= 12:
                print(ok(f"    Blackwell 架构 (sm_{cc_major}{cc_minor}) 检测成功 - 需要 PyTorch >= 2.7"))
            elif cc_major == 8 or cc_major == 9:
                print(ok(f"    Ampere/Ada 架构 (sm_{cc_major}{cc_minor}) 正常支持"))
            else:
                print(info(f"    架构: sm_{cc_major}{cc_minor}"))

        print(ok("CUDA 可用，GPU 已正确识别"))
        return True, gpu_name

    except Exception as e:
        print(err(f"CUDA 检测失败: {e}"))
        return False, None


def check_package(package_name: str, import_name: Optional[str] = None, min_version: Optional[str] = None) -> bool:
    """
    检测单个 Python 包是否已安装。

    Args:
        package_name: PyPI 包名（用于显示）
        import_name: import 时的模块名（可能与包名不同）
        min_version: 最低版本要求（字符串，可选）

    Returns:
        bool: 包存在且版本符合要求返回 True
    """
    import_name = import_name or package_name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        print(ok(f"{package_name} {version}"))
        return True
    except ImportError:
        print(err(f"{package_name} 未安装"))
        return False


def check_core_packages() -> bool:
    """
    检测所有核心依赖包。

    Returns:
        bool: 所有必须包都存在返回 True
    """
    section("核心依赖包检测")

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

    print("\n  【必须包】")
    all_ok = True
    for pkg_name, import_name in required_packages:
        result = check_package(pkg_name, import_name)
        if not result:
            all_ok = False

    print("\n  【可选包（CosyVoice3 相关）】")
    for pkg_name, import_name in optional_packages:
        check_package(pkg_name, import_name)

    return all_ok


def check_ffmpeg_binary() -> bool:
    """
    检测 ffmpeg 命令行工具是否可用（与 ffmpeg-python 包分开检测）。

    Returns:
        bool: ffmpeg 可用返回 True
    """
    section("FFmpeg 命令行工具")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # 提取版本号（第一行）
            first_line = result.stdout.split("\n")[0]
            print(ok(f"FFmpeg 可用: {first_line}"))
            return True
        else:
            print(err("FFmpeg 命令执行失败"))
            return False
    except FileNotFoundError:
        print(err("ffmpeg 命令未找到，请确保 ffmpeg 已通过 conda 安装或在 PATH 中"))
        return False
    except Exception as e:
        print(err(f"FFmpeg 检测异常: {e}"))
        return False


def check_model_files() -> None:
    """
    检测预训练模型是否已下载。
    """
    section("预训练模型")

    # 从环境变量或默认路径读取模型目录
    model_dir = os.environ.get("MODEL_DIR", "storage/pretrained_models")
    cosyvoice_dir = os.path.join(model_dir, "Fun-CosyVoice3-0.5B-2512")

    print(f"  模型目录: {os.path.abspath(model_dir)}")

    if not os.path.exists(model_dir):
        print(warn(f"模型目录不存在: {model_dir}"))
        print(info("  请先运行: python setup/download_models.py"))
        return

    if os.path.exists(cosyvoice_dir):
        # 检查关键文件
        key_files = ["config.yaml", "cosyvoice.yaml"]
        found = [f for f in key_files if os.path.exists(os.path.join(cosyvoice_dir, f))]
        if found:
            print(ok(f"Fun-CosyVoice3-0.5B-2512 模型已下载"))
            # 统计文件数量和总大小
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(cosyvoice_dir):
                for file in files:
                    fp = os.path.join(root, file)
                    total_size += os.path.getsize(fp)
                    file_count += 1
            print(info(f"  文件数: {file_count}，总大小: {total_size / (1024**2):.1f} MB"))
        else:
            print(warn("模型目录存在但关键文件缺失，建议重新下载"))
            print(info("  请运行: python setup/download_models.py"))
    else:
        print(warn("Fun-CosyVoice3-0.5B-2512 模型尚未下载"))
        print(info("  请运行: python setup/download_models.py"))


def main() -> int:
    """
    主函数，运行所有检测项目，返回退出码。

    Returns:
        int: 0 表示全部通过，1 表示有错误
    """
    print(f"\n{BOLD}{'═' * 52}{RESET}")
    print(f"{BOLD}  语音克隆服务 - 环境检测报告{RESET}")
    print(f"{BOLD}{'═' * 52}{RESET}")
    print(f"  平台: {platform.system()} {platform.release()}")
    print(f"  架构: {platform.machine()}")

    results = []

    # 按顺序执行所有检测
    results.append(("Python 版本", check_python()))
    pytorch_ok, _ = check_pytorch()
    results.append(("PyTorch", pytorch_ok))
    cuda_ok, _ = check_cuda()
    results.append(("CUDA/GPU", cuda_ok))
    results.append(("核心依赖", check_core_packages()))
    results.append(("FFmpeg", check_ffmpeg_binary()))

    # 模型文件检测（不计入通过/失败）
    check_model_files()

    # 汇总报告
    section("检测汇总")
    all_passed = True
    for name, passed in results:
        if passed:
            print(ok(f"{name}"))
        else:
            print(err(f"{name}"))
            all_passed = False

    print()
    if all_passed:
        print(f"{GREEN}{BOLD}✓ 所有检测项目通过！环境已就绪。{RESET}")
        print(info("  下一步: python setup/download_models.py（如尚未下载模型）"))
        print(info("  启动服务: start.bat"))
        return 0
    else:
        print(f"{RED}{BOLD}✗ 部分检测失败，请根据上方提示修复后重试。{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
