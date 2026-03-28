#!/usr/bin/env python3
"""
模型下载脚本
从 HuggingFace（或镜像源）下载 Fun-CosyVoice3-0.5B-2512 预训练模型。
支持通过 HF_ENDPOINT 环境变量配置镜像源，国内环境推荐使用 hf-mirror.com。

用法:
    python setup/download_models.py
    python setup/download_models.py --model-dir storage/pretrained_models
    HF_ENDPOINT=https://hf-mirror.com python setup/download_models.py
"""

import os
import sys
import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

# 在 import huggingface_hub 之前设置环境变量，确保 hf_hub 使用镜像
# 从环境变量读取，默认使用 hf-mirror.com（国内镜像）
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

# ── 颜色输出 ─────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

if sys.platform == "win32":
    os.system("")  # 激活 Windows ANSI 颜色


def ok(msg: str) -> str:
    return f"{GREEN}✓{RESET} {msg}"


def warn(msg: str) -> str:
    return f"{YELLOW}⚠{RESET} {msg}"


def err(msg: str) -> str:
    return f"{RED}✗{RESET} {msg}"


def info(msg: str) -> str:
    return f"{BLUE}ℹ{RESET} {msg}"


# ── 模型配置 ─────────────────────────────────────────────────────────────────
MODEL_REPO_ID = "FunAudioLLM/CosyVoice2-0.5B"  # CosyVoice3 的 HuggingFace 仓库
MODEL_DIR_NAME = "Fun-CosyVoice3-0.5B-2512"      # 本地存储目录名

# 需要下载的关键文件（用于验证完整性）
REQUIRED_FILES = [
    "cosyvoice.yaml",
    "campplus.onnx",
]


def check_huggingface_hub() -> bool:
    """
    检测 huggingface_hub 是否已安装。

    Returns:
        bool: 已安装返回 True
    """
    try:
        import huggingface_hub
        print(ok(f"huggingface_hub {huggingface_hub.__version__} 已安装"))
        return True
    except ImportError:
        print(err("huggingface_hub 未安装，请先运行 setup/install.bat"))
        return False


def download_model(model_dir: str, force: bool = False) -> bool:
    """
    下载 Fun-CosyVoice3-0.5B-2512 模型到指定目录。

    Args:
        model_dir: 模型存储根目录（相对或绝对路径）
        force: 是否强制重新下载（即使已存在）

    Returns:
        bool: 下载成功返回 True

    Raises:
        RuntimeError: 下载失败时抛出
    """
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub import HfApi

    target_dir = Path(model_dir) / MODEL_DIR_NAME
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}模型下载配置{RESET}")
    print(f"  仓库 ID  : {MODEL_REPO_ID}")
    print(f"  目标目录 : {target_dir.absolute()}")
    print(f"  HF 源    : {HF_ENDPOINT}")

    # 检查是否已下载
    if not force and _is_model_complete(str(target_dir)):
        print(ok("模型已完整下载，跳过（使用 --force 强制重新下载）"))
        return True

    print(f"\n{BOLD}开始下载模型...{RESET}")
    print(info("下载约 2GB，请耐心等待。国内用户建议使用 hf-mirror.com 镜像。"))

    try:
        # 使用 snapshot_download 下载整个仓库
        # huggingface_hub 会自动读取 HF_ENDPOINT 环境变量
        local_dir = snapshot_download(
            repo_id=MODEL_REPO_ID,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,  # Windows 不支持符号链接，使用实际文件
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],  # 跳过非 PyTorch 格式
        )
        print(ok(f"模型下载完成: {local_dir}"))

    except Exception as e:
        print(err(f"下载失败: {e}"))
        print(info("尝试备用方案：逐文件下载..."))

        # 备用方案：逐文件下载
        try:
            _download_files_individually(str(target_dir))
        except Exception as e2:
            print(err(f"备用下载也失败: {e2}"))
            print(info("请检查："))
            print(info("  1. 网络连接是否正常"))
            print(info("  2. HF_ENDPOINT 是否可访问"))
            print(info(f"  3. 尝试手动访问: {HF_ENDPOINT}/{MODEL_REPO_ID}"))
            return False

    # 验证完整性
    print(f"\n{BOLD}验证文件完整性...{RESET}")
    if _is_model_complete(str(target_dir)):
        print(ok("文件完整性验证通过"))
        _print_model_stats(str(target_dir))
        return True
    else:
        print(warn("部分关键文件缺失，下载可能不完整"))
        return False


def _download_files_individually(target_dir: str) -> None:
    """
    逐文件下载（当 snapshot_download 失败时的备用方案）。

    Args:
        target_dir: 本地存储目录路径
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    print(info("列出仓库文件列表..."))
    try:
        files = list(list_repo_files(MODEL_REPO_ID))
    except Exception as e:
        raise RuntimeError(f"无法获取文件列表: {e}")

    # 过滤不需要的文件
    skip_patterns = [".msgpack", "flax_model", "tf_model", "rust_model"]
    files = [f for f in files if not any(p in f for p in skip_patterns)]

    print(f"  共 {len(files)} 个文件需要下载")

    for i, filename in enumerate(files, 1):
        local_path = Path(target_dir) / filename
        if local_path.exists():
            print(info(f"  [{i}/{len(files)}] 已存在: {filename}"))
            continue

        print(f"  [{i}/{len(files)}] 下载: {filename}")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )


def _is_model_complete(target_dir: str) -> bool:
    """
    检查模型关键文件是否存在。

    Args:
        target_dir: 模型目录路径

    Returns:
        bool: 关键文件都存在返回 True
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
    打印模型目录的文件统计信息。

    Args:
        model_dir: 模型目录路径
    """
    total_size = 0
    file_count = 0
    dir_path = Path(model_dir)

    for filepath in dir_path.rglob("*"):
        if filepath.is_file():
            total_size += filepath.stat().st_size
            file_count += 1

    size_mb = total_size / (1024 ** 2)
    print(info(f"模型统计: {file_count} 个文件，共 {size_mb:.1f} MB"))


def save_download_info(model_dir: str) -> None:
    """
    保存下载信息到 download_info.json，便于后续版本管理。

    Args:
        model_dir: 模型根目录
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

    print(info(f"下载信息已保存: {info_path}"))


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="下载 Fun-CosyVoice3-0.5B-2512 预训练模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python setup/download_models.py
  python setup/download_models.py --model-dir /data/models
  python setup/download_models.py --force
  HF_ENDPOINT=https://hf-mirror.com python setup/download_models.py

镜像源说明:
  国内推荐使用 hf-mirror.com：
  export HF_ENDPOINT=https://hf-mirror.com  (Linux/Mac)
  set HF_ENDPOINT=https://hf-mirror.com     (Windows CMD)
        """
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "storage/pretrained_models"),
        help="模型存储目录（默认: storage/pretrained_models 或 MODEL_DIR 环境变量）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载，即使模型已存在"
    )
    return parser.parse_args()


def main() -> int:
    """
    主函数，执行模型下载流程。

    Returns:
        int: 0 表示成功，1 表示失败
    """
    print(f"\n{BOLD}{'═' * 52}{RESET}")
    print(f"{BOLD}  CosyVoice3 模型下载脚本{RESET}")
    print(f"{BOLD}{'═' * 52}{RESET}")
    print(f"  镜像源: {HF_ENDPOINT}")
    print(info("如需更换镜像源，请设置环境变量 HF_ENDPOINT"))

    args = parse_args()

    # 检测依赖
    if not check_huggingface_hub():
        return 1

    # 执行下载
    success = download_model(args.model_dir, force=args.force)

    if success:
        save_download_info(args.model_dir)
        print(f"\n{GREEN}{BOLD}✓ 模型下载完成！{RESET}")
        print(info("下一步: 复制 .env.example 为 .env，然后运行 start.bat 启动服务"))
        return 0
    else:
        print(f"\n{RED}{BOLD}✗ 模型下载失败，请检查上方错误信息。{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
