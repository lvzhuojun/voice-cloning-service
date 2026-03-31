# 语音克隆服务

**[English](README.md) | [中文](README_zh.md)**

基于 **GPT-SoVITS v2** 真实微调的 RESTful 语音克隆后端服务。

上传 1–3 分钟参考音频 → 服务端完成 GPT-SoVITS 微调 → 导出可携带的双文件模型（`.ckpt` + `.pth`），供下游系统管理、分发和推理集成。

---

## 功能特性

- **真实微调** — 完整的 GPT-SoVITS v2 训练流水线，非零样本克隆
- **8 步训练流水线** — 音频切片 → Whisper 转写 → BERT/HuBERT 特征提取 → GPT（s1）训练 → SoVITS（s2）训练 → 模型导出
- **可携带模型产物** — 每次训练生成独立目录（`{voice_id}_gpt.ckpt` + `{voice_id}_sovits.pth` + `metadata.json` + `reference.wav`）
- **完整 REST API** — 支持训练、试听、模型下载和声音管理
- **内置 Web 管理界面** — 访问 `/` 即可使用
- **Windows 原生支持** — 在 Windows 11 + RTX 显卡 + Anaconda 环境下验证通过

---

## 环境要求

| 组件 | 要求 |
|------|------|
| 操作系统 | Windows 10 / 11 |
| Conda | Miniconda 或 Anaconda |
| Python | 3.10（通过 conda 管理） |
| GPU | 支持 CUDA 的 NVIDIA 显卡 |
| 显存 | 建议 ≥ 6 GB |
| 磁盘 | 约 10 GB 可用空间（预训练模型 + 训练工作区） |

> **GPU 说明：** 默认 `setup/install.bat` 安装 PyTorch 2.7 + CUDA 12.8（适配 RTX 40/50 系列）。如需其他 CUDA 版本，请修改安装脚本。

---

## 安装步骤

### 第一步 — 克隆本仓库

```bash
git clone https://github.com/lvzhuojun/voice-cloning-service.git
cd voice-cloning-service
```

### 第二步 — 克隆 GPT-SoVITS

```bat
setup\clone_gptsovits.bat
```

将 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 源码克隆到 `GPT-SoVITS/` 目录（已加入 `.gitignore`，不随本仓库提交）。

### 第三步 — 安装依赖

```bat
setup\install.bat
```

该脚本会：
1. 创建 `voice-cloning` conda 环境（Python 3.10）
2. 安装 PyTorch 2.7 + CUDA 12.8
3. 从 `requirements-pip.txt` 安装所有 pip 依赖
4. 运行环境验证检查

### 第四步 — 下载预训练模型

```bat
conda run -n voice-cloning python setup/download_models.py
```

下载到 `storage/pretrained_models/GPT-SoVITS/`，总大小约 3–4 GB：

| 文件 | 说明 |
|------|------|
| `pretrained_s1.ckpt` | 基础 GPT（AR）模型 |
| `pretrained_s2G.pth` | 基础 SoVITS 生成器 |
| `pretrained_s2D.pth` | 基础 SoVITS 判别器 |
| `chinese-hubert-base/` | 音频特征提取器 |
| `chinese-roberta-wwm-ext-large/` | BERT 文本编码器 |

> **国内加速：** 下载器默认使用 `https://hf-mirror.com`。可在 `.env` 中设置 `HF_ENDPOINT` 替换镜像地址。

---

## 快速使用

### 1. 启动服务

```bat
start.bat
```

| 地址 | 说明 |
|------|------|
| `http://localhost:8000` | Web 管理界面 |
| `http://localhost:8000/docs` | Swagger API 文档 |
| `http://localhost:8000/api/health` | 服务健康检查 |

### 2. 发起训练

**方式一：本地目录训练**（将音频文件放入 `data/samples/<folder_name>/`）：

```http
POST /api/train/from-folder
Content-Type: application/json

{
  "folder_name": "my_speaker",
  "voice_name": "我的声音",
  "language": "zh"
}
```

**方式二：上传文件训练：**

```http
POST /api/train/from-upload
Content-Type: multipart/form-data

voice_name=我的声音
language=zh
files=<音频文件>
```

支持格式：`WAV / MP3 / M4A / FLAC / OGG`，最多 10 个文件，单文件最大 200 MB。

### 3. 查询训练状态

```http
GET /api/train/{task_id}/status
```

响应包含 `progress`（0–100）；当 `status` 为 `done` 时，`voice_id` 字段中返回声音 ID。

### 4. 试听合成

```http
POST /api/voices/{voice_id}/test
Content-Type: application/json

{
  "text": "你好，这是训练完成后的试听测试。",
  "speed": 1.0,
  "language": "zh"
}
```

返回 WAV 音频流。

### 5. 下载模型文件

```http
GET /api/voices/{voice_id}/download/gpt      # GPT 模型文件（.ckpt）
GET /api/voices/{voice_id}/download/sovits   # SoVITS 模型文件（.pth）
GET /api/voices/{voice_id}/download/all      # ZIP 压缩包（所有文件）
```

---

## 模型输出格式

每次训练成功后，会生成一个独立的声音目录：

```
storage/models/{voice_id}/
├── {voice_id}_gpt.ckpt      # GPT 文本到语义模型权重
├── {voice_id}_sovits.pth    # SoVITS 声码器生成器权重
├── metadata.json            # 声音元数据（名称、语言、训练参数等）
└── reference.wav            # 推理时使用的参考音频
```

**`metadata.json` 示例：**

```json
{
  "voice_id": "1511e200-f24d-4346-8af9-29d253d0dde5",
  "voice_name": "我的声音",
  "language": "zh",
  "created_at": "2026-03-30T15:24:37+00:00",
  "training_epochs_gpt": 15,
  "training_epochs_sovits": 8,
  "gpt_model_file": "1511e200-f24d-4346-8af9-29d253d0dde5_gpt.ckpt",
  "sovits_model_file": "1511e200-f24d-4346-8af9-29d253d0dde5_sovits.pth",
  "base_model_version": "GPT-SoVITS v2"
}
```

### 下游系统集成

任何支持 GPT-SoVITS 推理链的系统可直接加载这些文件：

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

## API 说明

### 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 服务状态、GPU 信息、声音数量、运行时长 |

### 训练管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/train/from-upload` | 从上传文件发起训练任务 |
| POST | `/api/train/from-folder` | 从本地目录发起训练任务 |
| GET | `/api/train/{task_id}/status` | 查询训练进度 |

### 声音管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/voices/` | 列出所有训练完成的声音 |
| GET | `/api/voices/{voice_id}` | 获取声音详情 |
| DELETE | `/api/voices/{voice_id}` | 删除声音及所有文件 |
| POST | `/api/voices/{voice_id}/test` | 试听合成（返回 WAV）|
| GET | `/api/voices/{voice_id}/download/gpt` | 下载 GPT 模型 |
| GET | `/api/voices/{voice_id}/download/sovits` | 下载 SoVITS 模型 |
| GET | `/api/voices/{voice_id}/download/all` | 下载 ZIP 压缩包 |

完整交互式文档见 `/docs`（Swagger）和 `/redoc`。

---

## 配置说明

将 `.env.example` 复制为 `.env` 并按需修改：

```env
HOST=0.0.0.0
PORT=8000

# HuggingFace 镜像（国内推荐）
HF_ENDPOINT=https://hf-mirror.com

# 训练参数
TRAINING_EPOCHS_GPT=15
TRAINING_EPOCHS_SOVITS=8
WHISPER_MODEL=medium

# 上传限制
MAX_UPLOAD_SIZE_MB=200
```

---

## 项目结构

```
voice-cloning-service/
├── app/
│   ├── api/            # FastAPI 路由（health、train、voices）
│   ├── core/           # 核心业务逻辑（trainer、tts_engine、voice_manager）
│   ├── models/         # Pydantic 数据模型
│   └── utils/          # 日志、文件工具
├── setup/
│   ├── install.bat     # 一键安装（Windows）
│   ├── clone_gptsovits.bat / .sh
│   ├── download_models.py
│   └── check_env.py
├── static/             # Web 管理界面
├── data/samples/       # 本地训练音频（已加入 .gitignore）
├── storage/            # 运行时数据（已加入 .gitignore）
│   ├── models/         # 已训练声音模型
│   ├── pretrained_models/
│   └── uploads/
├── tests/
├── GPT-SoVITS/         # GPT-SoVITS 源码（单独克隆，已加入 .gitignore）
├── start.bat
├── stop.bat
└── .env.example
```

---

## 注意事项

- **隐私 / 版权：** 本仓库默认**不跟踪**用户训练音频、训练生成的声音模型、预训练权重，以及单独克隆的 `GPT-SoVITS/` 源码；这些内容应只保留在 `data/`、`storage/` 和 `GPT-SoVITS/` 目录中。
- **推荐使用中文。** 中文（`zh`）的训练和试听已经过完整验证。
- **英文试听** 需要额外依赖（`wordsegment`、`g2p_en`、NLTK 数据），首次运行前请确认已安装。
- **训练任务状态** 保存在内存中，重启服务后任务历史将清空（磁盘上的模型文件不受影响）。
- **参考音频**（`reference.wav`）与模型文件一同保存，是当前内置试听接口的必要文件。

---

## License

本项目采用 [MIT License](LICENSE)。

GPT-SoVITS 是独立项目，拥有其自己的许可证，见 `GPT-SoVITS/LICENSE`。
