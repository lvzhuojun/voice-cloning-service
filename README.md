# Voice Cloning Service

基于 **GPT-SoVITS v2** 的语音音色克隆训练服务。
上传 1~3 分钟参考音频，自动完成切片、转录、特征提取和模型微调，
训练完成后导出独立的 GPT + SoVITS 双模型文件，供下游语音对话服务直接加载使用，**无需参考音频**。

---

## 与 CosyVoice3 方案的核心区别

| 对比项 | CosyVoice3（旧方案）| GPT-SoVITS（当前方案）|
|--------|--------------------|-----------------------|
| 推理依赖 | 每次推理都需提供参考音频 | 模型文件独立，推理零参考 |
| 可移植性 | 无法导出独立音色模型 | 导出 `.ckpt` + `.pth`，跨服务加载 |
| 音色学习方式 | Zero-shot embedding | **真实微调（Fine-tuning）** |
| 训练时长 | 无需训练 | 约 5~15 分钟（GPU）|
| 适用场景 | 快速体验 | **生产级部署、专属音色** |

详细迁移说明见 [IMPROVEMENTS.md](IMPROVEMENTS.md)。

---

## 技术栈

- **核心引擎**：GPT-SoVITS v2（AR + VITS 双模型架构）
- **语音识别**：OpenAI Whisper（训练时自动转录）
- **特征提取**：chinese-hubert-base（音频）+ chinese-roberta-wwm-ext-large（文本）
- **后端框架**：FastAPI + Uvicorn
- **音频处理**：pydub、librosa、soundfile、ffmpeg
- **运行环境**：Python 3.10，conda 管理
- **GPU 支持**：NVIDIA RTX 5060（Blackwell / sm_120），需 PyTorch >= 2.7

---

## API 文档概览

服务启动后访问 `http://localhost:8000/docs` 查看完整 Swagger 文档。

### 健康检查
```
GET /api/health          # 服务状态、GPU 信息、已有音色数量
```

### 训练相关
```
POST /api/train/from-upload         # 上传音频文件，创建训练任务
POST /api/train/from-folder         # 指定 data/samples/ 子文件夹，创建训练任务
GET  /api/train/{task_id}/status    # 查询训练进度（含当前阶段描述）
```

### 音色管理
```
GET    /api/voices                          # 列出所有已训练音色
GET    /api/voices/{voice_id}               # 获取音色详情
GET    /api/voices/{voice_id}/download/gpt      # 下载 GPT 模型 (.ckpt)
GET    /api/voices/{voice_id}/download/sovits   # 下载 SoVITS 模型 (.pth)
GET    /api/voices/{voice_id}/download/all      # 打包下载（含两个模型 + metadata.json）
DELETE /api/voices/{voice_id}               # 删除音色
POST   /api/voices/{voice_id}/test          # 合成测试语音
```

---

## 训练流程

上传音频后，后台自动完成以下步骤（可通过 `GET /api/train/{task_id}/status` 实时追踪进度）：

```
1. 音频切片     （3~10 秒/段，静音点分割）
2. Whisper 转录 （自动生成文本标注）
3. Hubert 特征提取  （CUDA 加速）
4. 语义 Token 提取  （s2G 量化器）
5. BERT 特征提取    （chinese-roberta-wwm-ext-large）
6. GPT 模型微调     （s1 AR 模型，约 3~8 分钟）
7. SoVITS 模型微调  （s2 VITS 模型，约 2~5 分钟）
8. 保存模型文件 + metadata.json
```

训练完成后，模型保存在：
```
storage/models/{voice_id}/
├── {voice_id}_gpt.ckpt
├── {voice_id}_sovits.pth
└── metadata.json
```

---

## 目录结构

```
voice-cloning-service/
├── app/
│   ├── api/          # FastAPI 路由（health, train, voices）
│   ├── core/         # 核心逻辑（trainer, tts_engine, voice_manager）
│   ├── models/       # Pydantic schema 定义
│   └── utils/        # 工具函数
├── data/
│   └── samples/      # 本地训练音频文件夹（按说话人子文件夹组织）
├── GPT-SoVITS/       # GPT-SoVITS 源码（单独克隆，不入版本库）
├── setup/
│   ├── install.bat           # 一键安装脚本
│   ├── clone_gptsovits.bat   # 克隆 GPT-SoVITS（Windows）
│   ├── clone_gptsovits.sh    # 克隆 GPT-SoVITS（Linux/Mac）
│   ├── download_models.py    # 下载预训练底模
│   └── check_env.py          # 环境检测脚本
├── static/
│   └── index.html    # Web 管理界面
├── storage/
│   ├── models/       # 训练好的音色模型（不入版本库）
│   ├── pretrained_models/    # 预训练底模（不入版本库）
│   └── uploads/      # 临时上传文件（不入版本库）
├── .env.example      # 环境变量配置示例
├── environment.yml   # conda 环境定义
├── requirements-pip.txt  # pip 依赖列表
└── start.bat         # 启动脚本
```

---

## 数据准备

在 `data/samples/` 目录下为每个说话人创建子文件夹，放入参考音频：

```
data/samples/
├── speaker_alice/
│   ├── sample_01.wav
│   └── sample_02.mp3
└── speaker_bob/
    └── recording.m4a
```

- 支持格式：WAV / MP3 / M4A / FLAC / OGG
- 建议总时长：1~3 分钟
- 音频要求：安静环境，语速正常，单说话人

---

## 模型文件格式

训练完成后输出两个独立模型文件，详细规范见 [VOICEPACK_FORMAT.md](VOICEPACK_FORMAT.md)。

---

## 快速开始

### 前置要求

- Windows 11
- NVIDIA GPU（推荐 RTX 5060 或更高，需 CUDA 支持）
- [Anaconda](https://www.anaconda.com/) 或 Miniconda
- Git（支持 SSH 推送；Git for Windows 包含 Git Bash）

> **RTX 5060 用户注意**：该显卡为 Blackwell 架构（compute capability sm_120），需要 PyTorch >= 2.7。`setup/install.bat` 已自动安装正确版本。

### 步骤 1：克隆本仓库

```bash
git clone git@github.com:lvzhuojun/voice-cloning-service.git
cd voice-cloning-service
```

### 步骤 2：安装 conda 环境及依赖

双击运行 `setup\install.bat`，脚本自动完成：
- 创建 `voice-cloning` conda 环境（Python 3.10）
- 安装 PyTorch 2.7 + CUDA 12.8
- 安装所有 pip 依赖（含 Whisper、transformers 等）

### 步骤 3：克隆 GPT-SoVITS 源码

双击运行 `setup\clone_gptsovits.bat`，将 GPT-SoVITS 克隆到 `./GPT-SoVITS/`。

### 步骤 4：下载预训练底模

```bat
conda run -n voice-cloning python setup/download_models.py
```

将从 HuggingFace 下载约 3~4 GB 文件：
- `pretrained_s1.ckpt`（GPT base 模型）
- `pretrained_s2G.pth` + `pretrained_s2D.pth`（SoVITS base 模型）
- `chinese-hubert-base/`（音频特征提取器）
- `chinese-roberta-wwm-ext-large/`（文本 BERT 编码器）

### 步骤 5：配置环境变量

```bat
copy .env.example .env
```

根据实际情况修改 `.env`（默认值通常无需改动）。

### 步骤 6：启动服务

```bat
start.bat
```

### 步骤 7：使用服务

- **Web 管理界面**：打开浏览器访问 `http://localhost:8000`
- **API 文档**：访问 `http://localhost:8000/docs`

---

## 常见问题

**Q：RTX 5060 无法识别 CUDA？**
A：确保使用 `setup/install.bat` 安装了 PyTorch 2.7 + cu128。运行 `python setup/check_env.py` 查看详细检测结果。

**Q：Whisper 转录速度很慢？**
A：默认使用 `medium` 模型。在 `.env` 中将 `WHISPER_MODEL=small` 可加速，但转录精度会降低。大型 GPU 可设置 `large`。

**Q：训练报 "GPT-SoVITS directory not found"？**
A：先运行 `setup\clone_gptsovits.bat` 克隆 GPT-SoVITS 源码。

**Q：训练报 "chinese-hubert-base not found"？**
A：先运行 `python setup/download_models.py` 下载预训练底模。

**Q：模型下载失败？**
A：检查网络连接，或在 `.env` 中设置 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像。

---

## License

本项目后端框架基于 MIT 协议。GPT-SoVITS 遵循其原始 MIT 许可证，详见 `GPT-SoVITS/LICENSE`。
