# Voice Cloning Service

基于 **CosyVoice3**（Fun-CosyVoice3-0.5B-2512）的语音音色克隆训练服务。
通过上传 3~10 段参考音频，即可提取说话人音色特征并打包为可复用的 `.voicepack` 文件，供下游 TTS 服务调用。

---

## 与原始 CosyVoice 的区别

| 特性 | 原始 CosyVoice | 本项目 |
|------|--------------|--------|
| 音色提取方式 | 单段参考音频 | **多段音频加权融合**（按信噪比加权） |
| 音色复用格式 | 无标准格式 | **标准化 `.voicepack` ZIP 格式** |
| 服务化 | 仅推理脚本 | **完整 REST API（FastAPI）** |
| 音频预处理 | 基础处理 | **自动质量检测 + 降噪 + 最优片段选取** |
| 管理界面 | 无 | **Web 管理界面（纯 HTML/JS）** |
| 国内适配 | 无 | **HuggingFace 镜像源支持** |

详细改进说明见 [IMPROVEMENTS.md](IMPROVEMENTS.md)。

---

## 技术栈

- **底座模型**：CosyVoice3 (Fun-CosyVoice3-0.5B-2512)
- **后端框架**：FastAPI + Uvicorn
- **音频处理**：librosa、soundfile、ffmpeg
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
POST /api/train/from-upload    # 上传音频文件，创建训练任务
POST /api/train/from-folder    # 指定服务器文件夹，创建训练任务
GET  /api/train/{task_id}/status   # 查询任务进度
```

### 音色管理
```
GET    /api/voices                        # 列出所有音色
GET    /api/voices/{voice_id}             # 获取音色详情
GET    /api/voices/{voice_id}/download    # 下载 .voicepack 文件
DELETE /api/voices/{voice_id}             # 删除音色
POST   /api/voices/{voice_id}/test        # 合成测试语音
```

---

## 数据准备

在 `data/samples/` 目录下为每个说话人创建子文件夹，放入参考音频：

```
data/samples/
├── speaker_alice/
│   ├── sample_01.wav
│   ├── sample_02.wav
│   └── sample_03.wav
└── speaker_bob/
    ├── clip1.mp3
    └── clip2.wav
```

详细说明见 [data/samples/README.md](data/samples/README.md)。

---

## .voicepack 格式

本项目定义了标准的音色包格式 `.voicepack`，供下游项目对接。
详细规范见 [VOICEPACK_FORMAT.md](VOICEPACK_FORMAT.md)。

---

## 快速开始

### 前置要求

- Windows 11
- NVIDIA GPU（推荐 RTX 5060 或更高，需 CUDA 支持）
- [Anaconda](https://www.anaconda.com/) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git（支持 SSH）

> **RTX 5060 用户注意**：该显卡为 Blackwell 架构（compute capability sm_120），需要 PyTorch >= 2.7。安装脚本已自动处理此兼容性问题。

### 步骤 1：克隆仓库

```bash
git clone git@github.com:lvzhuojun/voice-cloning-service.git
cd voice-cloning-service
```

> 如果使用了 CosyVoice 子模块，还需执行：
> ```bash
> git submodule update --init --recursive
> ```

### 步骤 2：安装环境

双击运行 `setup/install.bat`，或在命令行执行：

```batch
setup\install.bat
```

脚本将自动：
- 创建 `voice-cloning` conda 环境（Python 3.10）
- 安装所有依赖（含 PyTorch >= 2.7 with CUDA）
- 运行环境检测脚本

### 步骤 3：下载预训练模型

```batch
conda activate voice-cloning
python setup/download_models.py
```

国内用户默认使用 `https://hf-mirror.com` 镜像，约需下载 2GB 模型文件。

### 步骤 4：配置环境变量

```batch
copy .env.example .env
```

根据实际情况修改 `.env`（通常默认值即可）。

### 步骤 5：启动服务

```batch
start.bat
```

### 步骤 6：使用服务

- **Web 管理界面**：打开浏览器访问 `http://localhost:8000`
- **API 文档**：访问 `http://localhost:8000/docs`

### 步骤 7：训练音色（本地文件夹方式）

1. 在 `data/samples/` 下创建子文件夹，放入 3~10 段 WAV/MP3 音频
2. 在 Web 界面"本地文件夹训练"区填入子文件夹名，点击"开始训练"
3. 等待训练完成，音色包保存在 `storage/voicepacks/`

---

## 常见问题

**Q：RTX 5060 无法识别 CUDA？**
A：确保 PyTorch >= 2.7，运行 `python setup/check_env.py` 查看详细检测结果。

**Q：模型下载失败？**
A：检查 `.env` 中 `HF_ENDPOINT` 是否设置为可用镜像，或尝试科学上网后设置 `HF_ENDPOINT=https://huggingface.co`。

**Q：音质评分很低？**
A：确保录音环境安静，单段音频 5~30 秒，建议使用 16kHz 或更高采样率。

---

## License

本项目基于 CosyVoice 开源协议，详见 [LICENSE](LICENSE)。
