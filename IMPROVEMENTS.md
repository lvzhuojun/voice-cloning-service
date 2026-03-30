# 工程改进说明

本文档记录了本项目相对于原始参考实现所做的工程改进。

---

## 改进 1（原）：多段音频加权融合 Speaker Embedding（已废弃）

> **状态**：已由 GPT-SoVITS 真实微调方案取代（见改进 7）。

**原始方案**：CosyVoice3 推理时仅接受单段参考音频。

**历史方案**：
- 接受 3~10 段参考音频，按 SNR 加权融合 speaker embedding
- 输出 `.voicepack` ZIP 文件（含 embedding + 参考音频）

---

## 改进 2（原）：标准化 .voicepack 音色包格式（已废弃）

> **状态**：已由 GPT-SoVITS 双模型文件格式取代（见改进 7）。

**历史格式**：
- `.voicepack`（ZIP 格式）内含 `speaker_embedding.pt` + `reference_audio.wav` + JSON 配置
- 缺点：每次推理仍需参考音频，无法导出独立音色模型

---

## 改进 3（原）：音频质量自动检测与预处理

**保留**，在 `app/core/audio_processor.py` 中仍然有效：
- 格式标准化（自动转 16kHz 单声道 WAV）
- 质量检测（时长、SNR、静音段）
- 高通滤波降噪（< 80 Hz）
- 最优片段选取（滑动窗口）

---

## 改进 4（原）：REST API 服务化封装

完整 FastAPI REST API，保留并扩展：
- 支持文件上传训练和本地文件夹训练
- 异步任务执行 + 进度查询（task_id 机制）
- 新增三个下载端点（`/download/gpt`、`/download/sovits`、`/download/all`）
- 自动生成 Swagger/OpenAPI 文档

---

## 改进 5（原）：国内环境适配

保留 `HF_ENDPOINT` 环境变量支持，可在 `.env` 中配置镜像源。

---

## 改进 6（原）：RTX 5060 / Blackwell 架构 CUDA 兼容性处理

保留，`setup/install.bat` 安装 PyTorch 2.7 + CUDA 12.8（cu128）以支持 sm_120。

---

## 改进 7（新）：从 CosyVoice3 迁移到 GPT-SoVITS v2 真实微调

### 背景与动机

原始方案基于 CosyVoice3（Zero-shot TTS），存在以下根本性局限：

| 问题 | 说明 |
|------|------|
| 每次推理需参考音频 | 无法将"音色"从参考音频中分离出来独立保存 |
| 无法导出独立模型文件 | 无法将训练好的音色移植到其他服务 |
| Zero-shot 质量上限 | 音色相似度和稳定性不如真实微调方案 |

本项目目标是将训练好的音色提供给另一个**语音对话助手**项目使用，要求：
1. 导出独立模型文件，不依赖参考音频
2. 模型可在对话助手中直接加载，无需运行本服务
3. 合成效果稳定，不受每次请求的参考音频质量波动影响

### 改动范围

**替换的组件：**

| 文件 | 旧（CosyVoice3）| 新（GPT-SoVITS）|
|------|----------------|----------------|
| `app/core/embedding_extractor.py` | CosyVoice3 embedding 提取 | 已移除，功能并入 trainer |
| `app/core/voicepack_manager.py` | `.voicepack` ZIP 管理 | 已移除，替换为 `voice_manager.py` |
| `app/core/trainer.py` | 不存在（无需训练） | `GPTSoVITSTrainer`（完整 8 步流水线）|
| `app/core/tts_engine.py` | CosyVoice3 zero-shot 推理 | GPT-SoVITS 微调模型推理 |
| `app/api/voices.py` | 单一 `/download` 端点 | `/download/gpt`、`/download/sovits`、`/download/all` |
| `setup/download_models.py` | 下载 CosyVoice3 模型 | 下载 GPT-SoVITS 五个预训练组件 |
| `VOICEPACK_FORMAT.md` | `.voicepack` 格式规范 | GPT-SoVITS 双模型文件格式规范 |

**新增的组件：**
- `app/core/voice_manager.py`：管理 `storage/models/{voice_id}/` 目录结构
- `setup/clone_gptsovits.bat`：Windows 一键克隆脚本（检测 Git Bash）
- `setup/clone_gptsovits.sh`：bash 版克隆脚本

### 新训练流程（GPT-SoVITS 8 步流水线）

```
1. 音频切片     → pydub 按静音点切割，3~10 秒/段
2. Whisper 转录 → 自动生成文本标注，支持 zh/en/ja/ko
3. Hubert 特征  → chinese-hubert-base，CUDA 加速，输出 .pt 文件
4. 语义 Token   → pretrained_s2G.pth 量化器，从 Hubert 特征提取
5. BERT 特征    → chinese-roberta-wwm-ext-large，文本语义编码
6. GPT 微调     → s1 AR 模型，以 pretrained_s1.ckpt 为底模
7. SoVITS 微调  → s2G/s2D，以 pretrained_s2G/s2D.pth 为底模
8. 保存输出     → {voice_id}_gpt.ckpt + {voice_id}_sovits.pth + metadata.json
```

### 训练时间参考（RTX 5060，约 60 段 3~10 秒音频）

| 步骤 | 预计时间 |
|------|---------|
| 切片 + 转录（Whisper medium）| 2~4 分钟 |
| 特征提取（Hubert + 语义 + BERT）| 1~3 分钟 |
| GPT 微调（15 轮）| 3~6 分钟 |
| SoVITS 微调（8 轮）| 2~4 分钟 |
| **总计** | **约 8~17 分钟** |

### 输出格式

```
storage/models/{voice_id}/
├── {voice_id}_gpt.ckpt      # GPT AR 模型
├── {voice_id}_sovits.pth    # SoVITS 生成器
└── metadata.json            # 元数据（时长、轮数、版本等）
```

下游推理示例见 [VOICEPACK_FORMAT.md](VOICEPACK_FORMAT.md)。
