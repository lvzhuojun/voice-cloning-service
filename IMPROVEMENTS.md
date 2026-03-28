# 工程改进说明

本项目在 CosyVoice3 原始代码基础上进行了以下 6 项工程改进，
目标是将研究代码转化为可生产部署的服务。

---

## 改进 1：多段音频加权融合 Speaker Embedding

**原始方案**：CosyVoice3 推理时仅接受单段参考音频，音色稳定性依赖该段音频质量。

**本项目方案**：
- 接受 3~10 段参考音频
- 对每段音频独立提取 speaker embedding 向量
- 按各段音频的**信噪比（SNR）评分**计算归一化权重
- 对所有 embedding 进行**加权平均融合**，得到更稳定的音色表示

**实现位置**：`app/core/embedding_extractor.py` → `extract_and_fuse()`

**效果**：减少单段录音质量对最终音色的影响，提高克隆稳定性。

---

## 改进 2：标准化 .voicepack 音色包格式

**原始方案**：无标准的音色存储格式，每次推理需重新指定参考音频路径。

**本项目方案**：
- 定义 `.voicepack`（ZIP 格式）标准规范
- 内含：embedding 向量、参考音频、模型配置、元数据
- 唯一 `voice_id`（UUID v4）标识每个音色
- 便于跨项目、跨服务传输和复用

**实现位置**：`app/core/voicepack_manager.py`，格式规范见 `VOICEPACK_FORMAT.md`

**效果**：音色作为"资产"管理，一次训练，多处使用。

---

## 改进 3：音频质量自动检测与预处理

**原始方案**：原始 CosyVoice 未对输入参考音频做质量检测，低质量音频会导致音色克隆失败或效果差。

**本项目方案**：
- **格式标准化**：自动转换为 16kHz 单声道 WAV（librosa 实现）
- **质量检测**：检测时长合规性、估算信噪比、检测长时间静音段
- **简单降噪**：高通滤波去除麦克风低频噪声（< 80Hz）
- **最优片段选取**：从长音频自动截取信噪比最高的片段（滑动窗口）

**实现位置**：`app/core/audio_processor.py`

**效果**：降低对用户录音质量的要求，提高成功率。

---

## 改进 4：REST API 服务化封装

**原始方案**：CosyVoice3 提供命令行脚本和 Gradio Demo，无法程序化调用。

**本项目方案**：
- 完整的 FastAPI REST API
- 支持文件上传训练和本地文件夹训练两种模式
- 任务异步执行，支持进度查询（task_id 机制）
- 完整的音色 CRUD 管理接口
- 自动生成 Swagger/OpenAPI 文档

**实现位置**：`app/api/`，入口 `app/main.py`

---

## 改进 5：国内环境适配

**原始方案**：模型下载默认走 HuggingFace 官方源，国内访问不稳定。

**本项目方案**：
- 支持 `HF_ENDPOINT` 环境变量配置镜像源
- 默认使用 `https://hf-mirror.com`
- 下载脚本显示进度条，完成后校验文件完整性
- `environment.yml` 使用可靠 channel 顺序

**实现位置**：`setup/download_models.py`，`.env.example`

---

## 改进 6：RTX 5060 / Blackwell 架构 CUDA 兼容性处理

**背景**：
- NVIDIA RTX 5060 采用 Blackwell 架构（compute capability sm_120）
- PyTorch < 2.7 的稳定版本不包含 sm_120 的 CUDA kernel 编译
- 直接使用旧版 PyTorch 会导致 GPU 无法识别，回退到 CPU 运行

**本项目方案**：
- `environment.yml` 明确要求 PyTorch >= 2.7（通过 pip 从官方 wheel 源安装）
- 使用 `cu121` wheel（CUDA 12.1），CUDA 13.x 向下兼容 CUDA 12.x API
- `setup/check_env.py` 自动检测 GPU compute capability 并给出友好提示
- `setup/install.bat` 安装完成后运行环境检测，及时发现兼容性问题

**实现位置**：`environment.yml`，`setup/check_env.py`，`setup/install.bat`
