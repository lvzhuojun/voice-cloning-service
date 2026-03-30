# Voice Cloning Service

基于 GPT-SoVITS v2 的语音克隆服务。

当前版本已经完成从旧 CosyVoice 方案到 GPT-SoVITS 的核心迁移，支持：

- 上传或选择本地音频发起训练
- 自动执行切片、Whisper 转写、BERT/HuBERT/语义特征提取
- 训练并导出双文件模型
- 通过 Web UI 或 API 管理声音模型
- 使用已训练模型直接进行试听

导出格式：

```text
storage/models/{voice_id}/
  {voice_id}_gpt.ckpt
  {voice_id}_sovits.pth
  metadata.json
  reference.wav
```

## Current Status

当前仓库状态和代码一致：

- GPT-SoVITS 训练链路已实测跑通
- 模型导出已实测跑通
- `/api/voices/{voice_id}/test` 试听链路已打通
- 主仓库已推送到 `git@github.com:lvzhuojun/voice-cloning-service.git`

已知注意项：

- 中文试听是当前最稳的路径
- 英文试听依赖 `wordsegment`、`g2p_en` 和 NLTK 数据
- 首次英文试听时，`fast-langdetect` 可能会下载语言识别模型
- GPT-SoVITS 上游推理日志不会稳定打印“完成”字样，只要前端已返回音频，就说明本次推理已经完成

## Environment

项目当前实际运行环境：

- OS: Windows 11
- Python: 3.10
- Conda env: `voice-cloning`
- Conda root: `D:\Anaconda3`
- GPU: RTX 5060 Laptop
- PyTorch: 2.7.0 + cu128
- GPT-SoVITS dir: `./GPT-SoVITS/`

## Quick Start

### 1. Clone repo

```bat
git clone git@github.com:lvzhuojun/voice-cloning-service.git
cd voice-cloning-service
```

### 2. Prepare environment

推荐使用仓库里的安装脚本，或至少保证你已经有：

- `voice-cloning` conda 环境
- Python 3.10
- PyTorch 2.7 + CUDA 12.8
- `requirements-pip.txt` 里的依赖

如果你是复用当前环境，确保以下命令成功：

```bat
conda activate voice-cloning
python -m pip install -r requirements-pip.txt
```

英文 TTS 额外依赖：

```bat
python -m pip install wordsegment g2p_en
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

### 3. Clone GPT-SoVITS

```bat
setup\clone_gptsovits.bat
```

### 4. Download pretrained models

```bat
conda activate voice-cloning
python setup/download_models.py
```

下载完成后至少应存在：

```text
storage/pretrained_models/GPT-SoVITS/pretrained_s1.ckpt
storage/pretrained_models/GPT-SoVITS/pretrained_s2G.pth
storage/pretrained_models/GPT-SoVITS/pretrained_s2D.pth
storage/pretrained_models/GPT-SoVITS/chinese-hubert-base/config.json
storage/pretrained_models/GPT-SoVITS/chinese-roberta-wwm-ext-large/config.json
```

### 5. Start service

```bat
start.bat
```

启动后访问：

- Web UI: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/api/health`

## Training Workflow

训练接口有两种。

### Option A: Train from local folder

把音频放到：

```text
data/samples/<folder_name>/
```

调用：

```http
POST /api/train/from-folder
```

请求体示例：

```json
{
  "folder_name": "1wav",
  "voice_name": "test_voice",
  "language": "zh"
}
```

### Option B: Train from upload

调用：

```http
POST /api/train/from-upload
```

表单字段：

- `voice_name`
- `language`
- `files`

### Task status

训练是异步任务。创建成功后会返回 `task_id`。

查询接口：

```http
GET /api/train/{task_id}/status
```

训练完成后状态里会返回 `voice_id`。

## Voice Management

### List voices

```http
GET /api/voices
```

### Get one voice

```http
GET /api/voices/{voice_id}
```

### Download models

```http
GET /api/voices/{voice_id}/download/gpt
GET /api/voices/{voice_id}/download/sovits
GET /api/voices/{voice_id}/download/all
```

### Delete voice

```http
DELETE /api/voices/{voice_id}
```

## Test TTS

试听接口：

```http
POST /api/voices/{voice_id}/test
```

请求体示例：

```json
{
  "text": "你好，这是训练完成后的试听测试。",
  "speed": 1.0,
  "language": "zh"
}
```

建议先用中文短句验证：

- `你好`
- `你今天吃的什么`
- `今天天气不错，我们下午去公园散步吧。`

如果前端已经返回音频并可以播放，就说明本次推理完成了。不要单纯依据 GPT-SoVITS 控制台日志是否打印“结束”来判断。

## Training Output

每个声音目录包含：

```text
storage/models/{voice_id}/
  {voice_id}_gpt.ckpt
  {voice_id}_sovits.pth
  metadata.json
  reference.wav
```

其中：

- `gpt.ckpt` 是 GPT 文本到语义模型
- `sovits.pth` 是 SoVITS 声码器相关模型
- `metadata.json` 保存声音名称、语言、训练参数等
- `reference.wav` 是服务内部试听时使用的参考音频

详细格式见 [VOICEPACK_FORMAT.md](VOICEPACK_FORMAT.md)。

## Project Structure

```text
app/
  main.py
  api/
    train.py
    voices.py
  core/
    trainer.py
    tts_engine.py
    voice_manager.py

GPT-SoVITS/
  GPT_SoVITS/
    TTS_infer_pack/
    prepare_datasets/
    configs/

setup/
  clone_gptsovits.bat
  clone_gptsovits.sh
  download_models.py

storage/
  models/
  pretrained_models/

static/
  index.html
```

## Notes For Deployment

当前版本在 Windows 本地开发环境下已经针对以下问题做了兼容处理：

- HuggingFace 预训练模型下载路径更新
- GPT-SoVITS 子进程 `PYTHONPATH` 注入
- `opencc` / `wordsegment` / `g2p_en` 等运行时依赖补齐
- Windows DataLoader `num_workers=0` 兼容
- Windows Rich progress GBK 编码问题
- 推理期 `ERes2NetV2`、`fast_langdetect`、`bert_path`、工作目录硬编码问题

如果你迁移到新机器，优先确认：

1. `requirements-pip.txt` 全部安装完成
2. `setup/download_models.py` 下载成功
3. 英文 TTS 需要的 NLTK 数据已下载

## Related Docs

- [VOICEPACK_FORMAT.md](VOICEPACK_FORMAT.md)
- [IMPROVEMENTS.md](IMPROVEMENTS.md)
- [data/samples/README.md](data/samples/README.md)
