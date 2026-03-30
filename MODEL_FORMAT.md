# GPT-SoVITS Model Output Format

本项目当前的标准输出是 GPT-SoVITS 双文件模型格式，而不是旧 `.voicepack`。

## 导出结果

每个训练成功的声音都会生成一个独立目录：

```text
storage/models/{voice_id}/
  {voice_id}_gpt.ckpt
  {voice_id}_sovits.pth
  metadata.json
  reference.wav
```

## 文件说明

### 1. `{voice_id}_gpt.ckpt`

GPT 文本到语义模型。

作用：

- 负责把文本特征转换成语义 token
- 属于 GPT-SoVITS 推理链的一部分

### 2. `{voice_id}_sovits.pth`

SoVITS 语音生成模型。

作用：

- 负责把语义 token 转成最终音频
- 属于 GPT-SoVITS 推理链的一部分

### 3. `metadata.json`

声音元数据文件。

典型内容包括：

- `voice_id`
- `voice_name`
- `language`
- `created_at`
- `training_epochs_gpt`
- `training_epochs_sovits`
- `gpt_model_file`
- `sovits_model_file`
- `base_model_version`

示例：

```json
{
  "voice_id": "1511e200-f24d-4346-8af9-29d253d0dde5",
  "voice_name": "test_voice",
  "language": "zh",
  "created_at": "2026-03-30T15:24:37+00:00",
  "training_epochs_gpt": 15,
  "training_epochs_sovits": 8,
  "gpt_model_file": "1511e200-f24d-4346-8af9-29d253d0dde5_gpt.ckpt",
  "sovits_model_file": "1511e200-f24d-4346-8af9-29d253d0dde5_sovits.pth",
  "base_model_version": "GPT-SoVITS v2"
}
```

### 4. `reference.wav`

训练过程中保留下来的一段参考音频。

当前用途：

- 服务端试听时作为 `ref_audio_path`

说明：

- 这不影响模型双文件本身的导出
- 下游是否需要参考音频，取决于它自己的推理实现

## 模型存储位置

模型统一存放在：

```text
storage/models/{voice_id}/
```

例如：

```text
storage/models/1511e200-f24d-4346-8af9-29d253d0dde5/
```

## 下载方式

服务提供以下模型下载接口：

```http
GET /api/voices/{voice_id}/download/gpt
GET /api/voices/{voice_id}/download/sovits
GET /api/voices/{voice_id}/download/all
```

对应含义：

- `/download/gpt`
  下载 `{voice_id}_gpt.ckpt`
- `/download/sovits`
  下载 `{voice_id}_sovits.pth`
- `/download/all`
  下载完整 ZIP 包

ZIP 包中包含：

```text
{voice_id}_gpt.ckpt
{voice_id}_sovits.pth
metadata.json
reference.wav
```

## 下游系统如何使用

下游系统若兼容 GPT-SoVITS 推理链，可以直接使用这套模型文件：

- 加载 `{voice_id}_gpt.ckpt`
- 加载 `{voice_id}_sovits.pth`
- 使用 GPT-SoVITS 的 `TTS_infer_pack.TTS.TTS` 或兼容实现进行推理

本项目内部当前就是这样做的。

推理配置示例：

```python
from pathlib import Path
from TTS_infer_pack.TTS import TTS, TTS_Config

voice_id = "1511e200-f24d-4346-8af9-29d253d0dde5"
voice_dir = Path("storage/models") / voice_id
pretrained_dir = Path("storage/pretrained_models/GPT-SoVITS")

config = TTS_Config(
    {
        "custom": {
            "device": "cuda",
            "is_half": True,
            "version": "v2",
            "t2s_weights_path": str(voice_dir / f"{voice_id}_gpt.ckpt"),
            "vits_weights_path": str(voice_dir / f"{voice_id}_sovits.pth"),
            "cnhuhbert_base_path": str(pretrained_dir / "chinese-hubert-base"),
            "bert_base_path": str(pretrained_dir / "chinese-roberta-wwm-ext-large"),
        }
    }
)

tts = TTS(config)
```

## 这个项目交付给下游的核心价值

对下游系统来说，这个项目交付的核心不是“一个参考音频文件”，而是：

- 一个稳定的 `voice_id`
- 一套可管理、可下载、可归档的声音模型文件
- 一套和该声音绑定的元数据

换句话说，下游拿到的是“声音模型资产”，而不是一次性生成结果。

## 当前结论

当前项目已经能够：

- 从音频训练出 GPT-SoVITS 模型
- 把模型保存到 `storage/models/{voice_id}/`
- 通过 API 下载这些模型
- 通过当前服务对模型做试听验证

如果下游语音助手需要接入，这些导出的文件就是它的集成入口。
