# GPT-SoVITS Model Format

当前项目不再使用旧的 `.voicepack` ZIP 作为核心模型格式。

当前标准输出为 GPT-SoVITS 双文件格式：

```text
storage/models/{voice_id}/
  {voice_id}_gpt.ckpt
  {voice_id}_sovits.pth
  metadata.json
  reference.wav
```

## Files

### `{voice_id}_gpt.ckpt`

GPT 文本到语义模型权重。

用途：

- 文本特征到语义 token 预测
- 推理时由 GPT-SoVITS `TTS_infer_pack.TTS.TTS` 加载

### `{voice_id}_sovits.pth`

SoVITS 模型权重。

用途：

- 语义 token 到音频波形生成
- 推理时由 GPT-SoVITS `TTS_infer_pack.TTS.TTS` 加载

### `metadata.json`

声音元数据。

示例：

```json
{
  "voice_id": "1511e200-f24d-4346-8af9-29d253d0dde5",
  "voice_name": "test_voice",
  "language": "zh",
  "created_at": "2026-03-30T15:24:37.000000+00:00",
  "audio_duration_seconds": 0.0,
  "training_epochs_gpt": 15,
  "training_epochs_sovits": 8,
  "gpt_model_file": "1511e200-f24d-4346-8af9-29d253d0dde5_gpt.ckpt",
  "sovits_model_file": "1511e200-f24d-4346-8af9-29d253d0dde5_sovits.pth",
  "base_model_version": "GPT-SoVITS v2"
}
```

### `reference.wav`

训练流程保留的一段参考音频。

注意：

- 当前服务端试听接口仍会把它作为 `ref_audio_path`
- 这不影响双文件模型导出
- 如果下游系统要做到“完全不依赖任何参考音频”，需要它自己的推理方案也支持这种工作方式

## Inference Example

当前项目内部已经改为使用 GPT-SoVITS 官方 `TTS_infer_pack.TTS.TTS`。

最小示例：

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

inputs = {
    "text": "你好，这是一次试听测试。",
    "text_lang": "zh",
    "ref_audio_path": str(voice_dir / "reference.wav"),
    "prompt_text": "",
    "prompt_lang": "zh",
    "speed_factor": 1.0,
    "top_k": 15,
    "top_p": 1.0,
    "temperature": 1.0,
    "batch_size": 1,
}

sample_rate, audio = next(tts.run(inputs))
```

## API Downloads

服务提供三个下载接口：

```http
GET /api/voices/{voice_id}/download/gpt
GET /api/voices/{voice_id}/download/sovits
GET /api/voices/{voice_id}/download/all
```

其中 `/download/all` 会返回一个 ZIP，包含：

```text
{voice_id}_gpt.ckpt
{voice_id}_sovits.pth
metadata.json
reference.wav
```

## Compatibility Notes

- 当前标准版本：GPT-SoVITS v2
- 当前训练导出目标：`.ckpt + .pth`
- 已废弃：旧 CosyVoice `.voicepack` 主格式

如果你在新环境部署：

1. 先安装 `requirements-pip.txt`
2. 运行 `python setup/download_models.py`
3. 确保 `GPT-SoVITS` 子目录已 clone
