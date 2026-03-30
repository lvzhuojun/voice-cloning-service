# GPT-SoVITS 模型格式规范

本文档定义了本项目训练完成后输出的音色模型文件格式，供下游项目（语音对话助手等）直接加载使用。

---

## 文件结构

训练完成后，每个音色保存在独立目录中：

```
storage/models/{voice_id}/
├── {voice_id}_gpt.ckpt      ← GPT（s1 AR 模型）检查点
├── {voice_id}_sovits.pth    ← SoVITS（s2G 生成器）权重
└── metadata.json            ← 音色元数据
```

> `voice_id` 为 UUID v4 格式，例如 `3f2a1b4c-5d6e-7f8a-9b0c-1d2e3f4a5b6c`

---

## metadata.json 字段说明

```json
{
  "voice_id": "3f2a1b4c-5d6e-7f8a-9b0c-1d2e3f4a5b6c",
  "voice_name": "用户自定义名称（如：Alice-普通话）",
  "created_at": "2024-01-15T10:30:00+00:00",
  "language": "zh",
  "audio_duration_seconds": 142.5,
  "training_epochs_gpt": 15,
  "training_epochs_sovits": 8,
  "gpt_model_file": "{voice_id}_gpt.ckpt",
  "sovits_model_file": "{voice_id}_sovits.pth",
  "base_model_version": "GPT-SoVITS v2"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `voice_id` | string | UUID v4，全局唯一标识符 |
| `voice_name` | string | 用户自定义的音色名称 |
| `created_at` | string | ISO 8601 时间戳（含时区） |
| `language` | string | 主要语言，`"zh"` 或 `"en"` |
| `audio_duration_seconds` | float | 训练用音频总时长（秒） |
| `training_epochs_gpt` | integer | GPT 模型微调轮数 |
| `training_epochs_sovits` | integer | SoVITS 模型微调轮数 |
| `gpt_model_file` | string | GPT 模型文件名（`.ckpt`） |
| `sovits_model_file` | string | SoVITS 模型文件名（`.pth`） |
| `base_model_version` | string | 底模版本，固定值 `"GPT-SoVITS v2"` |

---

## 下游项目加载推理示例

以下示例展示另一个项目（如语音对话助手）如何加载本项目训练好的两个模型文件，直接进行语音合成，**无需参考音频**。

### 前置条件

```bash
# 克隆 GPT-SoVITS 源码（与训练服务共用同一个 GPT-SoVITS 仓库即可）
git clone https://github.com/RVC-Boss/GPT-SoVITS.git GPT-SoVITS
pip install torch transformers librosa soundfile
```

### 完整推理示例

```python
"""
voice_inference.py
加载 voice-cloning-service 训练好的双模型文件，直接合成语音。
将 VOICE_DIR、GPT_SOVITS_DIR、PRETRAINED_DIR 替换为你的实际路径。
"""

import sys
import json
import io
import numpy as np
import torch
import soundfile as sf

# ── 路径配置 ──────────────────────────────────────────────────────────────────
VOICE_DIR      = "storage/models/3f2a1b4c-5d6e-7f8a-9b0c-1d2e3f4a5b6c"
GPT_SOVITS_DIR = "GPT-SoVITS"
PRETRAINED_DIR = "storage/pretrained_models/GPT-SoVITS"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# GPT-SoVITS 源码加入 sys.path
sys.path.insert(0, GPT_SOVITS_DIR)
sys.path.insert(0, f"{GPT_SOVITS_DIR}/GPT_SoVITS")

# ── 读取 metadata ─────────────────────────────────────────────────────────────
with open(f"{VOICE_DIR}/metadata.json", encoding="utf-8") as f:
    meta = json.load(f)

voice_id    = meta["voice_id"]
gpt_path    = f"{VOICE_DIR}/{meta['gpt_model_file']}"
sovits_path = f"{VOICE_DIR}/{meta['sovits_model_file']}"

print(f"[INFO] Voice: {meta['voice_name']} | Language: {meta['language']}")
print(f"[INFO] GPT:    {gpt_path}")
print(f"[INFO] SoVITS: {sovits_path}")

# ── 加载 SoVITS 生成器 ────────────────────────────────────────────────────────
from module.models import SynthesizerTrn

sovits_ckpt = torch.load(sovits_path, map_location="cpu")
hps = sovits_ckpt.get("config", {})
model_hps = hps.get("model", {
    "hidden_channels": 192, "filter_channels": 768,
    "n_heads": 2, "n_layers": 6, "kernel_size": 3, "p_dropout": 0.0,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [8, 8, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "n_layers_q": 3, "use_spectral_norm": False,
    "gin_channels": 512, "semantic_frame_rate": "25hz",
})
data_hps = hps.get("data", {"filter_length": 1024, "hop_length": 320, "sampling_rate": 32000})

sovits_model = SynthesizerTrn(
    spec_channels=data_hps["filter_length"] // 2 + 1,
    segment_size=hps.get("train", {}).get("segment_size", 20),
    inter_channels=model_hps["hidden_channels"],
    hidden_channels=model_hps["hidden_channels"],
    filter_channels=model_hps["filter_channels"],
    n_heads=model_hps["n_heads"],
    n_layers=model_hps["n_layers"],
    kernel_size=model_hps["kernel_size"],
    p_dropout=0.0,
    resblock=model_hps["resblock"],
    resblock_kernel_sizes=model_hps["resblock_kernel_sizes"],
    resblock_dilation_sizes=model_hps["resblock_dilation_sizes"],
    upsample_rates=model_hps["upsample_rates"],
    upsample_initial_channel=model_hps["upsample_initial_channel"],
    upsample_kernel_sizes=model_hps["upsample_kernel_sizes"],
    n_layers_q=model_hps["n_layers_q"],
    use_spectral_norm=model_hps["use_spectral_norm"],
    gin_channels=model_hps["gin_channels"],
    semantic_frame_rate=model_hps["semantic_frame_rate"],
)
sovits_model.load_state_dict(sovits_ckpt.get("weight", sovits_ckpt), strict=False)
sovits_model.eval().to(DEVICE)
print("[OK] SoVITS model loaded")

# ── 加载 GPT AR 模型 ──────────────────────────────────────────────────────────
from AR.models.t2s_lightning_module import Text2SemanticLightningModule

gpt_ckpt   = torch.load(gpt_path, map_location="cpu")
gpt_config = gpt_ckpt.get("config", {})
gpt_model  = Text2SemanticLightningModule(config=gpt_config, top_k=3, is_train=False)
gpt_model.load_state_dict(
    gpt_ckpt.get("weight", gpt_ckpt.get("state_dict", gpt_ckpt)), strict=False
)
gpt_model.eval().to(DEVICE)
print("[OK] GPT model loaded")

# ── 加载 HuBERT 特征提取器 ────────────────────────────────────────────────────
from transformers import HubertModel, Wav2Vec2FeatureExtractor

hubert_dir      = f"{PRETRAINED_DIR}/chinese-hubert-base"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_dir)
hubert_model    = HubertModel.from_pretrained(hubert_dir)
hubert_model.eval().to(DEVICE)
print("[OK] HuBERT model loaded")


# ── 合成函数 ──────────────────────────────────────────────────────────────────
def synthesize(
    text: str,
    ref_wav_path: str,   # 短暂的参考音频（3~10 秒），用于声纹调制
    language: str = "zh",
    speed: float = 1.0,
    output_path: str = "output.wav",
) -> str:
    """
    使用加载好的微调模型合成语音。

    Args:
        text:          要合成的文本
        ref_wav_path:  参考音频路径（用于声纹特征，3~10 秒即可）
        language:      语言代码 ("zh" / "en")
        speed:         语速（0.5~2.0）
        output_path:   输出 WAV 路径

    Returns:
        str: 输出文件路径
    """
    import librosa

    # 读取并重采样参考音频 → 16kHz（HuBERT 要求）
    ref_audio, ref_sr = sf.read(ref_wav_path)
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    ref_16k = librosa.resample(ref_audio.astype(np.float32), orig_sr=ref_sr, target_sr=16000)

    # 提取参考音频的 HuBERT 特征 → 语义 token
    with torch.no_grad():
        inputs = feature_extractor(ref_16k, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        hubert_out = hubert_model(**inputs, output_hidden_states=True)
        ssl_content = hubert_out.hidden_states[9].squeeze(0).unsqueeze(0).transpose(1, 2)
        prompt_semantic = sovits_model.extract_latent(ssl_content)[0, 0]

    # GPT AR 解码：生成目标文本的语义 token 序列
    with torch.no_grad():
        pred_semantic = gpt_model.model.infer_panel(
            x=torch.zeros(1, 1, dtype=torch.long, device=DEVICE),
            x_lens=torch.tensor([1], device=DEVICE),
            prompts=prompt_semantic.unsqueeze(0).to(DEVICE),
            bert_feature=torch.zeros(1, 1024, 1, device=DEVICE),
            top_k=15,
            early_stop_num=50,
            temperature=1.0,
        )

    # SoVITS 解码：语义 token → 音频波形
    with torch.no_grad():
        ref_audio_32k = librosa.resample(ref_audio.astype(np.float32), orig_sr=ref_sr, target_sr=32000)
        refer = sovits_model._spec_from_wav(ref_audio_32k, 32000, DEVICE)
        audio_out = sovits_model.decode(
            pred_semantic.unsqueeze(0).unsqueeze(0),
            torch.LongTensor([pred_semantic.shape[-1]]).to(DEVICE),
            refer,
        )[0, 0].data.cpu().float().numpy()

    audio_out = np.clip(audio_out, -1.0, 1.0)
    sf.write(output_path, audio_out, 32000)
    print(f"[OK] Saved: {output_path}")
    return output_path


# ── 使用示例 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    synthesize(
        text="你好，我是一个语音克隆测试。请评估音质是否符合预期。",
        ref_wav_path="path/to/any_short_sample.wav",  # 任意 3~10 秒参考音
        language="zh",
        speed=1.0,
        output_path="output.wav",
    )
```

### 通过服务 API 调用（推荐）

如果下游服务运行于同一局域网，直接调用 REST API 更简洁：

```python
import requests

# 合成测试语音
resp = requests.post(
    "http://localhost:8000/api/voices/{voice_id}/test",
    json={"text": "你好，这是语音克隆测试。", "speed": 1.0, "language": "zh"},
)
if resp.ok:
    with open("output.wav", "wb") as f:
        f.write(resp.content)
    print("Saved output.wav")

# 下载 GPT 模型
resp = requests.get("http://localhost:8000/api/voices/{voice_id}/download/gpt")
with open("voice_gpt.ckpt", "wb") as f:
    f.write(resp.content)

# 打包下载全部文件（ZIP）
resp = requests.get("http://localhost:8000/api/voices/{voice_id}/download/all")
with open("voice_model.zip", "wb") as f:
    f.write(resp.content)
```

---

## 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 2.0 | 2026-03 | 从 CosyVoice3 迁移到 GPT-SoVITS v2 双文件格式，废弃 .voicepack 格式 |
| 1.0 | 2024-01 | 初始版本，CosyVoice3 + .voicepack ZIP 格式 |
