# .voicepack 格式规范

`.voicepack` 是本项目定义的标准音色包格式，用于存储和传输语音克隆所需的全部信息。
它是一个标准 ZIP 文件，后缀名为 `.voicepack`。

---

## 文件结构

```
{voice_id}.voicepack          ← ZIP 文件，以 voice_id 命名
└── {voice_id}/               ← ZIP 内部根目录，与文件名相同
    ├── speaker_embedding.pt  ← torch.Tensor，说话人 embedding 向量
    ├── reference_audio.wav   ← 最优参考音频（用于推理时输入模型）
    ├── model_config.json     ← 模型相关配置
    └── metadata.json         ← 音色元数据
```

> `voice_id` 为 UUID v4 格式字符串，例如 `3f2a1b4c-5d6e-7f8a-9b0c-1d2e3f4a5b6c`

---

## metadata.json 字段说明

```json
{
  "voice_id": "3f2a1b4c-5d6e-7f8a-9b0c-1d2e3f4a5b6c",
  "voice_name": "用户自定义名称（如：小明-普通话）",
  "created_at": "2024-01-15T10:30:00+08:00",
  "model_type": "cosyvoice3",
  "model_version": "Fun-CosyVoice3-0.5B-2512",
  "language": "zh",
  "sample_count": 5,
  "total_duration_seconds": 87.3,
  "embedding_dim": 192,
  "quality_score": 0.82
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `voice_id` | string | UUID v4，全局唯一标识符 |
| `voice_name` | string | 用户自定义的音色名称 |
| `created_at` | string | ISO 8601 时间戳（含时区） |
| `model_type` | string | 固定值 `"cosyvoice3"` |
| `model_version` | string | 使用的模型版本，固定值 `"Fun-CosyVoice3-0.5B-2512"` |
| `language` | string | 主要语言，`"zh"` 或 `"en"` |
| `sample_count` | integer | 训练时使用的参考音频数量 |
| `total_duration_seconds` | float | 所有参考音频的总时长（秒） |
| `embedding_dim` | integer | speaker embedding 向量维度 |
| `quality_score` | float | 综合音质评分，范围 0.0~1.0（由各段 SNR 加权平均得出） |

---

## model_config.json 字段说明

```json
{
  "sample_rate": 22050,
  "model_type": "cosyvoice3",
  "embedding_method": "multi_audio_weighted_average",
  "cosyvoice_config": {}
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `sample_rate` | integer | 参考音频采样率（Hz），固定 `22050` |
| `model_type` | string | 固定值 `"cosyvoice3"` |
| `embedding_method` | string | embedding 提取方法，`"multi_audio_weighted_average"` 表示多段加权融合 |
| `cosyvoice_config` | object | 保留字段，存储 CosyVoice3 特定配置，当前为空对象 |

---

## speaker_embedding.pt

使用 `torch.save()` 序列化的 `torch.Tensor`，形状为 `(embedding_dim,)`，数据类型为 `float32`。

---

## reference_audio.wav

- 格式：WAV，PCM 16-bit
- 采样率：22050 Hz（CosyVoice3 要求）
- 声道：单声道（Mono）
- 时长：5~15 秒（从所有参考音频中选取质量最高的一段）

---

## Python 读写示例

### 打包（创建 .voicepack）

```python
import zipfile
import json
import torch
import uuid
from datetime import datetime, timezone

voice_id = str(uuid.uuid4())
output_path = f"storage/voicepacks/{voice_id}.voicepack"

metadata = {
    "voice_id": voice_id,
    "voice_name": "测试音色",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model_type": "cosyvoice3",
    "model_version": "Fun-CosyVoice3-0.5B-2512",
    "language": "zh",
    "sample_count": 5,
    "total_duration_seconds": 87.3,
    "embedding_dim": 192,
    "quality_score": 0.82,
}

model_config = {
    "sample_rate": 22050,
    "model_type": "cosyvoice3",
    "embedding_method": "multi_audio_weighted_average",
    "cosyvoice_config": {},
}

# speaker_embedding 是形状为 (192,) 的 torch.Tensor
embedding_tensor: torch.Tensor = ...

with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
    # 写入 embedding
    import io
    buf = io.BytesIO()
    torch.save(embedding_tensor, buf)
    zf.writestr(f"{voice_id}/speaker_embedding.pt", buf.getvalue())

    # 写入参考音频
    with open("best_reference.wav", "rb") as f:
        zf.writestr(f"{voice_id}/reference_audio.wav", f.read())

    # 写入 JSON 配置
    zf.writestr(f"{voice_id}/metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    zf.writestr(f"{voice_id}/model_config.json", json.dumps(model_config, ensure_ascii=False, indent=2))
```

### 解包（读取 .voicepack）

```python
import zipfile
import json
import torch
import io

voicepack_path = "storage/voicepacks/{voice_id}.voicepack"

with zipfile.ZipFile(voicepack_path, "r") as zf:
    # 读取文件列表，确定 voice_id（ZIP 内根目录名）
    names = zf.namelist()
    voice_id = names[0].split("/")[0]

    # 读取 metadata
    with zf.open(f"{voice_id}/metadata.json") as f:
        metadata = json.load(f)

    # 读取 model_config
    with zf.open(f"{voice_id}/model_config.json") as f:
        model_config = json.load(f)

    # 读取 speaker embedding
    with zf.open(f"{voice_id}/speaker_embedding.pt") as f:
        buf = io.BytesIO(f.read())
        embedding = torch.load(buf, map_location="cpu")  # shape: (embedding_dim,)

    # 读取参考音频（写入临时文件供模型使用）
    with zf.open(f"{voice_id}/reference_audio.wav") as f:
        audio_bytes = f.read()
```

### 验证完整性

```python
REQUIRED_FILES = {"speaker_embedding.pt", "reference_audio.wav", "metadata.json", "model_config.json"}

def validate_voicepack(path: str) -> bool:
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if not names:
            return False
        voice_id = names[0].split("/")[0]
        inner_files = {n.split("/", 1)[1] for n in names if "/" in n}
        return REQUIRED_FILES.issubset(inner_files)
```

---

## 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0 | 2024-01 | 初始版本，定义基本格式 |
