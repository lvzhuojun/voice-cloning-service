# 训练数据放置说明

本目录用于放置语音克隆训练所需的参考音频。

---

## 目录结构

为每位说话人创建一个子文件夹，文件夹名即为训练时使用的 **音色标识**。

```
data/samples/
├── speaker_alice/          ← 说话人"alice"的参考音频
│   ├── sample_01.wav
│   ├── sample_02.wav
│   └── sample_03.wav
├── speaker_bob/            ← 说话人"bob"的参考音频
│   ├── clip1.mp3
│   ├── clip2.wav
│   └── clip3.m4a
└── my_voice/               ← 自定义名称均可
    ├── rec_001.wav
    └── rec_002.wav
```

---

## 音频要求

### 必须满足
- **每个说话人至少 3 段**，最多 10 段（更多不会提升效果）
- **每段时长**：5~30 秒（过短信息不足，过长建议裁剪）
- **支持格式**：WAV、MP3、M4A、FLAC、OGG（系统会自动转换）

### 强烈建议
- 安静环境录音，无明显背景噪音
- 说话人声音清晰，无遮挡或啸叫
- 内容为自然说话（非唱歌）
- 不同段内容有所不同，覆盖多种语调和句式
- 使用 16kHz 或更高采样率录音

### 不建议
- 不要放电话录音（8kHz 采样率质量不足）
- 不要放有背景音乐的音频
- 不要放多人混杂说话的音频

---

## 如何训练

### 方式一：通过 Web 界面
1. 启动服务（运行 `start.bat`）
2. 打开 `http://localhost:8000`
3. 在"本地文件夹训练"区填入子文件夹名（如 `speaker_alice`）
4. 填写自定义音色名称，点击"开始训练"

### 方式二：通过 API
```bash
curl -X POST http://localhost:8000/api/train/from-folder \
  -H "Content-Type: application/json" \
  -d '{
    "folder_name": "speaker_alice",
    "voice_name": "Alice 普通话",
    "language": "zh"
  }'
```

---

## 数据隐私说明

`data/samples/` 目录已被 `.gitignore` 排除在版本控制之外，
**不会被 git 追踪或上传到远程仓库**，请放心使用个人录音数据。
