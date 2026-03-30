# Sample Data Layout

`data/samples/` 用于本地目录训练模式。

接口：

```http
POST /api/train/from-folder
```

请求体里的 `folder_name` 就对应这里的子目录名。

## Directory Structure

示例：

```text
data/samples/
  1wav/
    001.wav
    002.wav
    003.wav

  speaker_alice/
    sample_01.wav
    sample_02.mp3

  speaker_bob/
    clip_01.m4a
    clip_02.flac
```

## Supported Formats

支持：

- `.wav`
- `.mp3`
- `.m4a`
- `.flac`
- `.ogg`

## Recommendations

- 单个声音建议提供 1 到 3 分钟干净语音
- 尽量使用同一个说话人
- 避免背景音乐、多人对话、强混响
- 服务端最多使用前 10 个文件
- 训练器会自动切成 3 到 10 秒的片段

## Example Request

```json
{
  "folder_name": "1wav",
  "voice_name": "test_voice",
  "language": "zh"
}
```

## Note

`data/samples/` 主要用于本地调试和开发验证，不建议把真实训练数据提交到 git。
