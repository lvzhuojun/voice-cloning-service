# Voice Cloning Service

基于 GPT-SoVITS v2 的语音克隆服务。

这个项目的目标很明确：

- 传入一批同一说话人的参考音频
- 在服务端完成 GPT-SoVITS 微调
- 导出该声音对应的模型文件
- 下游系统可以直接拿导出的模型做声音管理、分发和推理集成

## 这个项目现在能做什么

当前版本已经完成并验证了下面这条主流程：

1. 用户上传音频文件，或者选择 `data/samples/<folder_name>/` 下的本地音频目录
2. 服务执行 GPT-SoVITS 训练流水线
3. 训练完成后生成一个新的 `voice_id`
4. 在 `storage/models/{voice_id}/` 下导出该声音的模型文件
5. 前端或 API 可以直接对这个 `voice_id` 做试听
6. 也可以把模型文件下载给其他系统使用

## 训练完成后会得到什么

每次训练成功后，都会得到一个独立的声音目录：

```text
storage/models/{voice_id}/
  {voice_id}_gpt.ckpt
  {voice_id}_sovits.pth
  metadata.json
  reference.wav
```

例如：

```text
storage/models/1511e200-f24d-4346-8af9-29d253d0dde5/
  1511e200-f24d-4346-8af9-29d253d0dde5_gpt.ckpt
  1511e200-f24d-4346-8af9-29d253d0dde5_sovits.pth
  metadata.json
  reference.wav
```

这些文件的含义：

- `{voice_id}_gpt.ckpt`
  GPT 文本到语义模型权重
- `{voice_id}_sovits.pth`
  SoVITS 语音生成模型权重
- `metadata.json`
  记录声音名称、语言、训练参数、文件名等信息
- `reference.wav`
  当前服务内部试听时使用的参考音频

## 声音模型格式是什么

当前项目的标准输出格式不是旧 `.voicepack`，而是 GPT-SoVITS 双文件格式：

- `.ckpt`
- `.pth`

也就是：

```text
{voice_id}_gpt.ckpt + {voice_id}_sovits.pth
```

这是当前服务最核心的导出结果。

详细说明见 [VOICEPACK_FORMAT.md](VOICEPACK_FORMAT.md)。

## 这些模型存在哪里

模型统一保存在：

```text
storage/models/{voice_id}/
```

预训练基础模型保存在：

```text
storage/pretrained_models/GPT-SoVITS/
```

其中预训练目录包含：

```text
pretrained_s1.ckpt
pretrained_s2G.pth
pretrained_s2D.pth
chinese-hubert-base/
chinese-roberta-wwm-ext-large/
```

## 下游系统可以怎么用这些模型

当前项目已经支持把训练好的声音模型直接导出给下游系统。

下游系统可用方式包括：

- 直接下载 `.ckpt` 和 `.pth`
- 下载整包 ZIP
- 读取 `metadata.json` 获取声音信息

下载接口：

```http
GET /api/voices/{voice_id}/download/gpt
GET /api/voices/{voice_id}/download/sovits
GET /api/voices/{voice_id}/download/all
```

其中：

- `/download/gpt` 下载 GPT 模型
- `/download/sovits` 下载 SoVITS 模型
- `/download/all` 下载整包 ZIP

ZIP 内包含：

```text
{voice_id}_gpt.ckpt
{voice_id}_sovits.pth
metadata.json
reference.wav
```

## 是否可以给其他语音助手直接使用

可以，但要分两层理解。

第一层，模型导出和分发：

- 可以
- 当前项目已经能稳定导出 GPT-SoVITS 双文件模型
- 下游语音助手可以保存、管理、分发这些模型文件

第二层，直接用于下游推理：

- 取决于下游系统是否兼容 GPT-SoVITS 推理链路
- 如果下游也使用 GPT-SoVITS 官方推理方式，那么它可以直接加载这两个模型文件
- 如果下游是其他 TTS 框架，就需要它自己做适配

当前服务端自己的试听实现，已经改为使用 GPT-SoVITS 官方：

- `TTS_infer_pack.TTS.TTS`

也就是说，本项目内部已经证明：

- 训练出的模型可以被再次加载
- 加载后可以生成试听音频

## 目前试听的工作方式

当前试听接口：

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

注意：

- 当前服务端试听时仍会使用 `reference.wav`
- 这是当前服务内部的推理实现选择
- 不影响模型文件本身的导出和下游管理

## 已验证状态

当前仓库已经验证通过的内容：

- GPT-SoVITS 训练链路可完成
- 模型文件可导出到 `storage/models/{voice_id}/`
- 中文试听可返回音频
- 主仓库文档、代码、远程仓库已同步

当前最稳妥的使用建议：

- 训练和试听优先使用中文
- 若测试英文试听，需要额外依赖 `wordsegment`、`g2p_en` 和 NLTK 数据

## 快速使用步骤

### 1. 启动服务

```bat
start.bat
```

启动后访问：

- Web UI: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

### 2. 发起训练

方式一：本地目录训练

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

方式二：上传文件训练

```http
POST /api/train/from-upload
```

### 3. 查询训练状态

```http
GET /api/train/{task_id}/status
```

训练成功后会返回 `voice_id`。

### 4. 对已训练模型试听

```http
POST /api/voices/{voice_id}/test
```

### 5. 下载模型

```http
GET /api/voices/{voice_id}/download/gpt
GET /api/voices/{voice_id}/download/sovits
GET /api/voices/{voice_id}/download/all
```

## 项目目录

```text
app/
  api/
  core/

GPT-SoVITS/
  GPT_SoVITS/

setup/
  clone_gptsovits.bat
  clone_gptsovits.sh
  download_models.py

storage/
  models/
  pretrained_models/
  uploads/

data/
  samples/
```

## 相关文档

- [VOICEPACK_FORMAT.md](VOICEPACK_FORMAT.md)
- [IMPROVEMENTS.md](IMPROVEMENTS.md)
- [data/samples/README.md](data/samples/README.md)
