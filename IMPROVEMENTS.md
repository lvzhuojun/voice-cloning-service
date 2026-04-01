# Improvements

本文档保留本仓库迁移到 GPT-SoVITS 过程中的关键背景，便于后续维护时快速理解当前实现。

## 1. 模型方案迁移

项目当前核心输出已经统一为 GPT-SoVITS 双文件模型：

- `{voice_id}_gpt.ckpt`
- `{voice_id}_sovits.pth`
- `metadata.json`
- `reference.wav`

## 2. 当前主流程

当前服务的主要链路为：

1. 上传音频或读取 `data/samples/` 本地目录
2. 执行 GPT-SoVITS 官方训练流水线
3. 导出双文件模型
4. 通过内置 API 进行试听、下载和删除管理

## 3. 仓库约束

为避免把真实训练资源提交到远端，当前仓库约束如下：

- `data/samples/` 中的训练音频不提交
- `storage/models/` 中的训练后声音模型不提交
- `storage/pretrained_models/` 中的预训练权重不提交
- `storage/uploads/` 中的临时上传文件不提交
- `GPT-SoVITS/` 为单独克隆源码，不随本仓库提交

## 4. 维护提示

- 文档入口以 `README.md` / `README_zh.md` 为准
- 模型导出格式说明以 `MODEL_FORMAT.md` 为准
- 当前 `/api/voices/{voice_id}/download/all` 使用临时文件打包 ZIP，由 `FileResponse` 分块流式传输后自动清理，避免将 1 GB+ 模型文件全部载入内存
