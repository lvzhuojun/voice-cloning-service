# Improvements

本文件记录本仓库从旧方案迁移到 GPT-SoVITS 过程中的主要改动。

## 1. 模型方案从 CosyVoice 迁移到 GPT-SoVITS

当前项目核心目标改为：

- 用户上传音频
- GPT-SoVITS 微调
- 导出 `{voice_id}_gpt.ckpt` + `{voice_id}_sovits.pth`
- 下游通过双文件模型进行管理和推理

## 2. 文档全面重写

已重写并对齐当前实现：

- `README.md`
- `MODEL_FORMAT.md`
- `IMPROVEMENTS.md`

## 3. Web UI 适配新模型格式

前端页面已经适配：

- GPT 模型下载
- SoVITS 模型下载
- 整包 ZIP 下载
- 训练阶段状态展示
- 试听接口调用

## 4. 训练器重写为 GPT-SoVITS 官方脚本流水线

`app/core/trainer.py` 当前采用 8 步训练流水线：

1. 切片音频
2. Whisper 转写
3. BERT 特征提取
4. HuBERT 特征提取
5. 语义 token 提取
6. GPT 训练
7. SoVITS 训练
8. 导出双文件模型

## 5. TTS 推理层改为官方 TTS 封装

`app/core/tts_engine.py` 已从旧手写模型加载改为官方：

- `TTS_infer_pack.TTS.TTS`
- 按 `voice_id` 缓存独立 TTS 实例
- 每个声音独立绑定自己的 GPT / SoVITS 权重

## 6. 预训练模型下载器更新

`setup/download_models.py` 已修复并对齐当前 HuggingFace 路径，下载目标包括：

- `pretrained_s1.ckpt`
- `pretrained_s2G.pth`
- `pretrained_s2D.pth`
- `chinese-hubert-base/`
- `chinese-roberta-wwm-ext-large/`

## 7. CosyVoice 到 GPT-SoVITS 迁移说明

关键变化：

- 不再以 `.voicepack` 作为核心输出
- 改为 GPT-SoVITS 双文件模型
- 推理阶段依赖 GPT-SoVITS 运行时
- 训练耗时和依赖复杂度高于旧方案

## 8. Windows 训练兼容性修复

为适配当前 Windows + RTX 5060 环境，已补充：

- GPT-SoVITS 子进程 `PYTHONPATH` 注入
- `bert_path` 环境变量注入
- `opencc` 依赖补齐
- DataLoader `num_workers=0`
- Windows Rich progress 编码问题修复
- 训练日志文件落盘，便于排错

## 9. Windows 推理兼容性修复

已处理的推理期问题包括：

- `ERes2NetV2` 模块搜索路径
- `fast_langdetect` 缓存目录
- `wordsegment`
- `g2p_en`
- `averaged_perceptron_tagger_eng`
- GPT-SoVITS 相对路径工作目录问题
- `bert_path` / `cnhubert_base_path` 环境变量问题

## 10. 当前已验证结果

当前版本已经验证：

- 训练可完成
- 模型目录可正确导出
- 中文试听可返回音频
- 主仓库可正常提交和推送

仍需注意：

- 英文试听首次运行依赖较多，部署到新环境时要确保相关包和 NLTK 数据存在
- GPT-SoVITS 控制台日志本身不稳定，判断是否完成应以前端是否收到音频为准
