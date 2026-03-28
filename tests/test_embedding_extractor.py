"""
EmbeddingExtractor 单元测试

测试多段音频加权融合的核心逻辑，不依赖预训练模型（使用降级模式）。
"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.audio_processor import AudioProcessor, STANDARD_SAMPLE_RATE
from app.core.embedding_extractor import EmbeddingExtractor, COSYVOICE_EMBEDDING_DIM
from app.models.schemas import QualityReport


def make_audio(duration=5.0, sr=STANDARD_SAMPLE_RATE, amplitude=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def make_quality_report(score=0.8, duration=5.0):
    return QualityReport(
        duration_seconds=duration,
        duration_ok=True,
        snr_db=20.0,
        quality_score=score,
        has_long_silence=False,
        sample_rate=STANDARD_SAMPLE_RATE,
        channels=1,
        warnings=[],
    )


class TestEmbeddingExtractorFallback(unittest.TestCase):
    """测试降级模式（无预训练模型）下的 embedding 提取"""

    def setUp(self):
        # 使用不存在的模型目录，强制使用降级模式
        self.extractor = EmbeddingExtractor(model_dir="/nonexistent/model/dir")

    def test_extract_single_returns_tensor(self):
        """单段提取应返回 torch.Tensor"""
        audio = make_audio(duration=5.0)
        emb = self.extractor.extract_single(audio, STANDARD_SAMPLE_RATE)
        self.assertIsInstance(emb, torch.Tensor)

    def test_extract_single_dim(self):
        """提取结果维度应为 COSYVOICE_EMBEDDING_DIM"""
        audio = make_audio(duration=5.0)
        emb = self.extractor.extract_single(audio, STANDARD_SAMPLE_RATE)
        self.assertEqual(emb.ndim, 1)
        self.assertEqual(emb.shape[0], COSYVOICE_EMBEDDING_DIM)

    def test_extract_single_normalized(self):
        """提取结果应为 L2 归一化向量（模长约为1）"""
        audio = make_audio(duration=5.0)
        emb = self.extractor.extract_single(audio, STANDARD_SAMPLE_RATE)
        norm = torch.norm(emb, p=2).item()
        self.assertAlmostEqual(norm, 1.0, delta=0.01)

    def test_extract_and_fuse_single(self):
        """单段音频融合应等同于单段提取"""
        audio = make_audio(duration=5.0)
        sr = STANDARD_SAMPLE_RATE
        report = make_quality_report(score=1.0)

        fused, weights = self.extractor.extract_and_fuse([(audio, sr, report)])
        self.assertIsInstance(fused, torch.Tensor)
        self.assertEqual(fused.shape[0], COSYVOICE_EMBEDDING_DIM)
        self.assertEqual(len(weights), 1)
        self.assertAlmostEqual(weights["segment_0"], 1.0, places=3)

    def test_extract_and_fuse_multi_weights(self):
        """多段融合：权重应与质量分成比例"""
        sr = STANDARD_SAMPLE_RATE
        pairs = [
            (make_audio(5.0), sr, make_quality_report(score=0.2)),
            (make_audio(5.0), sr, make_quality_report(score=0.8)),
        ]
        _, weights = self.extractor.extract_and_fuse(pairs)

        self.assertEqual(len(weights), 2)
        # 第二段质量更高，权重应更大
        self.assertGreater(weights["segment_1"], weights["segment_0"])
        # 权重之和应约为 1.0
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, delta=0.01)

    def test_extract_and_fuse_equal_weights(self):
        """所有段质量相等时，权重应均等"""
        sr = STANDARD_SAMPLE_RATE
        pairs = [
            (make_audio(5.0), sr, make_quality_report(score=0.5)),
            (make_audio(5.0), sr, make_quality_report(score=0.5)),
            (make_audio(5.0), sr, make_quality_report(score=0.5)),
        ]
        _, weights = self.extractor.extract_and_fuse(pairs)

        expected_weight = 1.0 / 3
        for w in weights.values():
            self.assertAlmostEqual(w, expected_weight, delta=0.01)

    def test_extract_and_fuse_empty_raises(self):
        """空列表应抛出 ValueError"""
        with self.assertRaises(ValueError):
            self.extractor.extract_and_fuse([])

    def test_extract_and_fuse_result_normalized(self):
        """融合结果应为 L2 归一化"""
        sr = STANDARD_SAMPLE_RATE
        pairs = [(make_audio(5.0), sr, make_quality_report(0.8)) for _ in range(3)]
        fused, _ = self.extractor.extract_and_fuse(pairs)
        norm = torch.norm(fused, p=2).item()
        self.assertAlmostEqual(norm, 1.0, delta=0.01)

    def test_select_reference_audio_prefers_duration(self):
        """参考音频选取应优先选时长在 5~15 秒内的段"""
        sr = STANDARD_SAMPLE_RATE
        # 第一段时长合适，质量低；第二段时长过长，质量高
        short_report = make_quality_report(score=0.3, duration=8.0)   # 合适时长
        long_report = QualityReport(
            duration_seconds=30.0, duration_ok=True, snr_db=30.0,
            quality_score=0.9, has_long_silence=False,
            sample_rate=sr, channels=1, warnings=[],
        )
        pairs = [
            (make_audio(8.0), sr, short_report),
            (make_audio(30.0), sr, long_report),
        ]
        best_audio, best_sr, best_report = self.extractor.select_reference_audio(pairs)
        # 应选时长合适的那段（即使质量较低）
        self.assertAlmostEqual(best_report.duration_seconds, 8.0, delta=0.1)

    def test_select_reference_audio_fallback(self):
        """若无时长合适的段，选质量最高的"""
        sr = STANDARD_SAMPLE_RATE
        pairs = [
            (make_audio(1.0), sr, QualityReport(
                duration_seconds=1.0, duration_ok=False, snr_db=10.0,
                quality_score=0.3, has_long_silence=False, sample_rate=sr, channels=1, warnings=[]
            )),
            (make_audio(2.0), sr, QualityReport(
                duration_seconds=2.0, duration_ok=False, snr_db=25.0,
                quality_score=0.7, has_long_silence=False, sample_rate=sr, channels=1, warnings=[]
            )),
        ]
        _, _, best_report = self.extractor.select_reference_audio(pairs)
        self.assertAlmostEqual(best_report.quality_score, 0.7, delta=0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
