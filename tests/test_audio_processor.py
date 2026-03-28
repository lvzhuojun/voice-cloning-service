"""
音频预处理模块单元测试

测试 AudioProcessor 的核心功能：格式转换、质量检测、降噪、片段选取。
使用纯 numpy 生成测试音频信号，不依赖真实音频文件。
"""

import os
import sys
import tempfile
import unittest
import numpy as np

# 确保可以 import app 模块（从项目根目录运行）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.audio_processor import AudioProcessor, STANDARD_SAMPLE_RATE
from app.models.schemas import QualityReport, BatchProcessReport


def generate_sine_wave(
    frequency: float = 440.0,
    duration: float = 5.0,
    sample_rate: int = STANDARD_SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """生成正弦波测试音频"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def generate_noisy_sine(
    duration: float = 5.0,
    sample_rate: int = STANDARD_SAMPLE_RATE,
    noise_level: float = 0.05,
) -> np.ndarray:
    """生成带噪声的正弦波"""
    clean = generate_sine_wave(440.0, duration, sample_rate, 0.5)
    noise = (np.random.randn(len(clean)) * noise_level).astype(np.float32)
    return clean + noise


class TestAudioProcessorInit(unittest.TestCase):
    """测试 AudioProcessor 初始化"""

    def test_default_init(self):
        """测试默认参数初始化"""
        processor = AudioProcessor()
        self.assertEqual(processor.min_duration, 3.0)
        self.assertEqual(processor.max_duration, 60.0)
        self.assertEqual(processor.target_sample_rate, STANDARD_SAMPLE_RATE)

    def test_custom_init(self):
        """测试自定义参数初始化"""
        processor = AudioProcessor(min_duration=5.0, max_duration=30.0, target_sample_rate=16000)
        self.assertEqual(processor.min_duration, 5.0)
        self.assertEqual(processor.max_duration, 30.0)
        self.assertEqual(processor.target_sample_rate, 16000)


class TestConvertToStandard(unittest.TestCase):
    """测试格式转换功能"""

    def setUp(self):
        self.processor = AudioProcessor()
        self.sample_rate = STANDARD_SAMPLE_RATE

    def test_convert_wav_file(self):
        """测试 WAV 文件转换"""
        audio = generate_sine_wave(duration=5.0, sample_rate=self.sample_rate)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import soundfile as sf
            sf.write(f.name, audio, self.sample_rate)
            input_path = f.name

        try:
            result_audio, result_sr = self.processor.convert_to_standard(input_path)
            self.assertIsInstance(result_audio, np.ndarray)
            self.assertEqual(result_sr, self.sample_rate)
            self.assertEqual(result_audio.dtype, np.float32)
        finally:
            os.unlink(input_path)

    def test_convert_and_save(self):
        """测试转换并保存文件"""
        audio = generate_sine_wave(duration=5.0, sample_rate=self.sample_rate)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            import soundfile as sf
            sf.write(f_in.name, audio, self.sample_rate)
            input_path = f_in.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
            output_path = f_out.name

        try:
            self.processor.convert_to_standard(input_path, output_path)
            self.assertTrue(os.path.exists(output_path))
            size = os.path.getsize(output_path)
            self.assertGreater(size, 0)
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_file_not_found(self):
        """测试文件不存在时抛出异常"""
        with self.assertRaises(FileNotFoundError):
            self.processor.convert_to_standard("/nonexistent/path/audio.wav")


class TestCheckQuality(unittest.TestCase):
    """测试音频质量检测"""

    def setUp(self):
        self.processor = AudioProcessor(min_duration=3.0, max_duration=60.0)
        self.sr = STANDARD_SAMPLE_RATE

    def test_good_quality_audio(self):
        """测试高质量音频的检测结果"""
        audio = generate_sine_wave(duration=5.0, sample_rate=self.sr, amplitude=0.5)
        report = self.processor.check_quality(audio, self.sr)

        self.assertIsInstance(report, QualityReport)
        self.assertTrue(report.duration_ok)
        self.assertAlmostEqual(report.duration_seconds, 5.0, delta=0.1)
        self.assertGreater(report.quality_score, 0.5)

    def test_too_short_audio(self):
        """测试过短音频的检测"""
        audio = generate_sine_wave(duration=1.0, sample_rate=self.sr)
        report = self.processor.check_quality(audio, self.sr)

        self.assertFalse(report.duration_ok)
        self.assertGreater(len(report.warnings), 0)

    def test_too_long_audio(self):
        """测试过长音频的检测"""
        audio = generate_sine_wave(duration=65.0, sample_rate=self.sr)
        report = self.processor.check_quality(audio, self.sr)

        self.assertFalse(report.duration_ok)
        self.assertGreater(len(report.warnings), 0)

    def test_quality_score_range(self):
        """测试质量评分始终在 [0.0, 1.0] 范围内"""
        for duration in [1.0, 5.0, 15.0, 65.0]:
            audio = generate_noisy_sine(duration=duration, sample_rate=self.sr)
            report = self.processor.check_quality(audio, self.sr)
            self.assertGreaterEqual(report.quality_score, 0.0)
            self.assertLessEqual(report.quality_score, 1.0)

    def test_silence_detection(self):
        """测试长静音检测"""
        # 构造一段有长静音的音频（2秒信号 + 3秒静音 + 2秒信号）
        signal = generate_sine_wave(duration=2.0, sample_rate=self.sr, amplitude=0.5)
        silence = np.zeros(int(3.0 * self.sr), dtype=np.float32)
        audio = np.concatenate([signal, silence, signal])

        report = self.processor.check_quality(audio, self.sr)
        self.assertTrue(report.has_long_silence)


class TestDenoiseSimple(unittest.TestCase):
    """测试简单降噪功能"""

    def setUp(self):
        self.processor = AudioProcessor()
        self.sr = STANDARD_SAMPLE_RATE

    def test_denoise_preserves_shape(self):
        """测试降噪不改变音频形状"""
        audio = generate_noisy_sine(duration=5.0, sample_rate=self.sr)
        denoised = self.processor.denoise_simple(audio, self.sr)

        self.assertEqual(audio.shape, denoised.shape)
        self.assertEqual(denoised.dtype, np.float32)

    def test_denoise_reduces_low_freq(self):
        """测试降噪减弱了低频成分"""
        # 生成纯低频信号（30Hz，低于 80Hz 截止频率）
        t = np.linspace(0, 1.0, self.sr, endpoint=False)
        low_freq = (0.5 * np.sin(2 * np.pi * 30 * t)).astype(np.float32)
        denoised = self.processor.denoise_simple(low_freq, self.sr)

        # 低频信号应该被显著衰减
        original_rms = np.sqrt(np.mean(low_freq ** 2))
        denoised_rms = np.sqrt(np.mean(denoised ** 2))
        self.assertLess(denoised_rms, original_rms * 0.5)


class TestSelectBestSegment(unittest.TestCase):
    """测试最优片段选取"""

    def setUp(self):
        self.processor = AudioProcessor()
        self.sr = STANDARD_SAMPLE_RATE

    def test_short_audio_returns_full(self):
        """短音频（不足目标时长）直接返回全段"""
        audio = generate_sine_wave(duration=5.0, sample_rate=self.sr)
        segment, start_time, snr = self.processor.select_best_segment(
            audio, self.sr, target_duration=10.0
        )

        self.assertEqual(len(segment), len(audio))
        self.assertEqual(start_time, 0.0)

    def test_long_audio_returns_segment(self):
        """长音频返回最优片段"""
        # 前10秒低质量（低振幅），后10秒高质量（高振幅）
        low_quality = generate_sine_wave(duration=10.0, sample_rate=self.sr, amplitude=0.05)
        high_quality = generate_sine_wave(duration=10.0, sample_rate=self.sr, amplitude=0.8)
        audio = np.concatenate([low_quality, high_quality])

        segment, start_time, snr = self.processor.select_best_segment(
            audio, self.sr, target_duration=5.0
        )

        self.assertEqual(len(segment), int(5.0 * self.sr))
        # 最优片段应该来自后半段（高质量部分）
        self.assertGreater(start_time, 5.0)


class TestProcessBatch(unittest.TestCase):
    """测试批量处理"""

    def setUp(self):
        self.processor = AudioProcessor()
        self.sr = STANDARD_SAMPLE_RATE

    def test_batch_process_success(self):
        """测试批量处理成功场景"""
        import soundfile as sf

        # 创建临时音频文件
        temp_files = []
        for i in range(3):
            audio = generate_sine_wave(duration=5.0 + i, sample_rate=self.sr)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, self.sr)
                temp_files.append(f.name)

        try:
            processed, report = self.processor.process_batch(temp_files)

            self.assertIsInstance(report, BatchProcessReport)
            self.assertEqual(report.total_files, 3)
            self.assertEqual(report.success_count, 3)
            self.assertEqual(report.failed_count, 0)
            self.assertGreater(report.total_duration_seconds, 0)
        finally:
            for fp in temp_files:
                if os.path.exists(fp):
                    os.unlink(fp)

    def test_batch_process_with_invalid_file(self):
        """测试批量处理包含无效文件时不崩溃"""
        import soundfile as sf

        # 一个有效文件 + 一个不存在的文件
        audio = generate_sine_wave(duration=5.0, sample_rate=self.sr)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, self.sr)
            valid_path = f.name

        try:
            paths = [valid_path, "/nonexistent/file.wav"]
            processed, report = self.processor.process_batch(paths)

            self.assertEqual(report.total_files, 2)
            self.assertEqual(report.success_count, 1)
            self.assertEqual(report.failed_count, 1)
        finally:
            if os.path.exists(valid_path):
                os.unlink(valid_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
