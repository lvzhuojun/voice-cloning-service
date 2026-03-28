"""
VoicePackManager 单元测试

测试 .voicepack 文件的打包、解包、验证和管理功能。
使用随机生成的 embedding 和音频数据，不依赖预训练模型。
"""

import io
import json
import os
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import torch

# 确保可以 import app 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.voicepack_manager import (
    VoicePackManager,
    MODEL_TYPE,
    MODEL_VERSION,
    REQUIRED_FILES,
)
from app.models.schemas import VoiceInfo


def make_random_embedding(dim: int = 192) -> torch.Tensor:
    """生成随机 embedding 向量"""
    emb = torch.randn(dim)
    return emb / torch.norm(emb)  # L2 归一化


def make_random_audio(duration: float = 5.0, sample_rate: int = 22050) -> np.ndarray:
    """生成随机音频数组（正弦波）"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestVoicePackManagerInit(unittest.TestCase):
    """测试初始化"""

    def test_init_creates_dir(self):
        """测试初始化时自动创建存储目录"""
        with tempfile.TemporaryDirectory() as tmp:
            storage = os.path.join(tmp, "test_storage")
            manager = VoicePackManager(storage_dir=storage)
            self.assertTrue(os.path.exists(manager.voicepacks_dir))


class TestPack(unittest.TestCase):
    """测试打包功能"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manager = VoicePackManager(storage_dir=self.tmp)
        self.embedding = make_random_embedding()
        self.audio = make_random_audio()
        self.sample_rate = 22050

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pack_creates_file(self):
        """测试打包生成 .voicepack 文件"""
        voice_id = self.manager.pack(
            embedding=self.embedding,
            reference_audio=self.audio,
            reference_sample_rate=self.sample_rate,
            voice_name="测试音色",
        )
        self.assertIsNotNone(voice_id)
        voicepack_path = self.manager.get_voicepack_path(voice_id)
        self.assertTrue(os.path.exists(voicepack_path))

    def test_pack_returns_uuid(self):
        """测试返回的 voice_id 是 UUID v4 格式"""
        import uuid
        voice_id = self.manager.pack(
            embedding=self.embedding,
            reference_audio=self.audio,
            reference_sample_rate=self.sample_rate,
            voice_name="测试音色",
        )
        # 验证是合法的 UUID
        try:
            uuid.UUID(voice_id, version=4)
            valid = True
        except ValueError:
            valid = False
        self.assertTrue(valid)

    def test_pack_custom_voice_id(self):
        """测试使用自定义 voice_id"""
        custom_id = "12345678-1234-4321-abcd-123456789abc"
        voice_id = self.manager.pack(
            embedding=self.embedding,
            reference_audio=self.audio,
            reference_sample_rate=self.sample_rate,
            voice_name="自定义ID音色",
            voice_id=custom_id,
        )
        self.assertEqual(voice_id, custom_id)

    def test_pack_zip_structure(self):
        """测试 ZIP 内部文件结构符合规范"""
        voice_id = self.manager.pack(
            embedding=self.embedding,
            reference_audio=self.audio,
            reference_sample_rate=self.sample_rate,
            voice_name="结构测试",
        )
        voicepack_path = self.manager.get_voicepack_path(voice_id)

        with zipfile.ZipFile(voicepack_path, "r") as zf:
            names = zf.namelist()
            root = names[0].split("/")[0]
            self.assertEqual(root, voice_id)

            inner = {n.split("/", 1)[1] for n in names if "/" in n and n.split("/", 1)[1]}
            for required in REQUIRED_FILES:
                self.assertIn(required, inner, f"缺少必须文件: {required}")

    def test_pack_metadata_content(self):
        """测试 metadata.json 内容正确"""
        voice_id = self.manager.pack(
            embedding=self.embedding,
            reference_audio=self.audio,
            reference_sample_rate=self.sample_rate,
            voice_name="元数据测试",
            language="zh",
            sample_count=3,
            total_duration_seconds=15.0,
            quality_score=0.75,
        )
        voicepack_path = self.manager.get_voicepack_path(voice_id)

        with zipfile.ZipFile(voicepack_path, "r") as zf:
            with zf.open(f"{voice_id}/metadata.json") as f:
                meta = json.load(f)

        self.assertEqual(meta["voice_id"], voice_id)
        self.assertEqual(meta["voice_name"], "元数据测试")
        self.assertEqual(meta["model_type"], MODEL_TYPE)
        self.assertEqual(meta["model_version"], MODEL_VERSION)
        self.assertEqual(meta["language"], "zh")
        self.assertEqual(meta["sample_count"], 3)
        self.assertAlmostEqual(meta["total_duration_seconds"], 15.0, places=1)
        self.assertAlmostEqual(meta["quality_score"], 0.75, places=2)

    def test_pack_invalid_embedding_shape(self):
        """测试 2D embedding 时抛出 ValueError"""
        bad_embedding = torch.randn(4, 192)
        with self.assertRaises(ValueError):
            self.manager.pack(
                embedding=bad_embedding,
                reference_audio=self.audio,
                reference_sample_rate=self.sample_rate,
                voice_name="错误测试",
            )


class TestUnpack(unittest.TestCase):
    """测试解包功能"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manager = VoicePackManager(storage_dir=self.tmp)
        self.embedding = make_random_embedding(192)
        self.audio = make_random_audio()
        self.sample_rate = 22050
        self.voice_id = self.manager.pack(
            embedding=self.embedding,
            reference_audio=self.audio,
            reference_sample_rate=self.sample_rate,
            voice_name="解包测试",
            language="zh",
            sample_count=2,
            quality_score=0.8,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unpack_returns_embedding(self):
        """测试解包返回正确的 embedding"""
        emb, _, _, meta, _ = self.manager.unpack(self.voice_id)
        self.assertIsInstance(emb, torch.Tensor)
        self.assertEqual(emb.ndim, 1)
        self.assertEqual(emb.shape[0], 192)

    def test_unpack_returns_audio(self):
        """测试解包返回参考音频"""
        _, audio, sr, _, _ = self.manager.unpack(self.voice_id)
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
        self.assertIsInstance(sr, int)
        self.assertGreater(sr, 0)

    def test_unpack_returns_metadata(self):
        """测试解包返回正确的 metadata"""
        _, _, _, meta, _ = self.manager.unpack(self.voice_id)
        self.assertEqual(meta["voice_id"], self.voice_id)
        self.assertEqual(meta["voice_name"], "解包测试")

    def test_unpack_not_found(self):
        """测试解包不存在的音色包时抛出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.manager.unpack("00000000-0000-0000-0000-000000000000")


class TestValidate(unittest.TestCase):
    """测试格式验证功能"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manager = VoicePackManager(storage_dir=self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_validate_valid_pack(self):
        """测试合法音色包验证通过"""
        voice_id = self.manager.pack(
            embedding=make_random_embedding(),
            reference_audio=make_random_audio(),
            reference_sample_rate=22050,
            voice_name="验证测试",
        )
        is_valid, errors = self.manager.validate(voice_id)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_nonexistent(self):
        """测试不存在的音色包返回验证失败"""
        is_valid, errors = self.manager.validate("nonexistent-id")
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


class TestListAndDelete(unittest.TestCase):
    """测试列表和删除功能"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manager = VoicePackManager(storage_dir=self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list_empty(self):
        """测试空目录返回空列表"""
        voices = self.manager.list_all()
        self.assertEqual(len(voices), 0)

    def test_list_after_pack(self):
        """测试打包后列表长度正确"""
        for i in range(3):
            self.manager.pack(
                embedding=make_random_embedding(),
                reference_audio=make_random_audio(),
                reference_sample_rate=22050,
                voice_name=f"音色_{i}",
            )
        voices = self.manager.list_all()
        self.assertEqual(len(voices), 3)

    def test_list_returns_voiceinfo(self):
        """测试列表返回 VoiceInfo 对象"""
        self.manager.pack(
            embedding=make_random_embedding(),
            reference_audio=make_random_audio(),
            reference_sample_rate=22050,
            voice_name="列表测试",
        )
        voices = self.manager.list_all()
        self.assertEqual(len(voices), 1)
        self.assertIsInstance(voices[0], VoiceInfo)

    def test_delete_existing(self):
        """测试删除已存在的音色包"""
        voice_id = self.manager.pack(
            embedding=make_random_embedding(),
            reference_audio=make_random_audio(),
            reference_sample_rate=22050,
            voice_name="删除测试",
        )
        result = self.manager.delete(voice_id)
        self.assertTrue(result)
        self.assertEqual(len(self.manager.list_all()), 0)

    def test_delete_nonexistent(self):
        """测试删除不存在的音色包返回 False"""
        result = self.manager.delete("nonexistent-id")
        self.assertFalse(result)


class TestGetInfo(unittest.TestCase):
    """测试获取元数据功能"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manager = VoicePackManager(storage_dir=self.tmp)
        self.voice_id = self.manager.pack(
            embedding=make_random_embedding(),
            reference_audio=make_random_audio(),
            reference_sample_rate=22050,
            voice_name="信息测试",
            language="en",
            sample_count=5,
            quality_score=0.9,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_info_fields(self):
        """测试 get_info 返回正确的字段值"""
        info = self.manager.get_info(self.voice_id)
        self.assertEqual(info.voice_id, self.voice_id)
        self.assertEqual(info.voice_name, "信息测试")
        self.assertEqual(info.language, "en")
        self.assertEqual(info.sample_count, 5)
        self.assertAlmostEqual(info.quality_score, 0.9, places=2)

    def test_get_info_file_size(self):
        """测试 get_info 返回文件大小"""
        info = self.manager.get_info(self.voice_id)
        self.assertIsNotNone(info.voicepack_size_bytes)
        self.assertGreater(info.voicepack_size_bytes, 0)

    def test_get_info_not_found(self):
        """测试获取不存在音色的信息时抛出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.manager.get_info("nonexistent-id")


if __name__ == "__main__":
    unittest.main(verbosity=2)
