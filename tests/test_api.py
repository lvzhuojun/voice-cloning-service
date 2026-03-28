"""
FastAPI 接口集成测试

使用 FastAPI TestClient 测试所有 API 端点的基本行为。
不依赖预训练模型（合成类接口在模型未加载时会返回相应错误，验证错误码正确性）。
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

# 设置测试环境变量（在 import app 之前）
os.environ["STORAGE_DIR"] = tempfile.mkdtemp()
os.environ["MODEL_DIR"] = os.path.join(os.environ["STORAGE_DIR"], "pretrained_models")
os.environ["LOG_LEVEL"] = "WARNING"


def make_test_voicepack(storage_dir: str) -> str:
    """
    创建一个测试用的 .voicepack 文件，返回 voice_id。

    Args:
        storage_dir: 存储根目录

    Returns:
        str: voice_id
    """
    from app.core.voicepack_manager import VoicePackManager

    manager = VoicePackManager(storage_dir=storage_dir)
    embedding = torch.randn(192)
    embedding = embedding / torch.norm(embedding)
    t = np.linspace(0, 5.0, int(22050 * 5.0), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    voice_id = manager.pack(
        embedding=embedding,
        reference_audio=audio,
        reference_sample_rate=22050,
        voice_name="测试音色",
        language="zh",
        sample_count=3,
        quality_score=0.8,
    )
    return voice_id


class TestHealthAPI(unittest.TestCase):
    """测试健康检查 API"""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_health_returns_200(self):
        """健康检查应返回 200"""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)

    def test_health_response_structure(self):
        """健康检查响应应包含必要字段"""
        response = self.client.get("/api/health")
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("voice_count", data)
        self.assertIn("uptime_seconds", data)

    def test_health_status_value(self):
        """status 字段应为合法值"""
        response = self.client.get("/api/health")
        data = response.json()
        self.assertIn(data["status"], ["ok", "degraded", "error"])


class TestVoicesAPI(unittest.TestCase):
    """测试音色管理 API"""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app, raise_server_exceptions=False)
        cls.storage_dir = os.environ["STORAGE_DIR"]
        # 创建测试音色包
        cls.voice_id = make_test_voicepack(cls.storage_dir)

    def test_list_voices_returns_200(self):
        """列出音色应返回 200"""
        response = self.client.get("/api/voices/")
        self.assertEqual(response.status_code, 200)

    def test_list_voices_structure(self):
        """列出音色响应应包含 total 和 voices 字段"""
        response = self.client.get("/api/voices/")
        data = response.json()
        self.assertIn("total", data)
        self.assertIn("voices", data)
        self.assertIsInstance(data["voices"], list)

    def test_list_voices_contains_test_voice(self):
        """列出音色应包含测试创建的音色"""
        response = self.client.get("/api/voices/")
        data = response.json()
        voice_ids = [v["voice_id"] for v in data["voices"]]
        self.assertIn(self.voice_id, voice_ids)

    def test_get_voice_returns_200(self):
        """获取已存在音色应返回 200"""
        response = self.client.get(f"/api/voices/{self.voice_id}")
        self.assertEqual(response.status_code, 200)

    def test_get_voice_fields(self):
        """音色详情应包含正确字段"""
        response = self.client.get(f"/api/voices/{self.voice_id}")
        data = response.json()
        self.assertEqual(data["voice_id"], self.voice_id)
        self.assertEqual(data["voice_name"], "测试音色")
        self.assertEqual(data["language"], "zh")

    def test_get_voice_not_found(self):
        """获取不存在音色应返回 404"""
        response = self.client.get("/api/voices/nonexistent-voice-id")
        self.assertEqual(response.status_code, 404)

    def test_download_voice_returns_file(self):
        """下载音色包应返回文件"""
        response = self.client.get(f"/api/voices/{self.voice_id}/download")
        self.assertEqual(response.status_code, 200)
        # 检查返回的是 ZIP 文件
        content = response.content
        self.assertTrue(content[:4] == b"PK\x03\x04", "应返回 ZIP 文件")

    def test_download_voice_not_found(self):
        """下载不存在音色包应返回 404"""
        response = self.client.get("/api/voices/nonexistent-id/download")
        self.assertEqual(response.status_code, 404)

    def test_delete_voice(self):
        """删除音色应成功"""
        # 创建一个专门用于删除的音色
        del_voice_id = make_test_voicepack(self.storage_dir)
        response = self.client.delete(f"/api/voices/{del_voice_id}")
        self.assertEqual(response.status_code, 204)

        # 确认已删除
        get_response = self.client.get(f"/api/voices/{del_voice_id}")
        self.assertEqual(get_response.status_code, 404)

    def test_delete_voice_not_found(self):
        """删除不存在的音色应返回 404"""
        response = self.client.delete("/api/voices/nonexistent-id")
        self.assertEqual(response.status_code, 404)

    def test_test_voice_model_not_loaded(self):
        """模型未加载时测试合成应返回 503 或 500（非 200）"""
        response = self.client.post(
            f"/api/voices/{self.voice_id}/test",
            json={"text": "测试合成", "speed": 1.0, "language": "zh"},
        )
        # 模型未加载时应返回错误码（503 服务不可用 或 500）
        self.assertIn(response.status_code, [500, 503])

    def test_test_voice_not_found(self):
        """测试不存在音色合成应返回 404"""
        response = self.client.post(
            "/api/voices/nonexistent-id/test",
            json={"text": "测试", "speed": 1.0, "language": "zh"},
        )
        self.assertEqual(response.status_code, 404)


class TestTrainAPI(unittest.TestCase):
    """测试训练 API"""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app, raise_server_exceptions=False)

    def _make_audio_file(self) -> bytes:
        """生成最简单的 WAV 文件 bytes（供上传测试使用）"""
        import struct
        import wave

        buf = io.BytesIO()
        sample_rate = 22050
        duration = 5.0
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        with wave.open(buf, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

        return buf.getvalue()

    def test_upload_train_missing_files(self):
        """上传训练：缺少文件时应返回 422"""
        response = self.client.post(
            "/api/train/from-upload",
            data={"voice_name": "测试音色", "language": "zh"},
        )
        self.assertIn(response.status_code, [400, 422])

    def test_upload_train_missing_voice_name(self):
        """上传训练：缺少音色名称时应返回 422"""
        audio_bytes = self._make_audio_file()
        response = self.client.post(
            "/api/train/from-upload",
            files=[("files", ("test.wav", io.BytesIO(audio_bytes), "audio/wav"))],
            data={"language": "zh"},
        )
        self.assertIn(response.status_code, [400, 422])

    def test_upload_train_success(self):
        """上传训练：正常请求应返回 202 和 task_id"""
        audio_bytes = self._make_audio_file()
        response = self.client.post(
            "/api/train/from-upload",
            files=[("files", ("test.wav", io.BytesIO(audio_bytes), "audio/wav"))],
            data={"voice_name": "上传测试音色", "language": "zh"},
        )
        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertIn("task_id", data)
        self.assertIn("status", data)

    def test_task_status_not_found(self):
        """查询不存在的任务应返回 404"""
        response = self.client.get("/api/train/nonexistent-task-id/status")
        self.assertEqual(response.status_code, 404)

    def test_task_status_after_create(self):
        """创建任务后应能查询状态"""
        audio_bytes = self._make_audio_file()
        create_resp = self.client.post(
            "/api/train/from-upload",
            files=[("files", ("test.wav", io.BytesIO(audio_bytes), "audio/wav"))],
            data={"voice_name": "状态查询测试", "language": "zh"},
        )
        self.assertEqual(create_resp.status_code, 202)
        task_id = create_resp.json()["task_id"]

        status_resp = self.client.get(f"/api/train/{task_id}/status")
        self.assertEqual(status_resp.status_code, 200)
        task_data = status_resp.json()
        self.assertEqual(task_data["task_id"], task_id)
        self.assertIn(task_data["status"], ["pending", "processing", "done", "failed"])

    def test_folder_train_not_found(self):
        """文件夹训练：不存在的文件夹应返回 404"""
        response = self.client.post(
            "/api/train/from-folder",
            json={
                "folder_name": "nonexistent_folder_xyz",
                "voice_name": "测试",
                "language": "zh",
            },
        )
        self.assertEqual(response.status_code, 404)

    def test_folder_train_path_traversal(self):
        """文件夹训练：禁止路径穿越攻击"""
        response = self.client.post(
            "/api/train/from-folder",
            json={
                "folder_name": "../etc",
                "voice_name": "攻击测试",
                "language": "zh",
            },
        )
        # 应返回 400（参数校验失败）或 422（pydantic 校验失败）
        self.assertIn(response.status_code, [400, 422])


class TestRootEndpoint(unittest.TestCase):
    """测试根路径"""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_root_returns_200(self):
        """根路径应返回 200"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
