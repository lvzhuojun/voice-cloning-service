"""
API integration tests

Uses FastAPI TestClient to verify all endpoint behaviors.
No pretrained models required — synthesis endpoints return expected error codes
when GPT-SoVITS is not available.
"""

import io
import json
import os
import sys
import tempfile
import unittest
import uuid
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set test env vars before importing app modules
_STORAGE_DIR = tempfile.mkdtemp()
os.environ["STORAGE_DIR"] = _STORAGE_DIR
os.environ["MODEL_DIR"] = os.path.join(_STORAGE_DIR, "pretrained_models")
os.environ["LOG_LEVEL"] = "WARNING"


def make_test_voice(storage_dir: str) -> str:
    """
    Create a minimal GPT-SoVITS voice model directory for testing.
    Files are stubs only — no real model weights.

    Returns:
        str: voice_id
    """
    voice_id = str(uuid.uuid4())
    voice_dir = os.path.join(storage_dir, "models", voice_id)
    os.makedirs(voice_dir, exist_ok=True)

    metadata = {
        "voice_id": voice_id,
        "voice_name": "test_voice",
        "created_at": "2026-01-01T00:00:00+00:00",
        "language": "zh",
        "audio_duration_seconds": 60.0,
        "training_epochs_gpt": 15,
        "training_epochs_sovits": 8,
        "gpt_model_file": f"{voice_id}_gpt.ckpt",
        "sovits_model_file": f"{voice_id}_sovits.pth",
        "base_model_version": "GPT-SoVITS v2",
    }
    with open(os.path.join(voice_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    # Stub model files (zero-byte placeholders)
    open(os.path.join(voice_dir, f"{voice_id}_gpt.ckpt"), "wb").close()
    open(os.path.join(voice_dir, f"{voice_id}_sovits.pth"), "wb").close()

    # Minimal reference.wav (1s sine wave)
    ref_path = os.path.join(voice_dir, "reference.wav")
    sample_rate = 22050
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(ref_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return voice_id


class TestHealthAPI(unittest.TestCase):
    """Health check endpoint tests"""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_health_returns_200(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)

    def test_health_response_structure(self):
        response = self.client.get("/api/health")
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("gptsovits_ready", data)
        self.assertIn("voice_count", data)
        self.assertIn("uptime_seconds", data)

    def test_health_status_value(self):
        response = self.client.get("/api/health")
        data = response.json()
        self.assertIn(data["status"], ["ok", "degraded", "error"])


class TestVoicesAPI(unittest.TestCase):
    """Voice management API tests"""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app, raise_server_exceptions=False)
        cls.storage_dir = _STORAGE_DIR
        cls.voice_id = make_test_voice(cls.storage_dir)

    def test_list_voices_returns_200(self):
        response = self.client.get("/api/voices/")
        self.assertEqual(response.status_code, 200)

    def test_list_voices_structure(self):
        response = self.client.get("/api/voices/")
        data = response.json()
        self.assertIn("total", data)
        self.assertIn("voices", data)
        self.assertIsInstance(data["voices"], list)

    def test_list_voices_contains_test_voice(self):
        response = self.client.get("/api/voices/")
        data = response.json()
        voice_ids = [v["voice_id"] for v in data["voices"]]
        self.assertIn(self.voice_id, voice_ids)

    def test_get_voice_returns_200(self):
        response = self.client.get(f"/api/voices/{self.voice_id}")
        self.assertEqual(response.status_code, 200)

    def test_get_voice_fields(self):
        response = self.client.get(f"/api/voices/{self.voice_id}")
        data = response.json()
        self.assertEqual(data["voice_id"], self.voice_id)
        self.assertEqual(data["voice_name"], "test_voice")
        self.assertEqual(data["language"], "zh")

    def test_get_voice_not_found(self):
        response = self.client.get("/api/voices/nonexistent-voice-id")
        self.assertEqual(response.status_code, 404)

    def test_download_gpt_returns_file(self):
        response = self.client.get(f"/api/voices/{self.voice_id}/download/gpt")
        self.assertEqual(response.status_code, 200)

    def test_download_sovits_returns_file(self):
        response = self.client.get(f"/api/voices/{self.voice_id}/download/sovits")
        self.assertEqual(response.status_code, 200)

    def test_download_all_returns_zip(self):
        response = self.client.get(f"/api/voices/{self.voice_id}/download/all")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content[:4] == b"PK\x03\x04", "Response should be a ZIP file")

    def test_download_not_found(self):
        response = self.client.get("/api/voices/nonexistent-id/download/all")
        self.assertEqual(response.status_code, 404)

    def test_delete_voice(self):
        del_voice_id = make_test_voice(self.storage_dir)
        response = self.client.delete(f"/api/voices/{del_voice_id}")
        self.assertEqual(response.status_code, 204)
        get_response = self.client.get(f"/api/voices/{del_voice_id}")
        self.assertEqual(get_response.status_code, 404)

    def test_delete_voice_not_found(self):
        response = self.client.delete("/api/voices/nonexistent-id")
        self.assertEqual(response.status_code, 404)

    def test_test_voice_synthesis_fails_without_runtime(self):
        """Synthesis should return 503 or 500 when GPT-SoVITS is not set up."""
        response = self.client.post(
            f"/api/voices/{self.voice_id}/test",
            json={"text": "测试合成", "speed": 1.0, "language": "zh"},
        )
        self.assertIn(response.status_code, [500, 503])

    def test_test_voice_not_found(self):
        response = self.client.post(
            "/api/voices/nonexistent-id/test",
            json={"text": "测试", "speed": 1.0, "language": "zh"},
        )
        self.assertEqual(response.status_code, 404)


class TestTrainAPI(unittest.TestCase):
    """Training API tests"""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app, raise_server_exceptions=False)

    def _make_wav_bytes(self) -> bytes:
        sample_rate = 22050
        duration = 5.0
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    def test_upload_train_missing_files(self):
        response = self.client.post(
            "/api/train/from-upload",
            data={"voice_name": "test_voice", "language": "zh"},
        )
        self.assertIn(response.status_code, [400, 422])

    def test_upload_train_missing_voice_name(self):
        audio_bytes = self._make_wav_bytes()
        response = self.client.post(
            "/api/train/from-upload",
            files=[("files", ("test.wav", io.BytesIO(audio_bytes), "audio/wav"))],
            data={"language": "zh"},
        )
        self.assertIn(response.status_code, [400, 422])

    def test_upload_train_accepted(self):
        audio_bytes = self._make_wav_bytes()
        response = self.client.post(
            "/api/train/from-upload",
            files=[("files", ("test.wav", io.BytesIO(audio_bytes), "audio/wav"))],
            data={"voice_name": "upload_test_voice", "language": "zh"},
        )
        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertIn("task_id", data)
        self.assertIn("status", data)

    def test_task_status_not_found(self):
        response = self.client.get("/api/train/nonexistent-task-id/status")
        self.assertEqual(response.status_code, 404)

    def test_task_status_after_create(self):
        audio_bytes = self._make_wav_bytes()
        create_resp = self.client.post(
            "/api/train/from-upload",
            files=[("files", ("test.wav", io.BytesIO(audio_bytes), "audio/wav"))],
            data={"voice_name": "status_test_voice", "language": "zh"},
        )
        self.assertEqual(create_resp.status_code, 202)
        task_id = create_resp.json()["task_id"]

        status_resp = self.client.get(f"/api/train/{task_id}/status")
        self.assertEqual(status_resp.status_code, 200)
        task_data = status_resp.json()
        self.assertEqual(task_data["task_id"], task_id)
        self.assertIn(task_data["status"], ["pending", "processing", "done", "failed"])

    def test_folder_train_not_found(self):
        response = self.client.post(
            "/api/train/from-folder",
            json={"folder_name": "nonexistent_folder_xyz", "voice_name": "test", "language": "zh"},
        )
        self.assertEqual(response.status_code, 404)

    def test_folder_train_path_traversal_rejected(self):
        response = self.client.post(
            "/api/train/from-folder",
            json={"folder_name": "../etc", "voice_name": "attack_test", "language": "zh"},
        )
        self.assertIn(response.status_code, [400, 422])


class TestRootEndpoint(unittest.TestCase):
    """Root path tests"""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_root_returns_200(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
