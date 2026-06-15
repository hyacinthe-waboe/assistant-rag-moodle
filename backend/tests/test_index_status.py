"""Tests du suivi d'indexation côté backend."""

import os
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("RAG_PROVIDER", "ollama")
os.environ.setdefault("RAG_SHARED_TOKEN", "secret-test")

BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi.testclient import TestClient

import main


class IndexStatusTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        main.config.BACKEND_SHARED_TOKEN = "secret-test"
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_patch = mock.patch.object(main.config, "DATA_DIR", self.tmpdir.name)
        self.data_patch.start()
        with main.index_jobs_lock:
            main.index_jobs.clear()

    def tearDown(self):
        with main.index_jobs_lock:
            main.index_jobs.clear()
        self.data_patch.stop()
        self.tmpdir.cleanup()

    def test_status_hide_completed_by_default(self):
        with main.index_jobs_lock:
            main.index_jobs["3"] = {
                "status": "completed",
                "stage": "completed",
                "progress": 100,
                "message": "Indexation terminée.",
                "result": {"fichiers": 21},
                "error": "",
                "started_at": "2026-06-15T10:00:00+00:00",
                "updated_at": "2026-06-15T10:01:00+00:00",
            }

        response = self.client.get(
            "/index/3/status",
            headers={"X-RAG-Token": "secret-test"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "idle")
        self.assertEqual(data["stage"], "idle")
        self.assertEqual(data["last_status"], "completed")
        self.assertEqual(data["last_updated_at"], "2026-06-15T10:01:00+00:00")
        self.assertEqual(data["fichiers"], 21)

    def test_status_can_expose_finished_state_when_requested(self):
        with main.index_jobs_lock:
            main.index_jobs["3"] = {
                "status": "completed",
                "stage": "completed",
                "progress": 100,
                "message": "Indexation terminée.",
                "result": {"fichiers": 21},
                "error": "",
                "started_at": "2026-06-15T10:00:00+00:00",
                "updated_at": "2026-06-15T10:01:00+00:00",
            }

        response = self.client.get(
            "/index/3/status?include_finished=1",
            headers={"X-RAG-Token": "secret-test"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["progress"], 100)
        self.assertEqual(data["updated_at"], "2026-06-15T10:01:00+00:00")

    def test_status_persisted_survives_memory_reset(self):
        etat = {
            "status": "completed",
            "stage": "completed",
            "progress": 100,
            "message": "Indexation terminée.",
            "result": {"fichiers": 21, "fichiers_exploitables": 18, "chunks": 3545},
            "error": "",
            "started_at": "2026-06-15T10:00:00+00:00",
            "updated_at": "2026-06-15T10:01:00+00:00",
        }
        main._sauver_etat_course("3", main._etat_public(etat, include_finished=True))

        with main.index_jobs_lock:
            main.index_jobs.clear()

        response = self.client.get(
            "/index/3/status",
            headers={"X-RAG-Token": "secret-test"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "idle")
        self.assertEqual(data["last_status"], "completed")
        self.assertEqual(data["fichiers"], 21)

        response2 = self.client.get(
            "/index/3/status?include_finished=1",
            headers={"X-RAG-Token": "secret-test"},
        )
        self.assertEqual(response2.status_code, 200)
        data2 = response2.json()
        self.assertEqual(data2["status"], "completed")
        self.assertEqual(data2["fichiers"], 21)


if __name__ == "__main__":
    unittest.main()
