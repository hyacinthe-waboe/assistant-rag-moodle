"""Tests de l'authentification partagée du backend."""

import os
import sys
import unittest

os.environ.setdefault("RAG_PROVIDER", "ollama")
os.environ.setdefault("RAG_SHARED_TOKEN", "secret-test")

BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi.testclient import TestClient

import config
import main


main.config.BACKEND_SHARED_TOKEN = "secret-test"
client = TestClient(main.app)


class AuthPartageTests(unittest.TestCase):

    def test_health_refuse_sans_token(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 401)

    def test_health_accepte_avec_token(self):
        response = client.get("/health", headers={"X-RAG-Token": "secret-test"})
        self.assertEqual(response.status_code, 200)

    def test_ask_refuse_sans_token(self):
        response = client.post(
            "/ask",
            json={"course_id": "3", "question": "Bonjour ?"},
        )
        self.assertEqual(response.status_code, 401)

    def test_status_accepte_avec_token(self):
        response = client.get(
            "/index/3/status",
            headers={"X-RAG-Token": "secret-test"},
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
