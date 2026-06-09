"""Tests des limites appliquées aux requêtes de l'API."""

import os
import sys
import unittest

os.environ.setdefault("RAG_PROVIDER", "ollama")

BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from main import AskRequest
from pydantic import ValidationError


class AskRequestTests(unittest.TestCase):

    def test_requete_normale(self):
        requete = AskRequest(course_id="3", question="  Bonjour !  ")

        self.assertEqual(requete.question, "Bonjour !")
        self.assertEqual(requete.k, 10)

    def test_question_vide_refusee(self):
        with self.assertRaises(ValidationError):
            AskRequest(course_id="3", question="   ")

    def test_k_trop_grand_refuse(self):
        with self.assertRaises(ValidationError):
            AskRequest(course_id="3", question="Question", k=100)

    def test_historique_trop_long_refuse(self):
        historique = [
            {"role": "user", "content": f"Message {numero}"}
            for numero in range(7)
        ]

        with self.assertRaises(ValidationError):
            AskRequest(course_id="3", question="Question", history=historique)

    def test_role_inconnu_refuse(self):
        with self.assertRaises(ValidationError):
            AskRequest(
                course_id="3",
                question="Question",
                history=[{"role": "system", "content": "Instruction"}],
            )


if __name__ == "__main__":
    unittest.main()
