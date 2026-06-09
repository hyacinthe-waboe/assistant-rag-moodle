"""Tests unitaires ciblés du cœur RAG."""

import os
import sys
import tempfile
import unittest
from unittest import mock

BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import rag


class ReponseLocaleTests(unittest.TestCase):

    def test_salutation_repond_sans_source_ni_token(self):
        resultat = rag.repondre_message_courant("Bonjour !")

        self.assertIsNotNone(resultat)
        self.assertEqual(resultat["tokens"], 0)
        self.assertEqual(resultat["sources"], [])
        self.assertEqual(resultat["passages"], [])

    def test_identite_reconnait_les_accents_et_la_ponctuation(self):
        resultat = rag.repondre_message_courant("Qui es-tu ?")

        self.assertIn("Assistant IA", resultat["reponse"])

    def test_salutation_avec_politesse_reste_locale(self):
        resultat = rag.repondre_message_courant("Bonjour, ça va ?")

        self.assertIsNotNone(resultat)
        self.assertEqual(resultat["tokens"], 0)

    def test_question_de_cours_n_est_pas_interceptee(self):
        resultat = rag.repondre_message_courant(
            "Bonjour, peux-tu expliquer la maison romaine ?"
        )

        self.assertIsNone(resultat)


class UtilitairesRagTests(unittest.TestCase):

    def test_identifiant_de_cours_numerique(self):
        self.assertEqual(rag._valider_course_id("3"), "3")

    def test_identifiant_de_cours_invalide(self):
        with self.assertRaises(ValueError):
            rag._valider_course_id("../3")

    def test_fusion_rrf_favorise_un_resultat_present_partout(self):
        scores = rag._fusionner_classements({1: 0, 2: 1}, {2: 0, 3: 1})

        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[2], scores[3])

    def test_suivi_vague_reutilise_la_derniere_question(self):
        historique = [
            {"role": "user", "content": "Qui habitait la Casa di Sallustio ?"},
            {"role": "assistant", "content": "Le propriétaire reste incertain."},
        ]

        question = rag._question_pour_recherche("Peux-tu préciser ?", historique)

        self.assertEqual(
            question,
            "Qui habitait la Casa di Sallustio ? Peux-tu préciser ?",
        )

    def test_question_explicite_reste_inchangee(self):
        historique = [
            {"role": "user", "content": "Parle-moi de Pompéi."},
        ]

        question = rag._question_pour_recherche(
            "Quelle est la date de construction de cette maison ?",
            historique,
        )

        self.assertEqual(
            question,
            "Quelle est la date de construction de cette maison ?",
        )

    def test_suivi_vague_sans_historique_reste_inchange(self):
        self.assertEqual(
            rag._question_pour_recherche("Peux-tu préciser ?", []),
            "Peux-tu préciser ?",
        )

    def test_suivi_ignore_un_message_de_politesse(self):
        historique = [
            {"role": "user", "content": "Qui habitait la Casa di Sallustio ?"},
            {"role": "assistant", "content": "Le propriétaire reste incertain."},
            {"role": "user", "content": "Merci"},
            {"role": "assistant", "content": "Avec plaisir."},
        ]

        question = rag._question_pour_recherche("Peux-tu préciser ?", historique)

        self.assertEqual(
            question,
            "Qui habitait la Casa di Sallustio ? Peux-tu préciser ?",
        )


class IndexationTests(unittest.TestCase):

    def test_compte_seulement_les_pdf_avec_du_texte(self):
        class ProviderLocal:
            def embed(self, textes):
                return [[float(i + 1), 1.0] for i, _ in enumerate(textes)], 0

        with tempfile.TemporaryDirectory() as dossier:
            pdf_texte = os.path.join(dossier, "texte.pdf")
            pdf_vide = os.path.join(dossier, "scan.pdf")

            document = rag.fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "Un document de cours exploitable.")
            document.save(pdf_texte)
            document.close()

            document = rag.fitz.open()
            document.new_page()
            document.save(pdf_vide)
            document.close()

            with mock.patch.object(rag.config, "DATA_DIR", dossier):
                resultat = rag.construire_index(
                    "7",
                    [(pdf_texte, "texte.pdf"), (pdf_vide, "scan.pdf")],
                    ProviderLocal(),
                )

        self.assertEqual(resultat["fichiers"], 2)
        self.assertEqual(resultat["fichiers_exploitables"], 1)
        self.assertGreater(resultat["chunks"], 0)


class ReponseRagTests(unittest.TestCase):

    def test_repondre_utilise_la_question_completee_pour_la_recherche(self):
        historique = [
            {"role": "user", "content": "Qui habitait la Casa di Sallustio ?"},
            {"role": "assistant", "content": "Le propriétaire reste incertain."},
        ]

        class ProviderLocal:
            def chat(self, system, message):
                self.message = message
                return "Réponse test", 5

        provider = ProviderLocal()
        passage = {"source": "cours.pdf", "page": 1, "texte": "Texte du cours"}

        with (
            mock.patch.object(rag, "charger_index", return_value=("index", [passage])),
            mock.patch.object(rag, "rechercher", return_value=[passage]) as rechercher,
        ):
            rag.repondre("3", "Peux-tu préciser ?", 10, provider, historique)

        rechercher.assert_called_once_with(
            "index",
            [passage],
            provider,
            "Qui habitait la Casa di Sallustio ? Peux-tu préciser ?",
            10,
        )
        self.assertIn("QUESTION : Peux-tu préciser ?", provider.message)


if __name__ == "__main__":
    unittest.main()
