"""Tests unitaires ciblés du cœur RAG."""

import os
import sys
import tempfile
import unittest
from unittest import mock

BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import config
import rag


class PromptGeneralisteTests(unittest.TestCase):

    def test_prompt_ne_suppose_aucune_discipline(self):
        prompt = config.SYSTEM_PROMPT.lower()

        self.assertIn("assistant pédagogique généraliste", prompt)
        self.assertIn("sans supposer à l'avance sa discipline", prompt)
        for terme_specialise in ("archéologie", "fouille", "villa", "antique"):
            self.assertNotIn(terme_specialise, prompt)

    def test_prompt_conserve_les_garde_fous_rag(self):
        prompt = config.SYSTEM_PROMPT

        self.assertIn("[Extrait N]", prompt)
        self.assertIn("N'utilise aucune connaissance extérieure", prompt)
        self.assertIn(
            "Cette information n'est pas présente dans les ressources du cours.",
            prompt,
        )


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

    def test_salutation_et_identite_restent_locales(self):
        resultat = rag.repondre_message_courant("Bonjour, qui es-tu ?")

        self.assertIsNotNone(resultat)
        self.assertEqual(resultat["tokens"], 0)
        self.assertEqual(resultat["sources"], [])

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

    def test_extrait_les_entites_d_une_comparaison(self):
        entites = rag._extraire_entites_question(
            "Compare la villa de Goiffieux et la Casa di Sallustio."
        )

        self.assertEqual(entites, ["Goiffieux", "Casa di Sallustio"])
        self.assertEqual(
            rag._extraire_entites_question("Compare Goiffieux et Casa di Sallustio."),
            ["Goiffieux", "Casa di Sallustio"],
        )

    def test_diversification_reserve_des_passages_aux_deux_entites(self):
        chunks = [
            {"source": "global.pdf", "texte": "architecture générale"},
            {"source": "goiffieux.pdf", "texte": "Goiffieux localisation"},
            {"source": "goiffieux.pdf", "texte": "Goiffieux période"},
            {"source": "sallustio.pdf", "texte": "Casa di Sallustio localisation"},
            {"source": "sallustio.pdf", "texte": "Casa di Sallustio fonction"},
        ]

        resultat = rag._diversifier_par_entites(
            chunks,
            "Compare Goiffieux et Casa di Sallustio.",
            [0, 1, 2, 3, 4],
            5,
        )

        sources = {chunks[idx]["source"] for idx in resultat}
        self.assertIn("goiffieux.pdf", sources)
        self.assertIn("sallustio.pdf", sources)
        self.assertNotIn(0, resultat[:4])

    def test_filtre_entite_accepte_le_nom_dans_la_source(self):
        chunk = {
            "source": "3. M. Liciberto, Casa Sallustio.pdf",
            "texte": "Le vestibule ouvre sur les fauces.",
        }

        self.assertTrue(rag._chunk_concerne_entite(chunk, "Casa di Sallustio"))
        self.assertFalse(rag._chunk_concerne_entite(chunk, "Goiffieux"))

    def test_une_entite_est_prioritaire_sans_vocabulaire_metier(self):
        chunks = [
            {"source": "autre.pdf", "texte": "Résultat général très pertinent"},
            {"source": "alpha.pdf", "texte": "Alpha résultat spécifique"},
            {"source": "alpha.pdf", "texte": "Alpha autre information"},
        ]

        resultat = rag._diversifier_par_entites(
            chunks,
            "Quelles preuves concernent Alpha ?",
            [0, 1, 2],
            3,
        )

        self.assertEqual(resultat[:2], [1, 2])

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

    def test_passages_cites_ne_garde_que_les_extraits_references(self):
        passages = [
            {"source": "a.pdf", "page": 1, "texte": "A"},
            {"source": "b.pdf", "page": 2, "texte": "B"},
            {"source": "c.pdf", "page": 3, "texte": "C"},
        ]

        resultat = rag._passages_cites(
            "La réponse vient de [Extrait 2] puis de [Extrait 1].",
            passages,
        )

        self.assertEqual([p["source"] for p in resultat], ["a.pdf", "b.pdf"])

    def test_passages_cites_conserve_tout_sans_reference_valide(self):
        passages = [
            {"source": "a.pdf", "page": 1, "texte": "A"},
            {"source": "b.pdf", "page": 2, "texte": "B"},
        ]

        self.assertEqual(rag._passages_cites("Réponse sans référence.", passages), passages)

    def test_nettoyage_supprime_une_pseudo_preuve_auto_invalidee(self):
        reponse = (
            "1. Mesure directe vérifiée [Extrait 1].\n\n"
            "2. Objet associé, mais son lien direct n'est pas établi [Extrait 2].\n\n"
            "3. Élément suggérant seulement le phénomène [Extrait 3].\n\n"
            "Note : la couleur n'est pas considérée comme une preuve [Extrait 4].\n\n"
            "Remarque : ces objets ne sont pas présentés comme des preuves directes.\n\n"
            "Aucune autre preuve n'est mentionnée."
        )

        resultat = rag._nettoyer_reponse(
            reponse,
            "Quelles preuves confirment cette affirmation ?",
        )

        self.assertIn("Mesure directe", resultat)
        self.assertNotIn("Objet associé", resultat)
        self.assertNotIn("suggérant", resultat)
        self.assertNotIn("couleur", resultat)
        self.assertNotIn("ne sont pas présentés", resultat)
        self.assertNotIn("Aucune autre preuve", resultat)

    def test_nettoyage_retire_le_markdown_simple(self):
        resultat = rag._nettoyer_reponse(
            "**Fonction** : réponse [Extrait 1].",
            "Quelle est sa fonction ?",
        )

        self.assertEqual(resultat, "Fonction : réponse [Extrait 1].")

    def test_comparaison_supprime_une_puce_factuelle_sans_source(self):
        reponse = (
            "Alpha\n"
            "- Fonction documentée [Extrait 1].\n\n"
            "Comparaison :\n"
            "- Alpha est plus ancien que Beta."
        )

        resultat = rag._nettoyer_reponse(
            reponse,
            "Compare Alpha et Beta selon leurs fonctions.",
        )

        self.assertIn("Fonction documentée [Extrait 1]", resultat)
        self.assertNotIn("plus ancien", resultat)
        self.assertFalse(resultat.endswith("Comparaison :"))

    def test_nombre_absent_de_la_source_supprime_la_ligne(self):
        passages = [
            {
                "source": "cours.pdf",
                "page": 1,
                "texte": "Le document indique une origine au IIe siècle.",
            }
        ]
        reponse = (
            "Origine au IIe siècle [Extrait 1].\n"
            "Destruction en 79 ap. J.-C. [Extrait 1]."
        )

        resultat = rag._retirer_nombres_non_sources(reponse, passages)

        self.assertIn("Origine au IIe siècle", resultat)
        self.assertNotIn("79", resultat)

    def test_nombre_present_dans_la_source_est_conserve(self):
        passages = [
            {
                "source": "cours.pdf",
                "page": 1,
                "texte": "La valeur mesurée est de 18,5 unités.",
            }
        ]

        resultat = rag._retirer_nombres_non_sources(
            "La valeur est de 18,5 unités [Extrait 1].",
            passages,
        )

        self.assertIn("18,5", resultat)

    def test_proposition_valide_reste_si_une_date_voisine_est_inventee(self):
        passages = [
            {
                "source": "cours.pdf",
                "page": 1,
                "texte": "Le noyau primitif remonte au moins au IIe siècle av. J.-C.",
            }
        ]
        reponse = (
            "- Le noyau remonte au moins au IIe siècle av. J.-C., "
            "avec une utilisation jusqu'en 79 ap. J.-C. [Extrait 1]."
        )

        resultat = rag._retirer_nombres_non_sources(reponse, passages)

        self.assertIn("IIe siècle", resultat)
        self.assertNotIn("79", resultat)
        self.assertIn("[Extrait 1]", resultat)

    def test_instruction_preuves_exclut_le_contexte_seulement_compatible(self):
        instruction = rag._instructions_question(
            "Quelles preuves confirment cette affirmation ?"
        )

        self.assertIn("preuve directe", instruction)
        self.assertIn("corrélé", instruction)
        self.assertIn("maximum cinq éléments", instruction)
        self.assertIn("Ne transforme pas un objet associé", instruction)

    def test_instruction_comparaison_interdit_les_connaissances_externes(self):
        instruction = rag._instructions_question(
            "Compare Goiffieux et la Casa di Sallustio selon leurs fonctions et périodes."
        )

        self.assertIn("mêmes critères", instruction)
        self.assertIn("connaissances générales", instruction)
        self.assertIn("fonction, période", instruction)
        self.assertIn("250 mots maximum", instruction)
        self.assertIn("sans tableau", instruction)

    def test_axes_question_ne_retient_que_les_axes_demandes(self):
        axes = rag._axes_question(
            "Compare Alpha et Beta selon leurs localisations, fonctions et périodes."
        )

        self.assertEqual(axes, ["localisation", "fonction", "période"])

    def test_verification_activee_pour_preuves_et_comparaisons(self):
        self.assertTrue(
            rag._doit_verifier_reponse("Quelles preuves confirment cette idée ?")
        )
        self.assertTrue(rag._doit_verifier_reponse("Compare Alpha et Beta."))
        self.assertFalse(rag._doit_verifier_reponse("Définis ce concept."))

    def test_detection_generale_des_questions_de_preuves(self):
        self.assertTrue(rag._est_question_preuves("Quelles preuves sont données ?"))
        self.assertTrue(rag._est_question_preuves("Donne deux indices."))
        self.assertFalse(rag._est_question_preuves("Résume le chapitre."))

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

    def test_ocr_est_tente_sur_une_page_sans_texte(self):
        class FaussePage:
            def get_text(self, textpage=None):
                return "" if textpage is None else "Texte reconnu par OCR."

            def get_textpage_ocr(self, **options):
                self.options = options
                return object()

        page = FaussePage()
        faux_pdf = mock.MagicMock()
        faux_pdf.__enter__.return_value = [page]

        with (
            mock.patch.object(rag.fitz, "open", return_value=faux_pdf),
            mock.patch.object(rag.config, "OCR_ENABLED", True),
            mock.patch.object(rag.config, "OCR_LANGUAGE", "fra+eng"),
            mock.patch.object(rag.config, "OCR_DPI", 200),
        ):
            statistiques = {}
            segments = rag.extraire_texte_pdf("scan.pdf", "scan.pdf", statistiques)

        self.assertEqual(segments[0]["texte"], "Texte reconnu par OCR.")
        self.assertEqual(statistiques["pages_ocr"], 1)
        self.assertEqual(page.options["language"], "fra+eng")
        self.assertEqual(page.options["dpi"], 200)

    def test_absence_de_tesseract_ne_bloque_pas_l_extraction(self):
        class FaussePage:
            def get_text(self, textpage=None):
                return ""

            def get_textpage_ocr(self, **options):
                raise RuntimeError("Tesseract indisponible")

        faux_pdf = mock.MagicMock()
        faux_pdf.__enter__.return_value = [FaussePage()]

        with (
            mock.patch.object(rag.fitz, "open", return_value=faux_pdf),
            mock.patch.object(rag.config, "OCR_ENABLED", True),
        ):
            segments = rag.extraire_texte_pdf("scan.pdf", "scan.pdf")

        self.assertEqual(segments, [])

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

    def test_reindexation_reutilise_le_cache_d_un_pdf_inchange(self):
        class ProviderLocal:
            def __init__(self):
                self.appels = 0

            def embed(self, textes):
                self.appels += 1
                return [[float(i + 1), 1.0] for i, _ in enumerate(textes)], 0

            def cache_key(self):
                return "provider-test:v1"

        with tempfile.TemporaryDirectory() as dossier:
            pdf_texte = os.path.join(dossier, "texte.pdf")

            document = rag.fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "Un document de cours exploitable.")
            document.save(pdf_texte)
            document.close()

            provider = ProviderLocal()

            with mock.patch.object(rag.config, "DATA_DIR", dossier):
                premier = rag.construire_index("8", [(pdf_texte, "texte.pdf")], provider)
                deuxieme = rag.construire_index("8", [(pdf_texte, "texte.pdf")], provider)

        self.assertEqual(premier["fichiers_reindexes"], 1)
        self.assertEqual(deuxieme["fichiers_reutilises"], 1)
        self.assertEqual(provider.appels, 1)
        self.assertEqual(deuxieme["chunks"], premier["chunks"])


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
        self.assertIn("[Extrait N]", provider.message)

    def test_repondre_n_expose_que_les_passages_cites(self):
        class ProviderLocal:
            def chat(self, system, message):
                return "Information vérifiée [Extrait 2].", 5

        passages = [
            {"source": "a.pdf", "page": 1, "texte": "Texte A"},
            {"source": "b.pdf", "page": 2, "texte": "Texte B"},
        ]

        with (
            mock.patch.object(rag, "charger_index", return_value=("index", passages)),
            mock.patch.object(rag, "rechercher", return_value=passages),
        ):
            resultat = rag.repondre(
                "3",
                "Quelle information est vérifiée ?",
                10,
                ProviderLocal(),
            )

        self.assertEqual(resultat["sources"], ["b.pdf (p.2)"])
        self.assertEqual(len(resultat["passages"]), 1)
        self.assertEqual(resultat["passages"][0]["source"], "b.pdf")

    def test_repondre_fait_verifier_une_question_de_preuves(self):
        class ProviderLocal:
            def __init__(self):
                self.appels = []

            def chat(self, system, message):
                self.appels.append(message)
                if len(self.appels) == 1:
                    return "Élément seulement compatible [Extrait 1].", 5
                return "Preuve directe corrigée [Extrait 2].", 7

        passages = [
            {"source": "a.pdf", "page": 1, "texte": "Contexte compatible"},
            {"source": "b.pdf", "page": 2, "texte": "Preuve directe"},
        ]
        provider = ProviderLocal()

        with (
            mock.patch.object(rag, "charger_index", return_value=("index", passages)),
            mock.patch.object(rag, "rechercher", return_value=passages),
        ):
            resultat = rag.repondre(
                "3",
                "Quelles preuves confirment cette affirmation ?",
                10,
                provider,
            )

        self.assertEqual(len(provider.appels), 2)
        self.assertIn("vérificateur factuel", provider.appels[1])
        self.assertIn("à partir de zéro", provider.appels[1])
        self.assertIn("SUPPRIME entièrement", provider.appels[1])
        self.assertIn("tout nombre ou toute date", provider.appels[1])
        self.assertEqual(resultat["reponse"], "Preuve directe corrigée [Extrait 2].")
        self.assertEqual(resultat["sources"], ["b.pdf (p.2)"])
        self.assertEqual(resultat["tokens"], 12)


if __name__ == "__main__":
    unittest.main()
