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

    def test_prompt_encadre_la_lisibilite_des_listes(self):
        prompt = config.SYSTEM_PROMPT

        self.assertIn("rubriques courtes", prompt)
        self.assertIn("sous-listes imbriquées", prompt)
        self.assertIn("synthèse utile à un étudiant", prompt)
        self.assertIn("voix naturelle", prompt)
        self.assertIn("phrase normale", prompt)
        self.assertIn("Évite le style fiche automatique", prompt)
        self.assertIn("2 à 4 paragraphes", prompt)
        self.assertIn("préfère des paragraphes à des titres", prompt)


class ReponseLocaleTests(unittest.TestCase):

    def test_salutation_repond_sans_source_ni_token(self):
        resultat = rag.repondre_message_courant("Bonjour !")

        self.assertIsNotNone(resultat)
        self.assertEqual(resultat["tokens"], 0)
        self.assertEqual(resultat["sources"], [])
        self.assertEqual(resultat["passages"], [])

    def test_identite_n_est_pas_interceptee_localement(self):
        resultat = rag.repondre_message_courant("Qui es-tu ?")

        self.assertIsNone(resultat)

    def test_salutation_et_question_ne_sont_pas_interceptees(self):
        resultat = rag.repondre_message_courant("Bonjour, qui es-tu ?")

        self.assertIsNone(resultat)

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

    def test_passages_cites_lit_les_references_groupees(self):
        passages = [
            {"source": "a.pdf", "page": 1, "texte": "A"},
            {"source": "b.pdf", "page": 2, "texte": "B"},
            {"source": "c.pdf", "page": 3, "texte": "C"},
        ]

        resultat = rag._passages_cites(
            "La réponse s'appuie sur plusieurs idées [Extrait 1, Extrait 3].",
            passages,
        )

        self.assertEqual([p["source"] for p in resultat], ["a.pdf", "c.pdf"])

    def test_passages_cites_accepte_les_references_sans_crochets(self):
        passages = [
            {"source": "a.pdf", "page": 1, "texte": "A"},
            {"source": "b.pdf", "page": 2, "texte": "B"},
            {"source": "c.pdf", "page": 3, "texte": "C"},
        ]

        resultat = rag._passages_cites(
            "La réponse s'appuie sur Extraits 1 et 3.",
            passages,
        )

        self.assertEqual([p["source"] for p in resultat], ["a.pdf", "c.pdf"])

    def test_retrait_des_marqueurs_extraits_visibles(self):
        resultat = rag._retirer_marqueurs_extraits_visibles(
            "Idée principale [Extrait 1, Extrait 3]. Autre idée [Extrait 2]. "
            "+ [Historique de la conversation]"
        )

        self.assertEqual(resultat, "Idée principale. Autre idée.")

    def test_retrait_des_marqueurs_sans_crochets(self):
        resultat = rag._retirer_marqueurs_extraits_visibles(
            "Idée principale (Extrait 1). Autre idée Extraits 2 et 3."
        )

        self.assertEqual(resultat, "Idée principale. Autre idée.")

    def test_refus_global_varie_ne_renvoie_pas_de_sources(self):
        self.assertTrue(
            rag._reponse_signale_absence_information(
                "Les ressources du cours ne permettent pas de répondre à cette question."
            )
        )

    def test_refus_partiel_long_ne_supprime_pas_les_sources(self):
        reponse = (
            "Le document permet de répondre à une partie de la question avec un exemple. "
            "Il précise plusieurs éléments utiles sur le sujet demandé. "
            "En revanche, les ressources du cours ne permettent pas de répondre "
            "à l'autre partie avec certitude."
        )

        self.assertFalse(rag._reponse_signale_absence_information(reponse))

    def test_passages_cites_conserve_tout_sans_reference_valide(self):
        passages = [
            {"source": "a.pdf", "page": 1, "texte": "A"},
            {"source": "b.pdf", "page": 2, "texte": "B"},
        ]

        self.assertEqual(rag._passages_cites("Réponse sans référence.", passages), passages)

    def test_diversification_par_sources_evite_un_seul_document(self):
        chunks = [
            {"source": "a.pdf", "texte": "A1"},
            {"source": "a.pdf", "texte": "A2"},
            {"source": "b.pdf", "texte": "B"},
            {"source": "c.pdf", "texte": "C"},
        ]

        resultat = rag._diversifier_par_sources(chunks, [0, 1, 2, 3], 3)

        self.assertEqual(resultat, [0, 2, 3])

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

    def test_nettoyage_conserve_le_gras_markdown(self):
        resultat = rag._nettoyer_reponse(
            "**Fonction** : réponse [Extrait 1].",
            "Quelle est sa fonction ?",
        )

        self.assertEqual(resultat, "**Fonction** : réponse [Extrait 1].")

    def test_nettoyage_retire_les_asterisques_isoles(self):
        resultat = rag._nettoyer_reponse(
            "*Fonction* : réponse [Extrait 1].",
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

    def test_reformatage_rend_les_listes_plus_lisibles(self):
        reponse = (
            "1. Notions importantes\n"
            "- Première idée [Extrait 1]\n"
            "- Deuxième idée [Extrait 2]\n\n"
            "2. Autre axe\n"
            "- Point A [Extrait 3]"
        )

        resultat = rag._reformater_reponse_etudiante(
            reponse,
            "Cite moi les notions importantes",
        )

        self.assertIn("Notions importantes", resultat)
        self.assertIn("Première idée [Extrait 1]", resultat)
        self.assertIn("Deuxième idée [Extrait 2]", resultat)
        self.assertIn("\n\n**2. Autre axe**", resultat)

    def test_reformatage_aere_une_rubrique_numerotee_collee(self):
        resultat = rag._reformater_reponse_etudiante(
            "1. Les villas: des domaines polyvalents Les villas étaient des exploitations agricoles organisées.",
            "Résume de manière détaillée",
        )

        self.assertIn("**1. Les villas : des domaines polyvalents**", resultat)
        self.assertIn("\n\nLes villas étaient", resultat)

    def test_reformatage_limite_le_gras_aux_titres(self):
        resultat = rag._reformater_reponse_etudiante(
            "Ce cours explore **l'habitat romain** et ses **fonctions sociales**.\n\n"
            "**Les villas : des domaines polyvalents**\n"
            "Les villas combinent **résidence** et **production agricole**.",
            "Résume tout le cours de manière détaillée",
        )

        self.assertIn("Ce cours explore l'habitat romain", resultat)
        self.assertNotIn("**l'habitat romain**", resultat)
        self.assertIn("**1. Les villas : des domaines polyvalents**", resultat)
        self.assertIn("résidence et production agricole", resultat)
        self.assertNotIn("**résidence**", resultat)

    def test_reformatage_numerote_les_titres_detaillees_non_numerotes(self):
        resultat = rag._reformater_reponse_etudiante(
            "Introduction rapide du cours.\n\n"
            "Les villas : des domaines polyvalents\n"
            "Les villas combinent résidence et production.\n\n"
            "Les domus : des espaces sociaux flexibles\n"
            "Les domus organisent la vie familiale.",
            "Résume tout le cours de manière détaillée",
        )

        self.assertIn("**1. Les villas : des domaines polyvalents**", resultat)
        self.assertIn("**2. Les domus : des espaces sociaux flexibles**", resultat)

    def test_reformatage_ne_transforme_pas_une_phrase_liste_en_titre(self):
        resultat = rag._reformater_reponse_etudiante(
            "4. Méthodes d’étude et valorisation archéologique\n"
            "5. L’analyse des villas combine plusieurs approches :\n"
            "- Fouilles stratigraphiques.\n"
            "- Analyses matérielles.",
            "Résume tout le cours de manière détaillée",
        )

        self.assertIn("**4. Méthodes d’étude et valorisation archéologique**", resultat)
        self.assertIn("5. L’analyse des villas combine plusieurs approches :", resultat)
        self.assertNotIn("**5. L’analyse des villas combine plusieurs approches", resultat)

    def test_reformatage_aere_une_rubrique_sans_deux_points(self):
        resultat = rag._reformater_reponse_etudiante(
            "3. Méthodes d’étude et limites des sources et la dichotomie production consommation est remise en cause.",
            "Résume de manière détaillée",
        )

        self.assertIn("**3. Méthodes d’étude et limites des sources**", resultat)
        self.assertIn("\n\nEt la dichotomie", resultat)

    def test_reformatage_separe_un_titre_colle_a_sa_phrase(self):
        resultat = rag._reformater_reponse_etudiante(
            "Une structure divisée en zones fonctionnelles Le plan général montre une séparation claire.",
            "Explique simplement",
        )

        self.assertIn(
            "**Une structure divisée en zones fonctionnelles**\n\nLe plan général",
            resultat,
        )

    def test_reformatage_ne_coupe_pas_apres_une_preposition(self):
        resultat = rag._reformater_reponse_etudiante(
            "La villa de Goiffieux, située près de Saint-Laurent-d'Agny, est un domaine romain.",
            "Explique simplement",
        )

        self.assertNotIn("près de\n\nSaint", resultat)
        self.assertIn("près de Saint-Laurent", resultat)

    def test_reformatage_ne_coupe_pas_apres_en(self):
        resultat = rag._reformater_reponse_etudiante(
            "Olynthe illustre parfaitement l'urbanisme planifié en Grèce antique.",
            "Explique simplement",
        )

        self.assertNotIn("en\n\nGrèce", resultat)
        self.assertIn("en Grèce antique", resultat)

    def test_reformatage_ne_coupe_pas_un_nom_propre(self):
        resultat = rag._reformater_reponse_etudiante(
            "À l'inverse des villas, les domus urbaines comme la Casa Sallustio à Pompéi révèlent une organisation flexible.",
            "Résume tout le cours",
        )

        self.assertNotIn("Casa\n\nSallustio", resultat)
        self.assertIn("Casa Sallustio", resultat)

    def test_reformatage_corrige_un_accord_courant(self):
        resultat = rag._reformater_reponse_etudiante(
            "Les occupants commerceient avec les villes voisines et les pièces étaient adaptées aux réception.",
            "Explique simplement",
        )

        self.assertEqual(
            resultat,
            "Les occupants commerçaient avec les villes voisines et les pièces étaient adaptées aux réceptions.",
        )

    def test_reformatage_retire_un_fragment_orphelin(self):
        resultat = rag._reformater_reponse_etudiante(
            "La villa est un domaine organisé autour de plusieurs espaces.\n\n"
            "avec des traces d'échanges et d'activités agricoles.\n\n"
            "Elle sert surtout à comprendre les usages domestiques.",
            "Explique simplement",
        )

        self.assertNotIn("avec des traces", resultat)
        self.assertIn("La villa est un domaine", resultat)
        self.assertIn("Elle sert surtout", resultat)

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

    def test_nombres_avec_references_groupees_sont_verifies(self):
        passages = [
            {
                "source": "a.pdf",
                "page": 1,
                "texte": "La première mesure donne 18 unités.",
            },
            {
                "source": "b.pdf",
                "page": 2,
                "texte": "La seconde mesure donne 9 unités.",
            },
        ]

        resultat = rag._retirer_nombres_non_sources(
            "Les mesures sont 18 et 9 [Extraits 1 et 2].",
            passages,
        )

        self.assertIn("18", resultat)
        self.assertIn("9", resultat)

    def test_nombre_absent_est_supprime_meme_avec_reference_groupee(self):
        passages = [
            {
                "source": "a.pdf",
                "page": 1,
                "texte": "Le document indique seulement une origine au IIe siècle.",
            },
            {
                "source": "b.pdf",
                "page": 2,
                "texte": "Le document évoque une occupation ancienne.",
            },
        ]

        resultat = rag._retirer_nombres_non_sources(
            "La destruction date de 79 [Extraits 1 et 2].",
            passages,
        )

        self.assertNotIn("79", resultat)

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

    def test_instruction_synthese_generale_reste_a_hauteur_de_cours(self):
        instruction = rag._instructions_question("Cite les grandes lignes du cours")

        self.assertIn("vue d'ensemble courte du cours", instruction)
        self.assertIn("120 à 180 mots maximum", instruction)
        self.assertIn("2 à 4 axes principaux", instruction)
        self.assertIn("N'ouvre pas une longue sous-partie", instruction)
        self.assertIn("pas de numéros d'unités stratigraphiques", instruction)

    def test_instruction_synthese_detaillee_demande_des_parties_numerotees(self):
        instruction = rag._instructions_question(
            "Résume tout le cours de manière détaillée, avec les grandes parties."
        )

        self.assertIn("MODE SYNTHESE DETAILLEE", instruction)
        self.assertIn("3 à 5 parties numérotées", instruction)
        self.assertIn("gras uniquement sur les titres", instruction)

    def test_instruction_vulgarisation_evite_les_details_techniques(self):
        instruction = rag._instructions_question(
            "Explique ce sujet sans entrer dans les détails techniques"
        )

        self.assertIn("MODE VULGARISATION", instruction)
        self.assertIn("Évite les termes techniques rares", instruction)
        self.assertIn("2 ou 3 paragraphes fluides", instruction)
        self.assertIn("mots courants", instruction)
        self.assertIn("Ne transforme pas la réponse en liste de données", instruction)

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

    def test_suivi_validation_reverifie_avec_les_sources(self):
        historique = [
            {"role": "user", "content": "Cite moi les notions importantes du cours."},
            {"role": "assistant", "content": "1. Stratigraphie\n2. Cour à péristyle"},
        ]

        class ProviderLocal:
            def __init__(self):
                self.message = ""

            def chat(self, system, message):
                self.message = message
                return "Oui, mais seulement pour les notions citees dans les extraits [Extrait 1].", 5

        passage = {
            "source": "cours.pdf",
            "page": 1,
            "texte": "Le cours presente la stratigraphie et la cour a peristyle.",
        }
        provider = ProviderLocal()

        with (
            mock.patch.object(rag, "charger_index", return_value=("index", [passage])),
            mock.patch.object(rag, "rechercher", return_value=[passage]) as rechercher,
        ):
            resultat = rag.repondre(
                "3",
                "Donc c'est ça les notions importantes du cours ?",
                10,
                provider,
                historique,
            )

        rechercher.assert_called_once_with(
            "index",
            [passage],
            provider,
            "Cite moi les notions importantes du cours. Donc c'est ça les notions importantes du cours ?",
            10,
        )
        self.assertIn("MODE SUIVI / VERIFICATION", provider.message)
        self.assertIn("Ne confirme jamais automatiquement", provider.message)
        self.assertEqual(resultat["sources"], ["cours.pdf (p.1)"])
        self.assertEqual(resultat["tokens"], 5)

    def test_suivi_de_correction_reformule_avec_les_sources(self):
        historique = [
            {"role": "user", "content": "Résume le chapitre."},
            {"role": "assistant", "content": "Voici une liste très mécanique."},
        ]

        class ProviderLocal:
            def __init__(self):
                self.message = ""

            def chat(self, system, message):
                self.message = message
                return "Le chapitre presente une idee principale reformulee simplement [Extrait 1].", 5

        passage = {
            "source": "chapitre.pdf",
            "page": 2,
            "texte": "Le chapitre presente une idee principale.",
        }
        provider = ProviderLocal()

        with (
            mock.patch.object(rag, "charger_index", return_value=("index", [passage])),
            mock.patch.object(rag, "rechercher", return_value=[passage]),
        ):
            resultat = rag.repondre(
                "3",
                "Oui mais c'est bizarre, tu peux corriger ?",
                10,
                provider,
                historique,
            )

        self.assertIn("MODE SUIVI / VERIFICATION", provider.message)
        self.assertIn("reformule seulement", provider.message)
        self.assertEqual(resultat["tokens"], 5)
        self.assertEqual(resultat["sources"], ["chapitre.pdf (p.2)"])
        self.assertIn("reformulee simplement", resultat["reponse"])

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
        self.assertIn("première phrase naturelle", provider.message)
        self.assertIn("2 à 4 paragraphes courts", provider.message)

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
        self.assertEqual(resultat["reponse"], "Information vérifiée.")

    def test_repondre_hors_sujet_ne_renvoie_pas_d_extraits(self):
        class ProviderLocal:
            def chat(self, system, message):
                return (
                    "Cette information n'est pas présente dans les ressources du cours.",
                    5,
                )

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
                "Quelle est la capitale de la France ?",
                10,
                ProviderLocal(),
            )

        self.assertEqual(resultat["sources"], [])
        self.assertEqual(resultat["passages"], [])

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
        self.assertEqual(resultat["reponse"], "Preuve directe corrigée.")
        self.assertEqual(resultat["sources"], ["b.pdf (p.2)"])
        self.assertEqual(resultat["tokens"], 12)

    def test_repondre_sans_preuve_valide_revient_a_un_refus_propre(self):
        class ProviderLocal:
            def __init__(self):
                self.appels = 0

            def chat(self, system, message):
                self.appels += 1
                return "Élément seulement compatible, mais pas une preuve directe [Extrait 1].", 5

        passages = [
            {"source": "a.pdf", "page": 1, "texte": "Contexte compatible seulement."},
        ]

        with (
            mock.patch.object(rag, "charger_index", return_value=("index", passages)),
            mock.patch.object(rag, "rechercher", return_value=passages),
        ):
            resultat = rag.repondre(
                "3",
                "Quelles preuves confirment cette affirmation ?",
                10,
                ProviderLocal(),
            )

        self.assertEqual(
            resultat["reponse"],
            "Cette information n'est pas présente dans les ressources du cours.",
        )
        self.assertEqual(resultat["sources"], [])
        self.assertEqual(resultat["passages"], [])


if __name__ == "__main__":
    unittest.main()
