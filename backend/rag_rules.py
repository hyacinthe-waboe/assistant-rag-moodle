# -*- coding: utf-8 -*-
"""Regles textuelles utilisees par le moteur RAG.

Ce fichier regroupe les petits marqueurs de langage qui aident le backend a
classer une question, nettoyer une reponse ou reconnaitre un cas particulier.
La logique reste dans rag.py ; ici, on garde seulement les listes explicites.
"""

REPONSE_ABSENCE_INFORMATION = (
    "Cette information n'est pas présente dans les ressources du cours."
)

REPONSES_LOCALES = {
    "bonjour": "Bonjour ! Je suis l'Assistant IA de ce cours. Comment puis-je vous aider ?",
    "salut": "Bonjour ! Je suis l'Assistant IA de ce cours. Comment puis-je vous aider ?",
    "coucou": "Bonjour ! Je suis l'Assistant IA de ce cours. Comment puis-je vous aider ?",
    "hello": "Bonjour ! Je suis l'Assistant IA de ce cours. Comment puis-je vous aider ?",
    "bonjour ca va": "Bonjour ! Je suis l'Assistant IA de ce cours. Comment puis-je vous aider ?",
    "bonjour comment ca va": "Bonjour ! Je suis l'Assistant IA de ce cours. Comment puis-je vous aider ?",
    "bonjour comment vas tu": "Bonjour ! Je suis l'Assistant IA de ce cours. Comment puis-je vous aider ?",
    "salut ca va": "Bonjour ! Je suis l'Assistant IA de ce cours. Comment puis-je vous aider ?",
    "bonsoir": "Bonsoir ! Je suis l'Assistant IA de ce cours. Comment puis-je vous aider ?",
    "merci": "Avec plaisir.",
    "merci beaucoup": "Avec plaisir.",
    "au revoir": "Au revoir et à bientôt.",
    "a bientot": "Au revoir et à bientôt.",
}

SUIVIS_VAGUES = {
    "peux tu preciser",
    "peux tu developper",
    "peux tu expliquer davantage",
    "tu peux preciser",
    "tu peux developper",
    "et pourquoi",
    "pourquoi",
    "comment",
}

MARQUEURS_SUIVI_REPONSE = (
    "c est ca",
    "c est cela",
    "c est bien ca",
    "ca veut dire",
    "cela veut dire",
    "donc",
    "du coup",
    "en gros",
    "globalement",
    "tu confirmes",
    "tu es sur",
    "est ce que c est",
    "je vois",
    "j ai compris",
    "bizarre",
    "pas normal",
    "pas coherent",
    "pas clair",
    "corrige",
    "reformule",
    "plus simple",
    "plus clairement",
    "robotique",
)

MARQUEURS_SYNTHESE_GENERALE = (
    "grandes lignes",
    "grands axes",
    "themes principaux",
    "notions principales",
    "resume le cours",
    "resume tout",
    "resumer le cours",
    "resumer tout",
    "synthese du cours",
    "synthese globale",
    "de quoi parle le cours",
    "c est quoi ce cours",
)

MARQUEURS_SYNTHESE_DETAILLEE = (
    "detaillee",
    "detaille",
    "de maniere detaillee",
    "grandes parties",
    "exemples importants",
    "plan",
    "developpe",
)

MARQUEURS_VULGARISATION = (
    "sans entrer dans les details techniques",
    "sans details techniques",
    "pas trop technique",
    "pas de details techniques",
    "simplement",
    "simple",
    "comme si je decouvrais",
    "vulgarise",
    "vulgariser",
)

MOTS_PREUVES = {"preuve", "preuves", "indice", "indices"}

AXES_COMPARAISON = {
    "localisation": "localisation",
    "localisations": "localisation",
    "fonction": "fonction",
    "fonctions": "fonction",
    "periode": "période",
    "periodes": "période",
    "transformation": "transformation",
    "transformations": "transformation",
    "usage": "usage",
    "usages": "usage",
}

TERMES_PAR_AXE = {
    "localisation": [
        "localisation", "situé", "située", "lieu", "emplacement",
        "origine", "zone", "région",
    ],
    "fonction": [
        "fonction", "usage", "rôle", "objectif", "utilité",
        "activité", "application", "service",
    ],
    "période": [
        "période", "chronologie", "phase", "état", "siècle",
        "date", "époque", "durée", "évolution",
    ],
    "transformation": [
        "transformation", "réaménagement", "reconversion",
        "modification", "évolution", "changement",
    ],
    "usage": [
        "usage", "fonction", "utilisation", "emploi",
        "application", "pratique",
    ],
}

CONNECTEURS_FRAGMENT = (
    "avec ",
    "et ",
    "mais ",
    "dont ",
    "ou ",
    "qui ",
    "que ",
    "comme ",
    "par ailleurs ",
)

DEBUTS_PHRASE_TITRE = (
    "Le",
    "La",
    "Les",
    "L'",
    "Un",
    "Une",
    "Ce",
    "Cet",
    "Cette",
    "Ces",
    "Il",
    "Elle",
    "Ils",
    "Elles",
    "On",
    "Dans",
    "Au",
    "Aux",
    "À",
    "En",
    "Pour",
    "Cela",
    "C'est",
    "D'abord",
    "Ensuite",
    "Enfin",
    "Cependant",
    "Par exemple",
    "De plus",
)

DEBUTS_PHRASE_AERER = (
    "Le",
    "La",
    "Les",
    "L'",
    "Un",
    "Une",
    "Ce",
    "Cet",
    "Cette",
    "Ces",
    "Il",
    "Elle",
    "Ils",
    "Elles",
    "On",
    "Dans",
    "Au",
    "Aux",
    "À",
    "En",
    "Pour",
    "Cela",
    "C'est",
    "D'abord",
)

MAJUSCULES_FR = "A-ZÉÈÀÂÎÔÙÇ"

MOTS_LIAISON_A_RECOLLER = (
    r"près\s+de",
    "de",
    "du",
    "des",
    "d'",
    "en",
    "à",
    "au",
    "aux",
    "dans",
    "pour",
    "avec",
    "sans",
    "sur",
    "vers",
    "chez",
    "par",
    "entre",
    "comme",
    "et",
    "ou",
    "dont",
    "qui",
    "que",
)

CORRECTIONS_FRANCAIS = (
    (r"\bcommerceient\b", "commerçaient"),
    (r"\baux réception\b", "aux réceptions"),
    (r"\baux receptions\b", "aux réceptions"),
)

EXPRESSIONS_PREUVES_A_RETIRER = (
    "n'est pas une preuve",
    "ne constitue pas une preuve",
    "pas considéré comme une preuve",
    "pas considérée comme une preuve",
    "lien direct",
    "seulement compatible",
    "simplement compatible",
    "aucune autre preuve",
    "ne constituent pas des preuves",
    "suggérant",
    "suggère",
    "indice contextuel",
    "compatible avec",
    "couramment lié",
)

REFUS_INFORMATION_GLOBALE = (
    "cette information n est pas presente dans les ressources du cours",
    "cette information n est pas disponible dans les ressources du cours",
    "aucun passage pertinent trouve dans les ressources du cours",
    "les ressources du cours ne permettent pas de repondre",
    "les extraits ne permettent pas de repondre",
    "je ne peux pas repondre avec les ressources du cours",
    "je ne peux pas confirmer avec les ressources du cours",
    "ce point n est pas documente dans les ressources du cours",
    "information absente des ressources du cours",
)

MOTIF_MARQUEUR_EXTRAIT = (
    r"\s*(?:"
    r"\[\s*Extraits?\s+[^\]]+\]"
    r"|\(\s*Extraits?\s+[^)]+\)"
    r"|\bExtraits?\s+[\d\s,;et&+\-]+"
    r")"
)
