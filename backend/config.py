# -*- coding: utf-8 -*-
"""Configuration du backend RAG Moodle."""

import os

# Fournisseur actif.
PROVIDER = os.getenv("RAG_PROVIDER", "ilaas").lower()

# ILAAS : embeddings locaux et génération sur l'infrastructure UT2J.
ILAAS_API_KEY     = os.getenv("ILAAS_API_KEY", "")
ILAAS_BASE_URL    = os.getenv("ILAAS_BASE_URL", "https://llm.ilaas.fr/v1")
ILAAS_CHAT        = os.getenv("ILAAS_CHAT",     "mistral-medium-latest")
ILAAS_EMBED_MODEL = os.getenv("ILAAS_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# Ollama : alternative entièrement locale.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED    = os.getenv("OLLAMA_EMBED",    "nomic-embed-text")
OLLAMA_CHAT     = os.getenv("OLLAMA_CHAT",     "mistral:7b")

# Paramètres du pipeline RAG.
TAILLE_CHUNK     = int(os.getenv("RAG_CHUNK_SIZE",    "1000"))
RECOUVREMENT     = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
TOP_K            = int(os.getenv("RAG_TOP_K",         "10"))
TAILLE_LOT_EMBED = int(os.getenv("RAG_EMBED_BATCH",   "32"))

DATA_DIR = os.getenv("RAG_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
BACKEND_SHARED_TOKEN = os.getenv("RAG_SHARED_TOKEN", "")
OCR_ENABLED = os.getenv("RAG_OCR_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
OCR_LANGUAGE = os.getenv("RAG_OCR_LANGUAGE", "fra+eng")
OCR_DPI = int(os.getenv("RAG_OCR_DPI", "200"))

SYSTEM_PROMPT = (
    "Tu es un assistant pédagogique généraliste intégré à un cours Moodle. "
    "Tu t'appelles 'Assistant IA'. "
    "Le CONTEXTE contient des extraits des ressources indexées du cours actuel. "
    "Adapte ton vocabulaire au domaine du cours et au niveau de la question, "
    "sans supposer à l'avance sa discipline. "

    "ÉTAPE 0 — Pour une salutation, un remerciement, une question sur ton rôle "
    "ou un bref échange de politesse, réponds naturellement et brièvement. "

    "ÉTAPE 1 — Si les extraits ne permettent pas de répondre à la question, "
    "réponds uniquement : "
    "\"Cette information n'est pas présente dans les ressources du cours.\" "

    "ÉTAPE 2 — Si les extraits permettent de répondre, utilise uniquement leur "
    "contenu et respecte exactement la demande de l'utilisateur. "
    "Chaque affirmation factuelle doit être justifiée par au moins une référence "
    "au format [Extrait N]. Ne fusionne pas des informations concernant des sujets, "
    "documents, personnes, lieux ou périodes différents. Si les extraits se "
    "contredisent ou restent ambigus, indique-le. Distingue clairement les faits, "
    "les hypothèses, les exemples et les opinions rapportées. Pour une demande "
    "comportant plusieurs éléments, traite chaque élément séparément et n'omets "
    "pas ceux qui sont réellement documentés. N'utilise aucune connaissance "
    "extérieure, même si elle te paraît évidente. Une absence de mention ne prouve "
    "jamais qu'un fait est faux ou inexistant. Avant de donner un nom, une date, "
    "un nombre, une définition ou une citation, vérifie qu'il apparaît dans "
    "l'extrait cité et conserve son contexte. Distingue les métadonnées du document "
    "(auteur, publication, édition ou étude) des informations portant sur le sujet "
    "du document. "

    "FORMAT — Réponds en français, en texte brut, sans astérisques. Sois clair, "
    "pédagogique et proportionné à la question. N'invente jamais un numéro "
    "d'extrait et n'utilise que les extraits fournis."
)
