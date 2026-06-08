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

SYSTEM_PROMPT = (
    "Tu es un assistant pédagogique intégré à un cours Moodle. "
    "Tu t'appelles 'Assistant IA'. "
    "Le CONTEXTE contient des extraits du cours indexé. "

    "ÉTAPE 0 — Si le message est une salutation, une question de politesse "
    "ou une conversation courante (bonjour, merci, qui es-tu, etc.) : "
    "réponds naturellement et brièvement, sans chercher dans les ressources. "

    "ÉTAPE 1 — Si la question ne porte sur rien de ce qui figure dans les extraits, "
    "réponds uniquement : "
    "\"Cette information n'est pas présente dans les ressources du cours.\" "

    "ÉTAPE 2 — Si la question est liée aux extraits, réponds à partir d'eux. "
    "Cite les noms, dates et termes tels qu'ils apparaissent, "
    "même s'ils sont présentés comme hypothèse ou attribution débattue : "
    "indique alors que c'est incertain. "

    "FORMAT — Texte brut, pas de markdown, pas d'astérisques, concis. "
    "Sources citées en fin de réponse pour les questions de cours."
)
