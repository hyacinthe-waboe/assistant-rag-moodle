# -*- coding: utf-8 -*-
"""
Fournisseurs IA du backend RAG.

Deux providers disponibles, même interface embed() / chat() :
  - ILAASProvider  : embeddings locaux + génération via ILAAS UT2J  ← défaut
  - OllamaProvider : tout en local, zéro donnée sortante            ← option serveur
"""

import requests
import config


class ILAASProvider:
    """Provider principal — embeddings locaux + génération ILAAS.

    Souveraineté :
    - embed()  : sentence-transformers tourne en local, aucune donnée ne sort.
    - chat()   : la question et les extraits partent vers llm.ilaas.fr (réseau UT2J).
    """

    def __init__(self):
        if not config.ILAAS_API_KEY:
            raise RuntimeError(
                "Clé ILAAS manquante. "
                "Définissez : $env:ILAAS_API_KEY='votre_cle'  (PowerShell) "
                "ou  export ILAAS_API_KEY='votre_cle'  (bash)"
            )
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(config.ILAAS_EMBED_MODEL)
        self._base  = config.ILAAS_BASE_URL.rstrip("/")

    def embed(self, textes: list[str]) -> tuple[list[list[float]], int]:
        """Vectorisation locale — aucune donnée ne sort."""
        vecteurs = self._model.encode(textes, convert_to_numpy=True).tolist()
        return vecteurs, 0  # 0 token : calcul local, pas de quota

    def cache_key(self) -> str:
        """Identifiant stable du couple provider/modèle d'embeddings."""
        return f"ilaas:{config.ILAAS_EMBED_MODEL}"

    def chat(self, system: str, user: str, temperature: float = 0.1) -> tuple[str, int]:
        """Génération via endpoint ILAAS (compatible OpenAI)."""
        r = requests.post(
            f"{self._base}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.ILAAS_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.ILAAS_CHAT,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "temperature": temperature,
            },
            timeout=120,
        )
        r.raise_for_status()
        data   = r.json()
        texte  = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return texte, tokens


class OllamaProvider:
    """Provider 100% local via Ollama — zéro donnée sortante.

    À activer avec : RAG_PROVIDER=ollama
    Nécessite Ollama installé sur le serveur avec les modèles configurés.
    """

    def __init__(self):
        self._base = config.OLLAMA_BASE_URL.rstrip("/")

    def embed(self, textes: list[str]) -> tuple[list[list[float]], int]:
        vecteurs = []
        for texte in textes:
            r = requests.post(
                f"{self._base}/api/embeddings",
                json={"model": config.OLLAMA_EMBED, "prompt": texte},
                timeout=120,
            )
            r.raise_for_status()
            vecteurs.append(r.json()["embedding"])
        return vecteurs, 0

    def cache_key(self) -> str:
        """Identifiant stable du couple provider/modèle d'embeddings."""
        return f"ollama:{config.OLLAMA_EMBED}"

    def chat(self, system: str, user: str, temperature: float = 0.1) -> tuple[str, int]:
        r = requests.post(
            f"{self._base}/api/chat",
            json={
                "model": config.OLLAMA_CHAT,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["message"]["content"], 0


def obtenir_provider():
    """Retourne le provider configuré via RAG_PROVIDER."""
    if config.PROVIDER == "ilaas":
        return ILAASProvider()
    if config.PROVIDER == "ollama":
        return OllamaProvider()
    raise ValueError(
        f"Fournisseur inconnu : '{config.PROVIDER}'. "
        "Valeurs acceptées : ilaas | ollama"
    )
