# -*- coding: utf-8 -*-
"""
Cœur RAG — logique métier indépendante de l'API web et du fournisseur d'IA.

Reprend exactement la chaîne validée dans le prototype (rag_moodle.py), mais :
  - organisée par COURS (un index FAISS distinct par identifiant de cours) ;
  - branchée sur l'abstraction `providers` (ILAAS ou Ollama).

Chaîne : PDF -> texte -> chunks -> embeddings -> FAISS -> recherche -> réponse.
"""

import os
import pickle
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

try:
    import pymupdf as fitz
except ImportError:
    import fitz

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

import config

RRF_K = 60
MAX_MESSAGES_HISTORIQUE = 6
TAILLE_APERCU_PASSAGE = 400


# ---------------------------------------------------------------------------
#  Chemins de stockage par cours
# ---------------------------------------------------------------------------

def _valider_course_id(course_id: str) -> str:
    """Accepte uniquement les identifiants numériques envoyés par Moodle."""
    course_id = str(course_id)
    if not course_id.isdigit():
        raise ValueError("Identifiant de cours invalide.")
    return course_id


def _dossier_cours(course_id: str) -> str:
    """Renvoie (et crée) le dossier de stockage de l'index d'un cours."""
    course_id = _valider_course_id(course_id)
    dossier = os.path.join(config.DATA_DIR, course_id)
    os.makedirs(dossier, exist_ok=True)
    return dossier


def _chemins(course_id: str) -> tuple[str, str]:
    base = _dossier_cours(course_id)
    return os.path.join(base, "index_faiss.bin"), os.path.join(base, "chunks.pkl")


def index_existe(course_id: str) -> bool:
    f_index, f_chunks = _chemins(course_id)
    return os.path.exists(f_index) and os.path.exists(f_chunks)


# ---------------------------------------------------------------------------
#  Extraction + découpage
# ---------------------------------------------------------------------------

def extraire_texte_pdf(chemin_pdf: str, nom_source: str) -> list[dict]:
    """Extrait le texte d'un PDF, page par page, avec la source en métadonnée."""
    segments = []
    with fitz.open(chemin_pdf) as pdf:
        for num_page, page in enumerate(pdf, start=1):
            texte = page.get_text().strip()
            if texte:
                segments.append({"source": nom_source, "page": num_page, "texte": texte})
    return segments


def decouper_en_chunks(documents: list[dict]) -> list[dict]:
    """Découpe les pages en chunks homogènes avec recouvrement.

    Stratégie globale par source : toutes les pages d'un même PDF sont
    concaténées avant le découpage. Cela évite les micro-chunks isolés issus
    de diapositives à faible densité textuelle (ex : "longueur de 5,56 m"),
    dont les embeddings sont peu fiables et manquent systématiquement au
    retrieval. Les chunks résultants couvrent plusieurs pages et embedent
    un contexte sémantique plus riche.

    La page citée dans les sources correspond à la page de début du chunk
    (estimation par position cumulative dans le texte complet).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.TAILLE_CHUNK,
        chunk_overlap=config.RECOUVREMENT,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    par_source = defaultdict(list)
    for doc in documents:
        par_source[doc["source"]].append(doc)

    chunks = []
    for source, segments in par_source.items():
        segments_tries = sorted(segments, key=lambda d: d["page"])
        texte_complet, frontieres = _assembler_pages(segments_tries)
        chunks.extend(_decouper_source(source, texte_complet, frontieres, splitter))

    return chunks


def _assembler_pages(segments: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Concatène les pages et mémorise la position où chacune commence."""
    parties = []
    frontieres = []
    position = 0

    for segment in segments:
        frontieres.append((position, segment["page"]))
        partie = segment["texte"] + "\n\n"
        parties.append(partie)
        position += len(partie)

    return "".join(parties), frontieres


def _page_a_position(frontieres: list[tuple[int, int]], position: int) -> int:
    """Retourne la page qui contient approximativement une position du texte."""
    page = frontieres[0][1]
    for debut, numero_page in frontieres:
        if debut > position:
            break
        page = numero_page
    return page


def _decouper_source(source: str, texte: str, frontieres: list[tuple[int, int]],
                     splitter) -> list[dict]:
    """Découpe un PDF concaténé et rattache chaque chunk à sa page de début."""
    chunks = []
    position_curseur = 0

    for morceau in splitter.split_text(texte):
        position = texte.find(morceau[:80], position_curseur)
        if position == -1:
            position = position_curseur

        chunks.append({
            "source": source,
            "page": _page_a_position(frontieres, position),
            "texte": morceau,
        })
        position_curseur = max(0, position - config.RECOUVREMENT)

    return chunks


# ---------------------------------------------------------------------------
#  Construction / chargement de l'index
# ---------------------------------------------------------------------------

def construire_index(course_id: str, fichiers_pdf: list[tuple[str, str]], provider,
                     progression: Callable[[str, int, str], None] | None = None) -> dict:
    """Construit et persiste l'index FAISS d'un cours.

    `fichiers_pdf` : liste de (chemin_sur_disque, nom_affiché).
    Renvoie un résumé chiffré (nb chunks, tokens) pour le suivi quantitatif.
    """
    def signaler(etape: str, pourcentage: int, message: str) -> None:
        if progression:
            progression(etape, pourcentage, message)

    documents = []
    total_fichiers = len(fichiers_pdf)
    for numero, (chemin, nom) in enumerate(fichiers_pdf, start=1):
        signaler(
            "extraction",
            5 + int(35 * numero / total_fichiers),
            f"Extraction du PDF {numero}/{total_fichiers} : {nom}",
        )
        documents.extend(extraire_texte_pdf(chemin, nom))

    signaler("decoupage", 45, "Découpage du texte en passages.")
    chunks = decouper_en_chunks(documents)
    if not chunks:
        raise ValueError("Aucun texte exploitable (PDF vides ou scannés sans OCR ?).")

    # Vectorisation par lots, via le fournisseur choisi
    vecteurs, total_tokens = [], 0
    total_lots = max(1, (len(chunks) + config.TAILLE_LOT_EMBED - 1) // config.TAILLE_LOT_EMBED)
    for numero_lot, debut in enumerate(
        range(0, len(chunks), config.TAILLE_LOT_EMBED),
        start=1,
    ):
        signaler(
            "vectorisation",
            50 + int(45 * numero_lot / total_lots),
            f"Vectorisation des passages : lot {numero_lot}/{total_lots}.",
        )
        lot = [c["texte"] for c in chunks[debut:debut + config.TAILLE_LOT_EMBED]]
        vecs, tokens = provider.embed(lot)
        vecteurs.extend(vecs)
        total_tokens += tokens

    matrice = np.array(vecteurs, dtype="float32")
    faiss.normalize_L2(matrice)  # produit scalaire sur vecteurs normalisés = cosinus

    index = faiss.IndexFlatIP(matrice.shape[1])
    index.add(matrice)

    signaler("sauvegarde", 97, "Enregistrement du nouvel index.")
    f_index, f_chunks = _chemins(course_id)
    dossier = os.path.dirname(f_index)
    fd_chunks, tmp_chunks = tempfile.mkstemp(dir=dossier, suffix=".pkl.tmp")
    os.close(fd_chunks)
    fd_index, tmp_index = tempfile.mkstemp(dir=dossier, suffix=".faiss.tmp")
    os.close(fd_index)

    try:
        faiss.write_index(index, tmp_index)
        with open(tmp_chunks, "wb") as f:
            pickle.dump(chunks, f)
        os.replace(tmp_index, f_index)
        os.replace(tmp_chunks, f_chunks)
    finally:
        for chemin_temporaire in (tmp_index, tmp_chunks):
            if os.path.exists(chemin_temporaire):
                os.remove(chemin_temporaire)

    return {
        "course_id": course_id,
        "fichiers": len(fichiers_pdf),
        "chunks": len(chunks),
        "tokens_embedding": total_tokens,
    }


def charger_index(course_id: str):
    """Recharge l'index et les chunks d'un cours depuis le disque."""
    f_index, f_chunks = _chemins(course_id)
    index = faiss.read_index(f_index)
    with open(f_chunks, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ---------------------------------------------------------------------------
#  Recherche hybride : embeddings + mots-clés, puis fusion des classements
# ---------------------------------------------------------------------------

def _tokeniser(texte: str) -> list[str]:
    """Tokenisation simple : mots alphanumériques en minuscules (FR + chiffres)."""
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", texte.lower())


def _classer_par_embeddings(index, provider, question: str,
                            n_candidats: int) -> dict[int, int]:
    """Classe les chunks par proximité sémantique avec la question."""
    vecs, _ = provider.embed([question])
    vecteur_q = np.array(vecs, dtype="float32")
    faiss.normalize_L2(vecteur_q)

    _, indices_dense = index.search(vecteur_q, n_candidats)
    return {
        int(idx): rang
        for rang, idx in enumerate(indices_dense[0])
        if idx != -1
    }


def _classer_par_mots_cles(chunks: list[dict], question: str,
                           n_candidats: int) -> dict[int, int]:
    """Classe les chunks qui contiennent les mots importants de la question."""
    corpus = [_tokeniser(c["texte"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores_bm25 = bm25.get_scores(_tokeniser(question))

    indices_bm25 = sorted(
        range(len(scores_bm25)),
        key=lambda i: scores_bm25[i],
        reverse=True,
    )[:n_candidats]
    return {idx: rang for rang, idx in enumerate(indices_bm25)}


def _fusionner_classements(*classements: dict[int, int]) -> dict[int, float]:
    """Fusionne plusieurs classements avec Reciprocal Rank Fusion (RRF)."""
    candidats = set().union(*classements)
    return {
        idx: sum(
            1.0 / (RRF_K + classement[idx])
            for classement in classements
            if idx in classement
        )
        for idx in candidats
    }


def rechercher(index, chunks, provider, question: str, k: int) -> list[dict]:
    """Retourne les meilleurs passages après recherche hybride FAISS + BM25."""
    n_candidats = min(k * 3, len(chunks))
    rang_dense = _classer_par_embeddings(index, provider, question, n_candidats)
    rang_bm25 = _classer_par_mots_cles(chunks, question, n_candidats)
    scores_rrf = _fusionner_classements(rang_dense, rang_bm25)
    top_k = sorted(scores_rrf, key=scores_rrf.get, reverse=True)[:k]
    return [{**chunks[idx], "score": scores_rrf[idx]} for idx in top_k]


def _construire_contexte(passages: list[dict]) -> str:
    blocs = []
    for i, p in enumerate(passages, start=1):
        blocs.append(f"[Extrait {i} — source : {p['source']}, page {p['page']}]\n{p['texte']}")
    return "\n\n".join(blocs)


def _formater_historique(history: list[dict] | None) -> str:
    """Transforme les derniers échanges en texte lisible par le modèle."""
    if not history:
        return ""

    lignes = []
    for message in history[-MAX_MESSAGES_HISTORIQUE:]:
        role = "Étudiant" if message.get("role") == "user" else "Assistant"
        lignes.append(f"{role} : {message.get('content', '')}")

    return "\nHISTORIQUE DE LA CONVERSATION :\n" + "\n".join(lignes) + "\n"


def _exporter_passages(passages: list[dict]) -> list[dict]:
    """Prépare des extraits courts pour l'accordéon du plugin Moodle."""
    resultat = []
    for passage in passages:
        texte = passage["texte"]
        if len(texte) > TAILLE_APERCU_PASSAGE:
            texte = texte[:TAILLE_APERCU_PASSAGE] + "\u2026"
        resultat.append({
            "source": passage["source"],
            "page": passage["page"],
            "texte": texte,
        })
    return resultat


def repondre(course_id: str, question: str, k: int, provider,
             history: list[dict] | None = None) -> dict:
    """Pipeline complet : recherche -> prompt ancré -> génération.

    `history` : liste des derniers échanges [{role, content}, …] envoyée
    par le plugin Moodle. Injectée comme contexte textuel pour donner au
    modèle la mémoire des échanges précédents (max 3 derniers échanges).
    """
    index, chunks = charger_index(course_id)
    passages = rechercher(index, chunks, provider, question, k)

    if not passages:
        return {"reponse": "Aucun passage pertinent trouvé dans les ressources du cours.",
                "sources": [], "passages": [], "tokens": 0}

    contexte = _construire_contexte(passages)

    message = (
        f"CONTEXTE (extraits du cours) :\n{contexte}"
        f"{_formater_historique(history)}\n"
        f"QUESTION : {question}\n\n"
        f"Réponds en français à partir du CONTEXTE ci-dessus."
    )

    reponse, tokens = provider.chat(config.SYSTEM_PROMPT, message)

    sources = sorted({f"{p['source']} (p.{p['page']})" for p in passages})
    return {"reponse": reponse, "sources": sources,
            "passages": _exporter_passages(passages), "tokens": tokens}
