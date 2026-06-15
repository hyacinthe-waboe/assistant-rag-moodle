# -*- coding: utf-8 -*-
"""
Cœur RAG — logique métier indépendante de l'API web et du fournisseur d'IA.

Reprend exactement la chaîne validée dans le prototype (rag_moodle.py), mais :
  - organisée par COURS (un index FAISS distinct par identifiant de cours) ;
  - branchée sur l'abstraction `providers` (ILAAS ou Ollama).

Chaîne : PDF -> texte -> chunks -> embeddings -> FAISS -> recherche -> réponse.
"""

import os
import json
import pickle
import re
import tempfile
import hashlib
import unicodedata
from collections import defaultdict
from collections.abc import Callable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

try:
    import pymupdf as fitz
except ImportError:
    import fitz

import config

RRF_K = 60
MAX_MESSAGES_HISTORIQUE = 6
TAILLE_APERCU_PASSAGE = 400
CACHE_VERSION = 1


def _recursive_character_text_splitter(*args, **kwargs):
    """Charge le splitter seulement au moment où l'indexation en a besoin."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(*args, **kwargs)


# ---------------------------------------------------------------------------
#  Messages courants traités localement
# ---------------------------------------------------------------------------

def _normaliser_message(texte: str) -> str:
    """Normalise une phrase courte pour comparer des formulations simples."""
    texte = unicodedata.normalize("NFKD", texte.lower())
    texte = "".join(caractere for caractere in texte
                    if not unicodedata.combining(caractere))
    return " ".join(re.findall(r"[a-z0-9]+", texte))


def repondre_message_courant(question: str) -> dict | None:
    """Répond localement aux messages simples, sans recherche ni appel au LLM."""
    message = _normaliser_message(question)
    identite = {
        "qui es tu",
        "tu es qui",
        "quel est ton role",
        "que peux tu faire",
    }

    if message in {
        "bonjour",
        "salut",
        "coucou",
        "hello",
        "bonjour ca va",
        "bonjour comment ca va",
        "bonjour comment vas tu",
        "salut ca va",
    }:
        reponse = "Bonjour ! Je suis l'Assistant IA de ce cours. Comment puis-je vous aider ?"
    elif message == "bonsoir":
        reponse = "Bonsoir ! Je suis l'Assistant IA de ce cours. Comment puis-je vous aider ?"
    elif message in {"merci", "merci beaucoup"}:
        reponse = "Avec plaisir."
    elif message in {"au revoir", "a bientot"}:
        reponse = "Au revoir et à bientôt."
    elif message in identite or any(
        message == f"{salutation} {formulation}"
        for salutation in ("bonjour", "salut", "hello", "bonsoir")
        for formulation in identite
    ):
        reponse = (
            "Je suis l'Assistant IA du cours. "
            "Je réponds aux questions à partir des ressources indexées."
        )
    else:
        return None

    return {
        "reponse": reponse,
        "sources": [],
        "passages": [],
        "tokens": 0,
    }


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


def _manifest_path(course_id: str) -> str:
    return os.path.join(_dossier_cours(course_id), "manifest.json")


def _cache_path(course_id: str, empreinte: str) -> str:
    return os.path.join(_dossier_cours(course_id), f"cache_{empreinte}.pkl")


def _empreinte_fichier(chemin: str) -> str:
    hachage = hashlib.sha256()
    with open(chemin, "rb") as f:
        for morceau in iter(lambda: f.read(1024 * 1024), b""):
            hachage.update(morceau)
    return hachage.hexdigest()


def _cache_signature(provider) -> str:
    identifiant = getattr(provider, "cache_key", None)
    if callable(identifiant):
        return identifiant()
    return provider.__class__.__name__


def _charger_manifest(course_id: str) -> dict:
    chemin = _manifest_path(course_id)
    if not os.path.exists(chemin):
        return {"cache_version": CACHE_VERSION, "files": {}}
    with open(chemin, "r", encoding="utf-8") as f:
        try:
            manifest = json.load(f)
        except json.JSONDecodeError:
            return {"cache_version": CACHE_VERSION, "files": {}}
    if not isinstance(manifest, dict):
        return {"cache_version": CACHE_VERSION, "files": {}}
    manifest.setdefault("cache_version", CACHE_VERSION)
    manifest.setdefault("files", {})
    return manifest


def _sauver_manifest(course_id: str, manifest: dict) -> None:
    chemin = _manifest_path(course_id)
    dossier = os.path.dirname(chemin)
    fd, tmp = tempfile.mkstemp(dir=dossier, suffix=".manifest.tmp")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, chemin)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _charger_cache_fichier(course_id: str, empreinte: str) -> dict | None:
    chemin = _cache_path(course_id, empreinte)
    if not os.path.exists(chemin):
        return None
    with open(chemin, "rb") as f:
        return pickle.load(f)


def _sauver_cache_fichier(course_id: str, empreinte: str, donnees: dict) -> None:
    chemin = _cache_path(course_id, empreinte)
    dossier = os.path.dirname(chemin)
    fd, tmp = tempfile.mkstemp(dir=dossier, suffix=".cache.tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            pickle.dump(donnees, f)
        os.replace(tmp, chemin)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def index_existe(course_id: str) -> bool:
    f_index, f_chunks = _chemins(course_id)
    return os.path.exists(f_index) and os.path.exists(f_chunks)


# ---------------------------------------------------------------------------
#  Extraction + découpage
# ---------------------------------------------------------------------------

def extraire_texte_pdf(chemin_pdf: str, nom_source: str,
                       statistiques: dict | None = None) -> list[dict]:
    """Extrait le texte natif, puis tente un OCR sur les pages sans texte."""
    segments = []
    with fitz.open(chemin_pdf) as pdf:
        for num_page, page in enumerate(pdf, start=1):
            texte = page.get_text().strip()
            utilise_ocr = False
            if not texte and config.OCR_ENABLED:
                try:
                    textpage = page.get_textpage_ocr(
                        language=config.OCR_LANGUAGE,
                        dpi=config.OCR_DPI,
                        full=True,
                    )
                    texte = page.get_text(textpage=textpage).strip()
                    utilise_ocr = bool(texte)
                except (RuntimeError, OSError):
                    # Tesseract est une dépendance système optionnelle.
                    texte = ""
            if texte:
                segments.append({"source": nom_source, "page": num_page, "texte": texte})
                if utilise_ocr and statistiques is not None:
                    statistiques["pages_ocr"] = statistiques.get("pages_ocr", 0) + 1
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
    splitter = _recursive_character_text_splitter(
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

    manifest = _charger_manifest(course_id)
    signature_provider = _cache_signature(provider)
    cle_cache = {
        "cache_version": CACHE_VERSION,
        "provider": signature_provider,
        "taille_chunk": config.TAILLE_CHUNK,
        "recouvrement": config.RECOUVREMENT,
        "ocr_enabled": config.OCR_ENABLED,
        "ocr_language": config.OCR_LANGUAGE,
        "ocr_dpi": config.OCR_DPI,
    }

    if (
        manifest.get("cache_version") != CACHE_VERSION
        or manifest.get("provider") != signature_provider
        or manifest.get("taille_chunk") != config.TAILLE_CHUNK
        or manifest.get("recouvrement") != config.RECOUVREMENT
        or manifest.get("ocr_enabled") != config.OCR_ENABLED
        or manifest.get("ocr_language") != config.OCR_LANGUAGE
        or manifest.get("ocr_dpi") != config.OCR_DPI
    ):
        manifest = {**cle_cache, "files": {}}

    if not fichiers_pdf:
        raise ValueError("Aucun PDF fourni.")

    fichiers_exploitables = 0
    fichiers_reutilises = 0
    fichiers_reindexes = 0
    fichiers_ocr = 0
    pages_ocr = 0
    chunks_all = []
    vecteurs_all = []
    tokens_nouveaux = 0
    nouveaux_files = {}
    total_fichiers = len(fichiers_pdf)
    for numero, (chemin, nom) in enumerate(fichiers_pdf, start=1):
        empreinte = _empreinte_fichier(chemin)
        entree_cache = manifest.get("files", {}).get(empreinte)
        signaler(
            "extraction",
            5 + int(35 * numero / total_fichiers),
            f"Extraction du PDF {numero}/{total_fichiers} : {nom}",
        )
        cache_fichier = None
        if entree_cache and entree_cache.get("signature") == signature_provider:
            cache_fichier = _charger_cache_fichier(course_id, empreinte)
        if cache_fichier:
            fichiers_reutilises += 1
            fichiers_exploitables += int(cache_fichier.get("fichiers_exploitables", 0))
            fichiers_ocr += int(cache_fichier.get("fichiers_ocr", 0))
            pages_ocr += int(cache_fichier.get("pages_ocr", 0))
            chunks_cache = [
                {**chunk, "source": nom}
                for chunk in cache_fichier.get("chunks", [])
            ]
            chunks_all.extend(chunks_cache)
            vecteurs_all.extend(cache_fichier.get("vecteurs", []))
            nouveaux_files[empreinte] = {
                "signature": signature_provider,
                "nom": nom,
                "cache": os.path.basename(_cache_path(course_id, empreinte)),
            }
            continue

        fichiers_reindexes += 1
        statistiques_ocr = {}
        segments = extraire_texte_pdf(chemin, nom, statistiques_ocr)
        if statistiques_ocr.get("pages_ocr", 0):
            fichiers_ocr += 1
            pages_ocr += statistiques_ocr["pages_ocr"]
        chunks = decouper_en_chunks(segments) if segments else []
        if segments:
            fichiers_exploitables += 1
        if not chunks:
            manifest.setdefault("files", {}).pop(empreinte, None)
            continue

        vecs, tokens = provider.embed([c["texte"] for c in chunks])
        chunks_all.extend(chunks)
        vecteurs_all.extend(vecs)
        tokens_nouveaux += tokens
        _sauver_cache_fichier(course_id, empreinte, {
            "signature": signature_provider,
            "fichiers_exploitables": 1 if segments else 0,
            "fichiers_ocr": 1 if statistiques_ocr.get("pages_ocr", 0) else 0,
            "pages_ocr": statistiques_ocr.get("pages_ocr", 0),
            "chunks": chunks,
            "vecteurs": vecs,
        })
        nouveaux_files[empreinte] = {
            "signature": signature_provider,
            "nom": nom,
            "cache": os.path.basename(_cache_path(course_id, empreinte)),
        }

    signaler("decoupage", 45, "Découpage du texte en passages.")
    if not chunks_all:
        raise ValueError("Aucun texte exploitable (PDF vides ou scannés sans OCR ?).")

    signaler(
        "vectorisation",
        90,
        "Reconstruction de l'index à partir des caches des fichiers.",
    )

    matrice = np.array(vecteurs_all, dtype="float32")
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
            pickle.dump(chunks_all, f)
        os.replace(tmp_index, f_index)
        os.replace(tmp_chunks, f_chunks)
        _sauver_manifest(course_id, {
            **cle_cache,
            "files": nouveaux_files,
        })
    finally:
        for chemin_temporaire in (tmp_index, tmp_chunks):
            if os.path.exists(chemin_temporaire):
                os.remove(chemin_temporaire)

    return {
        "course_id": course_id,
        "fichiers": len(fichiers_pdf),
        "fichiers_exploitables": fichiers_exploitables,
        "fichiers_reutilises": fichiers_reutilises,
        "fichiers_reindexes": fichiers_reindexes,
        "fichiers_ocr": fichiers_ocr,
        "pages_ocr": pages_ocr,
        "chunks": len(chunks_all),
        "tokens_embedding": tokens_nouveaux,
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


def _extraire_entites_question(question: str) -> list[str]:
    """Repère les noms propres utiles à une recherche comparative."""
    motif = (
        r"\b[A-ZÀ-ÖØ-Þ][\wÀ-ÿ'-]*"
        r"(?:\s+(?:(?:de|du|des|di|d'|la|le|les)\s+)?"
        r"[A-ZÀ-ÖØ-Þ][\wÀ-ÿ'-]*)*"
    )
    ignores = {
        "Compare",
        "Comparaison",
        "Pourquoi",
        "Comment",
        "Quels",
        "Quelles",
        "Quel",
        "Quelle",
        "Donne",
        "Résume",
    }
    resultat = []
    for expression in re.findall(motif, question):
        expression = expression.strip()
        for mot_ignore in ignores:
            prefixe = f"{mot_ignore} "
            if expression.startswith(prefixe):
                expression = expression[len(prefixe):].strip()
                break
        if expression in ignores or len(expression) < 4:
            continue
        if expression not in resultat:
            resultat.append(expression)
    return resultat


def _chunk_concerne_entite(chunk: dict, entite: str) -> bool:
    """Vérifie qu'un passage ou son nom de source mentionne bien l'entité."""
    mots_liaison = {"de", "du", "des", "di", "d", "la", "le", "les"}
    termes = [
        mot
        for mot in _normaliser_message(entite).split()
        if mot not in mots_liaison
    ]
    contenu = _normaliser_message(
        f"{chunk.get('source', '')} {chunk.get('texte', '')}"
    )
    mots_contenu = set(contenu.split())
    return bool(termes) and all(terme in mots_contenu for terme in termes)


def _axes_question(question: str) -> list[str]:
    """Repère les axes explicitement demandés dans une comparaison."""
    variantes = {
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
    axes = []
    for mot in _normaliser_message(question).split():
        axe = variantes.get(mot)
        if axe and axe not in axes:
            axes.append(axe)
    return axes


def _est_question_preuves(question: str) -> bool:
    """Détecte une demande explicite de preuves ou d'indices."""
    mots = set(_normaliser_message(question).split())
    return bool(mots.intersection({"preuve", "preuves", "indice", "indices"}))


def _diversifier_par_entites(chunks: list[dict], question: str,
                             indices_globaux: list[int], k: int) -> list[int]:
    """Réserve des passages à chaque entité d'une question multi-sujets."""
    entites = _extraire_entites_question(question)
    if not entites:
        return indices_globaux[:k]

    if len(entites) == 1:
        selection = [
            idx
            for idx in indices_globaux
            if _chunk_concerne_entite(chunks[idx], entites[0])
        ]
        selection.extend(idx for idx in indices_globaux if idx not in selection)
        return selection[:k]

    quota = max(2, k // len(entites))
    axes = _axes_question(question)

    termes_axes = {
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

    selection = []
    for entite in entites:
        termes_requete = [entite]
        for axe in axes:
            termes_requete.extend(termes_axes[axe])
        requete = " ".join(termes_requete)
        classement = _classer_par_mots_cles(chunks, requete, len(chunks))
        ajoutes = 0
        for idx, _ in sorted(classement.items(), key=lambda item: item[1]):
            if idx not in selection and _chunk_concerne_entite(chunks[idx], entite):
                selection.append(idx)
                ajoutes += 1
                if ajoutes >= quota:
                    break

    for idx in indices_globaux:
        if idx not in selection:
            selection.append(idx)
        if len(selection) >= k:
            break
    return selection[:k]


def rechercher(index, chunks, provider, question: str, k: int) -> list[dict]:
    """Retourne les meilleurs passages après recherche hybride FAISS + BM25."""
    n_candidats = min(k * 5, len(chunks))
    rang_dense = _classer_par_embeddings(index, provider, question, n_candidats)
    rang_bm25 = _classer_par_mots_cles(chunks, question, n_candidats)
    scores_rrf = _fusionner_classements(rang_dense, rang_bm25)
    indices_tries = sorted(scores_rrf, key=scores_rrf.get, reverse=True)

    # Pour une demande de preuves, conserve un éventail lexical plus large centré
    # sur le sujet nommé. Cela permet au vérificateur de retrouver une preuve
    # explicite même si la recherche sémantique privilégie un passage contextuel.
    if _est_question_preuves(question):
        entites = _extraire_entites_question(question)
        classement_complet = _classer_par_mots_cles(chunks, question, len(chunks))
        indices_lexicaux = [
            idx
            for idx, _ in sorted(classement_complet.items(), key=lambda item: item[1])
        ]
        if len(entites) == 1:
            indices_lexicaux = [
                idx
                for idx in indices_lexicaux
                if _chunk_concerne_entite(chunks[idx], entites[0])
            ]
        indices_tries = indices_lexicaux[:k] + [
            idx for idx in indices_tries if idx not in indices_lexicaux[:k]
        ]

    top_k = _diversifier_par_entites(chunks, question, indices_tries, k)
    return [{**chunks[idx], "score": scores_rrf.get(idx, 0.0)} for idx in top_k]


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


def _question_pour_recherche(question: str, history: list[dict] | None) -> str:
    """Complète un suivi vague avec la dernière question de l'étudiant."""
    suivis_vagues = {
        "peux tu preciser",
        "peux tu developper",
        "peux tu expliquer davantage",
        "tu peux preciser",
        "tu peux developper",
        "et pourquoi",
        "pourquoi",
        "comment",
    }
    if _normaliser_message(question) not in suivis_vagues or not history:
        return question

    for message in reversed(history):
        contenu = message.get("content", "").strip()
        if (
            message.get("role") == "user"
            and contenu
            and not repondre_message_courant(contenu)
        ):
            return f"{contenu} {question}"

    return question


def _instructions_question(question: str) -> str:
    """Ajoute des garde-fous adaptés à la forme de la question."""
    normalisee = _normaliser_message(question)
    instructions = []

    if _est_question_preuves(question):
        instructions.append(
            "MODE PREUVES : ne présente comme preuve que les éléments que les extraits "
            "relient explicitement à l'affirmation demandée. Un élément seulement "
            "compatible, proche ou corrélé ne constitue pas automatiquement une "
            "preuve directe. Ne transforme pas un objet associé, une datation ou un "
            "contexte général en preuve. Donne uniquement une liste courte des preuves "
            "directes, avec au maximum cinq éléments. N'ajoute ni contexte historique, "
            "ni période non demandée, ni rubrique sur l'absence d'autres preuves. "
            "Avant de répondre, vérifie silencieusement que chaque élément est "
            "explicitement relié à l'affirmation dans l'extrait cité."
        )

    if normalisee.startswith("compare") or " comparaison " in f" {normalisee} ":
        axes = _axes_question(question)
        portee = (
            f"Les seuls axes autorisés sont : {', '.join(axes)}. "
            if axes
            else "Limite-toi strictement aux critères formulés dans la question. "
        )
        instructions.append(
            "MODE COMPARAISON : traite séparément chaque sujet et applique les mêmes "
            f"critères à chacun. {portee}"
            "N'ajoute aucune rubrique sur l'architecture, le mobilier, le statut social "
            "ou un autre thème non demandé. Pour chaque axe, donne au maximum deux faits "
            "par sujet, puis une différence synthétique. Ne complète pas une partie avec "
            "tes connaissances générales et ne transforme pas une absence de mention en "
            "différence. Une date de publication, de fouille ou d'étude n'est pas une "
            "période du sujet. Réponse attendue : 250 mots maximum, sans tableau."
        )

    if not instructions:
        return ""
    return "\nCONSIGNES SPÉCIFIQUES :\n" + "\n".join(instructions) + "\n"


def _doit_verifier_reponse(question: str) -> bool:
    """Active une seconde passe pour les questions les plus sujettes aux extrapolations."""
    normalisee = _normaliser_message(question)
    return bool(
        _est_question_preuves(question)
        or normalisee.startswith("compare")
        or " comparaison " in f" {normalisee} "
    )


def _verifier_reponse(provider, contexte: str, question: str,
                       reponse_initiale: str) -> tuple[str, int]:
    """Fait relire une réponse complexe et retire les affirmations non étayées."""
    message = (
        f"CONTEXTE DE RÉFÉRENCE :\n{contexte}\n\n"
        f"QUESTION : {question}\n\n"
        f"BROUILLON À NE PAS PRÉSUMER CORRECT :\n{reponse_initiale}\n\n"
        "Tu es maintenant vérificateur factuel. Réponds de nouveau à la question "
        "à partir de zéro en parcourant TOUS les extraits du contexte. Le brouillon "
        "sert uniquement à repérer des erreurs possibles : ne conserve ni ses choix "
        "de sources ni ses omissions. Réécris directement la réponse finale corrigée, "
        "sans commenter ton travail. Pour chaque affirmation : "
        "1. vérifie que l'extrait cité affirme réellement cette information ; "
        "2. supprime toute connaissance extérieure, extrapolation ou conclusion "
        "plus forte que le texte ; "
        "3. ne considère jamais un élément seulement associé, une datation ou un "
        "contexte compatible comme une preuve directe ; "
        "une preuve d'un outil, d'une conséquence, d'une production ou d'une activité "
        "voisine ne prouve pas automatiquement la proposition exacte demandée ; "
        "4. distingue les dates du sujet des dates de publication, de recherche ou "
        "d'étude ; "
        "5. conserve uniquement des références [Extrait N] existantes. "
        "6. tout nombre ou toute date doit apparaître explicitement dans l'extrait "
        "cité ; sinon, supprime cette précision. "
        "Respecte strictement la portée de la question. Pour une comparaison, garde "
        "uniquement les axes demandés et 250 mots maximum. Pour une demande de "
        "preuves, recherche les preuves les plus explicites dans TOUS les extraits "
        "et donne au maximum cinq preuves directes. Si un élément est décrit comme "
        "seulement compatible, associé, indirect ou sans lien direct établi, SUPPRIME "
        "entièrement cet élément de la réponse. Si une information n'est pas démontrée "
        "par le contexte, retire-la au lieu de la reformuler."
    )
    return provider.chat(config.SYSTEM_PROMPT, message)


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


def _passages_cites(reponse: str, passages: list[dict]) -> list[dict]:
    """Garde les extraits explicitement cités par le modèle, avec repli sûr."""
    indices = {
        int(numero) - 1
        for numero in re.findall(r"\[\s*Extrait\s+(\d+)\s*\]", reponse, flags=re.IGNORECASE)
    }
    cites = [passages[idx] for idx in sorted(indices) if 0 <= idx < len(passages)]
    return cites or passages


def _nettoyer_reponse(reponse: str, question: str) -> str:
    """Retire le markdown simple et les pseudo-preuves explicitement invalidées."""
    reponse = reponse.replace("*", "")
    if _est_question_preuves(question):
        exclusions = (
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
        blocs = re.split(r"\n\s*\n", reponse)
        conserves = []
        for bloc in blocs:
            bloc_normalise = _normaliser_message(bloc)
            mots_bloc = set(bloc_normalise.split())
            commentaire_exclusion = (
                "preuve" in mots_bloc
                or "preuves" in mots_bloc
            ) and bool(mots_bloc.intersection({"pas", "aucune", "aucun"}))
            expression_exclue = any(
                _normaliser_message(expression) in bloc_normalise
                for expression in exclusions
            )
            if not commentaire_exclusion and not expression_exclue:
                conserves.append(bloc)
        reponse = "\n\n".join(conserves).strip()

    normalisee = _normaliser_message(question)
    if normalisee.startswith("compare") or " comparaison " in f" {normalisee} ":
        lignes = [
            ligne
            for ligne in reponse.splitlines()
            if not (
                ligne.lstrip().startswith(("-", "•"))
                and not re.search(r"\[\s*Extrait\s+\d+\s*\]", ligne, re.IGNORECASE)
            )
        ]
        while lignes and re.fullmatch(
            r"\s*(comparaison|synthèse|différences?|ressemblances?)\s*:?\s*",
            lignes[-1],
            flags=re.IGNORECASE,
        ):
            lignes.pop()
        reponse = "\n".join(lignes).strip()
    return reponse


def _retirer_nombres_non_sources(reponse: str, passages: list[dict]) -> str:
    """Retire les propositions contenant un nombre absent des extraits cités."""
    lignes = []
    for ligne in reponse.splitlines():
        references = [
            int(numero) - 1
            for numero in re.findall(
                r"\[\s*Extrait\s+(\d+)\s*\]",
                ligne,
                flags=re.IGNORECASE,
            )
        ]
        references = [idx for idx in references if 0 <= idx < len(passages)]
        if not references:
            lignes.append(ligne)
            continue

        references_texte = "".join(
            f"[Extrait {idx + 1}]"
            for idx in references
        )
        sans_references = re.sub(
            r"\[\s*Extrait\s+\d+\s*\]",
            "",
            ligne,
            flags=re.IGNORECASE,
        )
        textes_sources = " ".join(passages[idx]["texte"] for idx in references)
        textes_sources = textes_sources.replace(",", ".")
        nombres_ligne = re.findall(
            r"(?<![\w])\d+(?:[.,]\d+)?(?![\w])",
            re.sub(r"^\s*\d+\s*[.)-]\s*", "", sans_references),
        )
        if all(
            nombre.replace(",", ".") in textes_sources
            for nombre in nombres_ligne
        ):
            lignes.append(ligne)
            continue

        prefixe = ""
        correspondance_prefixe = re.match(r"^(\s*(?:[-•]|\d+\s*[.)-])\s*)", sans_references)
        if correspondance_prefixe:
            prefixe = correspondance_prefixe.group(1)
            sans_references = sans_references[len(prefixe):]

        propositions = re.split(
            r";\s+|,\s+(?=(?:avec|mais|et|puis|tandis|alors|jusqu|depuis|après|avant)\b)",
            sans_references,
            flags=re.IGNORECASE,
        )
        propositions_valides = []
        for proposition in propositions:
            nombres = re.findall(
                r"(?<![\w])\d+(?:[.,]\d+)?(?![\w])",
                proposition,
            )
            if all(
                nombre.replace(",", ".") in textes_sources
                for nombre in nombres
            ):
                propositions_valides.append(proposition.strip())

        if propositions_valides:
            contenu = ", ".join(propositions_valides).rstrip(" ,")
            contenu = re.sub(r"\s+([.,;:!?])", r"\1", contenu)
            terminaison = contenu[-1] if contenu.endswith((".", "?", "!")) else ""
            contenu = contenu.rstrip(" .?!")
            lignes.append(
                f"{prefixe}{contenu} {references_texte}{terminaison}".rstrip()
            )

    return "\n".join(lignes).strip()


def repondre(course_id: str, question: str, k: int, provider,
             history: list[dict] | None = None) -> dict:
    """Pipeline complet : recherche -> prompt ancré -> génération.

    `history` : liste des derniers échanges [{role, content}, …] envoyée
    par le plugin Moodle. Injectée comme contexte textuel pour donner au
    modèle la mémoire des échanges précédents (max 3 derniers échanges).
    """
    reponse_locale = repondre_message_courant(question)
    if reponse_locale:
        return reponse_locale

    index, chunks = charger_index(course_id)
    question_recherche = _question_pour_recherche(question, history)
    passages = rechercher(index, chunks, provider, question_recherche, k)

    if not passages:
        return {"reponse": "Aucun passage pertinent trouvé dans les ressources du cours.",
                "sources": [], "passages": [], "tokens": 0}

    contexte = _construire_contexte(passages)

    message = (
        f"CONTEXTE (extraits du cours) :\n{contexte}"
        f"{_formater_historique(history)}\n"
        f"{_instructions_question(question)}"
        f"QUESTION : {question}\n\n"
        "Réponds en français uniquement à partir du CONTEXTE ci-dessus. "
        "Ajoute [Extrait N] après chaque affirmation factuelle. "
        "Réponds directement, sans introduction générique, sans conclusion ajoutée, "
        "sans tableau et sans section qui n'est pas demandée. "
        "Vérifie silencieusement chaque affirmation avant de rédiger la réponse finale. "
        "Si aucune réponse fiable ne peut être établie, dis que l'information "
        "n'est pas présente dans les ressources du cours."
    )

    reponse, tokens = provider.chat(config.SYSTEM_PROMPT, message)
    if _doit_verifier_reponse(question):
        reponse, tokens_verification = _verifier_reponse(
            provider,
            contexte,
            question,
            reponse,
        )
        tokens += tokens_verification

    reponse = _nettoyer_reponse(reponse, question)
    reponse = _retirer_nombres_non_sources(reponse, passages)
    passages_utilises = _passages_cites(reponse, passages)
    sources = sorted({
        f"{p['source']} (p.{p['page']})"
        for p in passages_utilises
    })
    return {"reponse": reponse, "sources": sources,
            "passages": _exporter_passages(passages_utilises), "tokens": tokens}
