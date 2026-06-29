# -*- coding: utf-8 -*-
"""
Cœur RAG — logique métier indépendante de l'API web et du fournisseur d'IA.

Le moteur organise un index FAISS distinct par cours et utilise l'abstraction
`providers` pour fonctionner avec ILAAS ou Ollama.

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
import rag_rules as rules

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
    """Répond localement uniquement aux politesses évidentes."""
    message = _normaliser_message(question)
    reponse = rules.REPONSES_LOCALES.get(message)
    if reponse is None:
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


def _ecrire_json_atomique(chemin: str, donnees: dict) -> None:
    """Écrit un JSON sans risque de fichier partiellement écrit."""
    dossier = os.path.dirname(chemin)
    fd, tmp = tempfile.mkstemp(dir=dossier, suffix=".json.tmp")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(donnees, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, chemin)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _ecrire_pickle_atomique(chemin: str, donnees) -> None:
    """Écrit un fichier pickle avec remplacement atomique."""
    dossier = os.path.dirname(chemin)
    fd, tmp = tempfile.mkstemp(dir=dossier, suffix=".pkl.tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            pickle.dump(donnees, f)
        os.replace(tmp, chemin)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


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
    _ecrire_json_atomique(_manifest_path(course_id), manifest)


def _charger_cache_fichier(course_id: str, empreinte: str) -> dict | None:
    chemin = _cache_path(course_id, empreinte)
    if not os.path.exists(chemin):
        return None
    with open(chemin, "rb") as f:
        return pickle.load(f)


def _sauver_cache_fichier(course_id: str, empreinte: str, donnees: dict) -> None:
    _ecrire_pickle_atomique(_cache_path(course_id, empreinte), donnees)


def index_existe(course_id: str) -> bool:
    f_index, f_chunks = _chemins(course_id)
    return os.path.exists(f_index) and os.path.exists(f_chunks)


def _config_cache(provider) -> dict:
    """Paramètres qui rendent un cache d'indexation encore réutilisable."""
    return {
        "cache_version": CACHE_VERSION,
        "provider": _cache_signature(provider),
        "taille_chunk": config.TAILLE_CHUNK,
        "recouvrement": config.RECOUVREMENT,
        "ocr_enabled": config.OCR_ENABLED,
        "ocr_language": config.OCR_LANGUAGE,
        "ocr_dpi": config.OCR_DPI,
    }


def _manifest_est_compatible(manifest: dict, cache_config: dict) -> bool:
    """Vérifie que le cache correspond encore aux réglages actuels."""
    return all(manifest.get(cle) == valeur for cle, valeur in cache_config.items())


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
    cache_config = _config_cache(provider)
    signature_provider = cache_config["provider"]

    if not _manifest_est_compatible(manifest, cache_config):
        manifest = {**cache_config, "files": {}}

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
            **cache_config,
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
    axes = []
    for mot in _normaliser_message(question).split():
        axe = rules.AXES_COMPARAISON.get(mot)
        if axe and axe not in axes:
            axes.append(axe)
    return axes


def _est_question_preuves(question: str) -> bool:
    """Détecte une demande explicite de preuves ou d'indices."""
    mots = set(_normaliser_message(question).split())
    return bool(mots.intersection(rules.MOTS_PREUVES))


def _est_question_comparaison(question: str) -> bool:
    """Détecte une demande de comparaison explicite."""
    normalisee = _normaliser_message(question)
    return normalisee.startswith("compare") or " comparaison " in f" {normalisee} "


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

    selection = []
    for entite in entites:
        termes_requete = [entite]
        for axe in axes:
            termes_requete.extend(rules.TERMES_PAR_AXE[axe])
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


def _diversifier_par_sources(chunks: list[dict], indices_globaux: list[int], k: int) -> list[int]:
    """Évite qu'une synthèse générale soit dominée par un seul document."""
    selection = []
    sources_vues = set()
    for idx in indices_globaux:
        source = chunks[idx].get("source")
        if source in sources_vues:
            continue
        selection.append(idx)
        sources_vues.add(source)
        if len(selection) >= k:
            return selection

    for idx in indices_globaux:
        if idx not in selection:
            selection.append(idx)
        if len(selection) >= k:
            break
    return selection


def rechercher(index, chunks, provider, question: str, k: int) -> list[dict]:
    """Retourne les meilleurs passages après recherche hybride FAISS + BM25."""
    n_candidats = min(k * 5, len(chunks))
    if _est_question_synthese_generale(question):
        n_candidats = min(max(k * 10, 80), len(chunks))
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

    if _est_question_synthese_generale(question):
        top_k = _diversifier_par_sources(chunks, indices_tries, k)
    else:
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
    if not history:
        return question

    if _est_question_suivi_reponse(question, history):
        for message in reversed(history):
            contenu = message.get("content", "").strip()
            if (
                message.get("role") == "user"
                and contenu
                and not repondre_message_courant(contenu)
            ):
                return f"{contenu} {question}"
        return question

    if _normaliser_message(question) not in rules.SUIVIS_VAGUES:
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


def _est_question_suivi_reponse(question: str, history: list[dict] | None) -> bool:
    """Détecte les réactions courtes à une réponse précédente."""
    if not history:
        return False

    normalisee = _normaliser_message(question)
    mots = normalisee.split()
    if len(mots) > 18:
        return False

    return any(marqueur in normalisee for marqueur in rules.MARQUEURS_SUIVI_REPONSE)


def _est_question_synthese_generale(question: str) -> bool:
    """Détecte une demande de vue d'ensemble du cours ou de ses grands axes."""
    normalisee = _normaliser_message(question)
    return any(marqueur in normalisee for marqueur in rules.MARQUEURS_SYNTHESE_GENERALE)


def _est_question_synthese_detaillee(question: str) -> bool:
    """Détecte une synthèse demandée avec plan ou développement."""
    normalisee = _normaliser_message(question)
    return _est_question_synthese_generale(question) and any(
        marqueur in normalisee for marqueur in rules.MARQUEURS_SYNTHESE_DETAILLEE
    )


def _est_question_sans_details_techniques(question: str) -> bool:
    """Détecte une demande volontairement vulgarisée."""
    normalisee = _normaliser_message(question)
    return any(marqueur in normalisee for marqueur in rules.MARQUEURS_VULGARISATION)


def _instructions_question(question: str, history: list[dict] | None = None) -> str:
    """Ajoute des garde-fous adaptés à la forme de la question."""
    instructions = []

    if _est_question_suivi_reponse(question, history):
        instructions.append(
            "MODE SUIVI / VERIFICATION : l'utilisateur reagit a la reponse "
            "precedente. Ne confirme jamais automatiquement. Relis l'historique "
            "et confronte la reponse precedente aux extraits du CONTEXTE. "
            "Si la reponse precedente est correcte, confirme brievement en "
            "citant les extraits qui le prouvent. Si elle est fausse, incomplete "
            "ou trop forte, corrige-la clairement. Si les extraits ne permettent "
            "pas de trancher, dis que tu ne peux pas confirmer avec les ressources "
            "du cours. Pour une demande de reformulation, reformule seulement "
            "l'idee concernee a partir des extraits, sans repartir dans un "
            "inventaire complet du cours."
        )

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

    if _est_question_comparaison(question):
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

    if _est_question_synthese_detaillee(question):
        instructions.append(
            "MODE SYNTHESE DETAILLEE : réponds avec une synthèse structurée mais "
            "lisible. Réponse attendue : 250 à 380 mots maximum. Commence par une "
            "phrase qui présente le thème général, puis donne 3 à 5 parties "
            "numérotées. Chaque partie doit avoir un titre court et numéroté, seul "
            "sur sa ligne, puis un paragraphe explicatif. Mets le gras uniquement "
            "sur les titres de parties, pas sur les mots dans les paragraphes. "
            "Ajoute les exemples importants quand ils éclairent vraiment l'idée, "
            "sans transformer la réponse en inventaire technique."
        )
    elif _est_question_synthese_generale(question):
        instructions.append(
            "MODE SYNTHESE GENERALE : réponds comme une vue d'ensemble courte du cours. "
            "Réponse attendue : 120 à 180 mots maximum. Commence par une phrase qui "
            "nomme le thème général, puis présente 2 à 4 axes principaux. Pour chaque "
            "axe, explique l'idée en une phrase et cite au maximum un exemple nommé si "
            "cela aide. Regroupe les exemples au lieu de les détailler. N'ouvre pas une "
            "longue sous-partie sur un cas d'étude. Ne donne pas de numéros d'unités "
            "stratigraphiques, de couches, d'inventaire de mobilier ou de datations "
            "fines sauf si la question les demande clairement. "
            "L'objectif est d'aider un étudiant à comprendre la structure du cours, "
            "pas de produire un rapport de fouille."
        )

    if _est_question_sans_details_techniques(question):
        instructions.append(
            "MODE VULGARISATION : la question demande une explication simple. "
            "Évite les termes techniques rares, les codes, les cotes, les sigles, "
            "les typologies précises, les dates trop fines, les noms de séries "
            "d'objets et les inventaires. Préfère les mots courants aux termes "
            "spécialisés ; si un terme technique est indispensable, explique-le en "
            "quelques mots. Réponse attendue : 2 ou 3 paragraphes fluides, avec "
            "une comparaison ou une image simple si cela aide. Ne transforme pas "
            "la réponse en liste de données."
        )

    if not instructions:
        return ""
    return "\nCONSIGNES SPÉCIFIQUES :\n" + "\n".join(instructions) + "\n"


def _doit_verifier_reponse(question: str) -> bool:
    """Active une seconde passe pour les questions les plus sujettes aux extrapolations."""
    return _est_question_preuves(question) or _est_question_comparaison(question)


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
        "Respecte strictement la portée de la question. Pour une comparaison, "
        "garde uniquement les axes demandés et 250 mots maximum. Pour une "
        "demande de preuves, recherche les preuves les plus explicites dans "
        "TOUS les extraits et donne au maximum cinq preuves directes. Si un "
        "élément est décrit comme seulement compatible, associé, indirect ou "
        "sans lien direct établi, SUPPRIME entièrement cet élément de la "
        "réponse. Si une information n'est pas démontrée par le contexte, "
        "retire-la au lieu de la reformuler. Quand la réponse contient plusieurs "
        "idées, garde une structure lisible avec des paragraphes courts, ou des "
        "sections simples si cela aide la compréhension."
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


def _indices_extraits_references(texte: str) -> set[int]:
    """Retrouve les numéros d'extraits cités, même si le format varie un peu."""
    indices = set()
    for bloc in re.findall(rules.MOTIF_MARQUEUR_EXTRAIT, texte or "", flags=re.IGNORECASE):
        indices.update(int(numero) - 1 for numero in re.findall(r"\d+", bloc))
    return indices


def _passages_cites(reponse: str, passages: list[dict]) -> list[dict]:
    """Garde les extraits explicitement cités par le modèle, avec repli sûr."""
    indices = _indices_extraits_references(reponse)
    cites = [passages[idx] for idx in sorted(indices) if 0 <= idx < len(passages)]
    return cites or passages


def _retirer_marqueurs_extraits_visibles(reponse: str) -> str:
    """Retire les marqueurs techniques du texte affiché à l'utilisateur."""
    texte = re.sub(rules.MOTIF_MARQUEUR_EXTRAIT, "", reponse or "", flags=re.IGNORECASE)
    texte = re.sub(
        r"\s*\+?\s*\[\s*Historique(?:\s+de\s+la\s+conversation)?\s*\]",
        "",
        texte,
        flags=re.IGNORECASE,
    )
    texte = re.sub(
        r"(?im)^\s*(?:\+?\s*)?Historique(?:\s+de\s+la\s+conversation)?\s*:?\s*$",
        "",
        texte,
    )
    texte = re.sub(r"\s+([,.;!?])", r"\1", texte)
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    return texte.strip()


def _reponse_signale_absence_information(reponse: str) -> bool:
    """Détecte le refus attendu quand les ressources ne répondent pas."""
    normalisee = _normaliser_message(reponse)
    mots = normalisee.split()
    refus_detecte = any(
        refus in normalisee
        for refus in rules.REFUS_INFORMATION_GLOBALE
    )
    return bool(refus_detecte and (len(mots) <= 18 or any(
        normalisee.startswith(refus)
        for refus in rules.REFUS_INFORMATION_GLOBALE
    )))


def _nettoyer_reponse(reponse: str, question: str) -> str:
    """Retire les astérisques isolés et les pseudo-preuves invalidées."""
    reponse = re.sub(r"(?<!\*)\*(?!\*)", "", reponse)
    if _est_question_preuves(question):
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
                for expression in rules.EXPRESSIONS_PREUVES_A_RETIRER
            )
            if not commentaire_exclusion and not expression_exclue:
                conserves.append(bloc)
        reponse = "\n\n".join(conserves).strip()

    if _est_question_comparaison(question):
        lignes = [
            ligne
            for ligne in reponse.splitlines()
            if not (
                ligne.lstrip().startswith(("-", "•"))
                and not _indices_extraits_references(ligne)
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
        references = sorted(_indices_extraits_references(ligne))
        references = [idx for idx in references if 0 <= idx < len(passages)]
        if not references:
            lignes.append(ligne)
            continue

        references_texte = "".join(
            f"[Extrait {idx + 1}]"
            for idx in references
        )
        sans_references = re.sub(
            rules.MOTIF_MARQUEUR_EXTRAIT,
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


def _retirer_fragments_orphelins(texte: str) -> str:
    """Supprime les petits départs de phrase qui ressemblent à une coupure."""
    paragraphes = re.split(r"\n\s*\n", texte or "")
    conserves = []

    for paragraphe in paragraphes:
        bloc = paragraphe.strip()
        if not bloc:
            continue
        normalise = _normaliser_message(bloc)
        commence_comme_suite = normalise.startswith(rules.CONNECTEURS_FRAGMENT)
        commence_en_minuscule = bloc[:1].islower()
        commence_par_ponctuation = bloc.startswith((",", ";"))
        if conserves and commence_comme_suite and (
            commence_en_minuscule or commence_par_ponctuation
        ):
            continue
        conserves.append(bloc)

    return "\n\n".join(conserves)


def _debut_phrase_texte() -> str:
    """Débuts de phrase fréquents utilisés pour repérer un titre collé."""
    mots = "|".join(re.escape(mot) for mot in rules.DEBUTS_PHRASE_TITRE)
    return rf"(?:{mots})\b"


def _titre_rubrique(titre: str) -> str:
    """Normalise un intertitre court sans changer son sens."""
    titre = re.sub(r"\s*:\s*", " : ", titre.strip())
    titre = re.sub(r"\s{2,}", " ", titre)
    return titre.rstrip(" .")


def _majuscule_initiale(texte: str) -> str:
    """Met une majuscule au premier caractère alphabétique."""
    if not texte:
        return texte
    for index, caractere in enumerate(texte):
        if caractere.isalpha():
            return texte[:index] + caractere.upper() + texte[index + 1:]
    return texte


def _aerer_rubriques_numerotees(texte: str) -> str:
    """Transforme les rubriques numérotées collées en intertitres lisibles."""
    debut_phrase = _debut_phrase_texte()
    lignes = []
    for ligne in texte.splitlines():
        correspondance = re.match(r"^(\s*)(\d+[.)])\s+(.+)$", ligne)
        if not correspondance:
            lignes.append(ligne)
            continue

        indentation, numero, contenu = correspondance.groups()
        titre = ""
        corps = ""
        avec_deux_points = re.match(
            rf"^(.+?:\s*[^.!?\n]{{3,90}}?)\s+({debut_phrase}.+)$",
            contenu,
        )
        if avec_deux_points:
            titre, corps = avec_deux_points.groups()
        else:
            suite_en_et = re.match(
                r"^(.{24,90}?)\s+(et\s+(?:la|le|les|l'|un|une)\b.+)$",
                contenu,
                flags=re.IGNORECASE,
            )
            if suite_en_et:
                titre, corps = suite_en_et.groups()
            else:
                phrase_collee = re.match(
                    rf"^(.{{12,90}}?)\s+({debut_phrase}.+)$",
                    contenu,
                )
                if phrase_collee:
                    titre, corps = phrase_collee.groups()

        if titre and corps:
            titre = _titre_rubrique(titre)
            corps = _majuscule_initiale(corps.strip())
            lignes.append(f"{indentation}**{numero} {titre}**")
            lignes.append("")
            lignes.append(f"{indentation}{corps}")
            continue

        lignes.append(ligne)

    return "\n".join(lignes)


def _retirer_gras(texte: str) -> str:
    """Retire seulement les marqueurs Markdown de gras."""
    return re.sub(r"\*\*(.+?)\*\*", r"\1", texte)


def _ligne_titre_rubrique(ligne: str) -> bool:
    """Reconnaît un intertitre court, sans viser les phrases ordinaires."""
    propre = _retirer_gras(ligne).strip()
    propre = re.sub(r"^\d+[.)]\s+", "", propre)
    if not propre or len(propre) > 95:
        return False
    if propre.startswith(("-", "•")) or propre.endswith((".", "?", "!")):
        return False
    if propre.endswith(":") or propre.endswith(" :"):
        return False
    if " : " in propre:
        gauche, droite = propre.split(" : ", 1)
        if len(droite.split()) > 7:
            return False
        return 2 <= len(gauche.split()) <= 8 and bool(droite.strip())
    mots = propre.split()
    return 2 <= len(mots) <= 8


def _harmoniser_gras_et_titres(texte: str, question: str) -> str:
    """Garde le gras pour les titres seulement et numérote les plans détaillés."""
    lignes = texte.splitlines()
    resultat = []
    numero_titre = 0
    numerotation_detaillee = _est_question_synthese_detaillee(question)

    for ligne in lignes:
        propre = ligne.strip()
        if not propre:
            resultat.append(ligne)
            continue

        titre = _ligne_titre_rubrique(propre)
        if not titre:
            resultat.append(_retirer_gras(ligne))
            continue

        indentation = ligne[:len(ligne) - len(ligne.lstrip())]
        contenu = _titre_rubrique(_retirer_gras(propre))

        if numerotation_detaillee:
            if _retirer_gras(propre).strip().endswith((":")):
                resultat.append(_retirer_gras(ligne))
                continue
            correspondance_numero = re.match(r"^(\d+[.)])\s+(.+)$", contenu)
            if correspondance_numero:
                numero_existant = int(re.match(r"\d+", correspondance_numero.group(1)).group(0))
                numero_titre = max(numero_titre, numero_existant)
                contenu = (
                    f"{correspondance_numero.group(1)} "
                    f"{correspondance_numero.group(2)}"
                )
            else:
                numero_titre += 1
                contenu = f"{numero_titre}. {contenu}"

        resultat.append(f"{indentation}**{contenu}**")

    return "\n".join(resultat)


def _corriger_formulations_courantes(texte: str) -> str:
    """Corrige quelques accrocs de français fréquents dans les sorties LLM."""
    for motif, remplacement in rules.CORRECTIONS_FRANCAIS:
        texte = re.sub(motif, remplacement, texte, flags=re.IGNORECASE)
    return texte


def _separer_titre_colle(texte: str) -> str:
    """Sépare un intertitre si le modèle l'a collé au début du paragraphe."""
    debuts = "|".join(re.escape(mot) for mot in rules.DEBUTS_PHRASE_AERER)
    motif = rf"(?m)^([{rules.MAJUSCULES_FR}][^.!?\n:,;()]{{8,70}})\s+((?:{debuts})\b)"
    return re.sub(motif, r"\1\n\n\2", texte)


def _recoller_mot_liaison_coupe(texte: str) -> str:
    """Répare les coupures créées juste après un petit mot de liaison."""
    mots = "|".join(rules.MOTS_LIAISON_A_RECOLLER)
    motif = rf"\b({mots})\n\n(?=[{rules.MAJUSCULES_FR}])"
    return re.sub(motif, r"\1 ", texte, flags=re.IGNORECASE)


def _reformater_reponse_etudiante(reponse: str, question: str) -> str:
    """Nettoie légèrement la réponse sans imposer de structure lourde."""
    texte = (reponse or "").replace("\r\n", "\n").strip()
    if not texte:
        return ""
    texte = re.sub(r"[ \t]+\n", "\n", texte)
    texte = _aerer_rubriques_numerotees(texte)
    texte = _separer_titre_colle(texte)
    texte = _recoller_mot_liaison_coupe(texte)
    texte = _retirer_fragments_orphelins(texte)
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    texte = re.sub(r"\s+([,.;!?])", r"\1", texte)
    texte = _corriger_formulations_courantes(texte)
    texte = _harmoniser_gras_et_titres(texte, question)
    return texte.strip()

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
        f"{_instructions_question(question, history)}"
        f"QUESTION : {question}\n\n"
        "Réponds en français uniquement à partir du CONTEXTE ci-dessus. "
        "Ajoute [Extrait N] après chaque affirmation factuelle. "
        "Réponds directement, avec une première phrase naturelle qui donne l'idée "
        "principale avant les détails. Évite les introductions génériques, les "
        "conclusions ajoutées, les tableaux et les sections non demandées. "
        "Sauf demande explicitement détaillée, vise 2 à 4 paragraphes courts et "
        "environ 180 à 260 mots maximum. Pour une question simple, préfère des "
        "paragraphes fluides à des titres. Si tu utilises un titre, mets-le seul "
        "sur sa ligne puis explique avec des phrases complètes. N'utilise le gras "
        "qu'avec **...** sur les titres de parties ou de rubriques, jamais sur des "
        "mots isolés dans les paragraphes. Si tu fais des rubriques, ne colle jamais "
        "le titre et son explication sur la même ligne. "
        "Si la question demande de citer, lister ou résumer des notions, donne une "
        "synthèse lisible en quelques rubriques, sans sous-listes imbriquées et sans "
        "inventaire technique inutile. Regroupe les détails proches, limite-toi aux "
        "notions centrales et n'ajoute pas d'exemples précis si la question ne les "
        "demande pas. Pour les rubriques principales, évite les listes numérotées, "
        "sauf si la question demande un résumé détaillé, des grandes parties ou un "
        "plan : dans ce cas, utilise 3 à 5 titres de parties numérotés. "
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
    reponse = _reformater_reponse_etudiante(reponse, question)
    if not reponse.strip():
        return {
            "reponse": rules.REPONSE_ABSENCE_INFORMATION,
            "sources": [],
            "passages": [],
            "tokens": tokens,
        }
    if _reponse_signale_absence_information(reponse):
        return {"reponse": reponse, "sources": [], "passages": [], "tokens": tokens}

    passages_utilises = _passages_cites(reponse, passages)
    sources = sorted({
        f"{p['source']} (p.{p['page']})"
        for p in passages_utilises
    })
    return {"reponse": _retirer_marqueurs_extraits_visibles(reponse), "sources": sources,
            "passages": _exporter_passages(passages_utilises), "tokens": tokens}
