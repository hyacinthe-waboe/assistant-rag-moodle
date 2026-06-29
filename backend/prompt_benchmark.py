"""Banc d'essai manuel du comportement RAG avec le vrai fournisseur configuré.

Le script crée des cours temporaires, les indexe via l'API locale, pose des
questions représentatives, affiche les résultats et supprime les données créées.
"""

import json
import os
import shutil
import tempfile
import time
import urllib.request
import uuid

import fitz


BASE_URL = os.getenv("RAG_BENCHMARK_URL", "http://127.0.0.1:8000")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ATTENTE_INDEXATION_SECONDES = 180


def _requete_json(path: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    requete = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=headers,
    )
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        return json.loads(reponse.read().decode("utf-8"))


def _creer_pdf(chemin: str, pages: list[str]) -> None:
    document = fitz.open()
    for texte in pages:
        page = document.new_page()
        page.insert_textbox(
            fitz.Rect(50, 50, 545, 790),
            texte,
            fontsize=11,
            fontname="helv",
        )
    document.save(chemin)
    document.close()


def _multipart_pdf(chemin: str) -> tuple[bytes, str]:
    frontiere = f"----ragbenchmark{uuid.uuid4().hex}"
    nom = os.path.basename(chemin)
    with open(chemin, "rb") as fichier:
        contenu = fichier.read()
    corps = (
        f"--{frontiere}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{nom}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8")
    corps += contenu
    corps += f"\r\n--{frontiere}--\r\n".encode("utf-8")
    return corps, frontiere


def _indexer(course_id: str, chemin_pdf: str) -> None:
    corps, frontiere = _multipart_pdf(chemin_pdf)
    requete = urllib.request.Request(
        f"{BASE_URL}/index/{course_id}",
        data=corps,
        headers={"Content-Type": f"multipart/form-data; boundary={frontiere}"},
    )
    with urllib.request.urlopen(requete, timeout=60):
        pass

    for _ in range(ATTENTE_INDEXATION_SECONDES):
        etat = _requete_json(f"/index/{course_id}/status?include_finished=1")
        if etat.get("status") == "completed":
            return
        if etat.get("status") == "failed":
            raise RuntimeError(etat.get("error") or "Indexation échouée")
        time.sleep(1)
    raise TimeoutError("Indexation trop longue")


def _poser(course_id: str, question: str, history: list[dict] | None = None) -> dict:
    return _requete_json(
        "/ask",
        {
            "course_id": course_id,
            "question": question,
            "k": 10,
            "history": history or [],
        },
    )


def _evaluer(texte: str, requis: list[str], interdits: list[str],
             un_parmi: list[list[str]] | None = None) -> tuple[bool, list[str]]:
    normalise = texte.casefold()
    erreurs = []
    for terme in requis:
        if terme.casefold() not in normalise:
            erreurs.append(f"manquant: {terme}")
    for terme in interdits:
        if terme.casefold() in normalise:
            erreurs.append(f"interdit: {terme}")
    for groupe in un_parmi or []:
        if not any(terme.casefold() in normalise for terme in groupe):
            erreurs.append(f"aucun parmi: {', '.join(groupe)}")
    return not erreurs, erreurs


def _supprimer_cours_crees(cours_ids: list[str]) -> None:
    """Supprime les données temporaires créées par le benchmark."""
    for course_id in cours_ids:
        shutil.rmtree(os.path.join(DATA_DIR, course_id), ignore_errors=True)


def main() -> int:
    suffixe = str(int(time.time()))[-6:]
    cours_sciences = f"91{suffixe}"
    cours_info = f"92{suffixe}"
    cours_histoire = f"93{suffixe}"
    cours_crees = [cours_sciences, cours_info, cours_histoire]

    scenarios = []
    with tempfile.TemporaryDirectory(prefix="rag_prompt_benchmark_") as dossier:
        sciences = os.path.join(dossier, "sciences.pdf")
        informatique = os.path.join(dossier, "informatique.pdf")
        histoire = os.path.join(dossier, "histoire.pdf")

        _creer_pdf(sciences, [
            (
                "Etude expérimentale du catalyseur Zeta\n\n"
                "L'article a été publié en 2025. L'expérience a été menée en 2018.\n"
                "L'hypothèse testée est que le catalyseur Zeta accélère la réaction R.\n"
                "Preuves directes : dans cinq répétitions, le temps moyen est passé "
                "de 18 minutes sans catalyseur à 9 minutes avec Zeta. Le groupe témoin "
                "est resté proche de 18 minutes.\n"
                "Une coloration bleue a aussi été observée. Le rapport précise "
                "explicitement que cette couleur n'est pas une preuve de l'effet "
                "catalytique et peut venir du récipient."
            )
        ])
        _creer_pdf(informatique, [
            (
                "Comparaison de deux systèmes logiciels\n\n"
                "Le document comparatif a été publié en 2026.\n"
                "Atlas a été introduit en 2012. Sa fonction est de classer des textes. "
                "Il reçoit du texte UTF-8 et produit une catégorie.\n"
                "Boréal a été introduit en 2019. Sa fonction est de détecter des "
                "anomalies dans des séries numériques. Il reçoit une série de nombres "
                "et produit un score d'anomalie.\n"
                "Aucun des deux systèmes ne fournit de traduction automatique."
            )
        ])
        _creer_pdf(histoire, [
            (
                "Notice documentaire sur le traité d'Orme\n\n"
                "Cette notice a été rédigée et publiée en 2024. Le traité d'Orme a "
                "été signé en 1919. Les négociations avaient commencé en 1918. "
                "La date 2024 est uniquement la date de publication de la notice."
            )
        ])

        _indexer(cours_sciences, sciences)
        _indexer(cours_info, informatique)
        _indexer(cours_histoire, histoire)

        scenarios.extend([
            {
                "nom": "preuves_scientifiques",
                "resultat": _poser(
                    cours_sciences,
                    "Quelles preuves directes montrent que le catalyseur Zeta accélère la réaction ?",
                ),
                "requis": ["18", "9", "cinq"],
                "interdits": ["coloration bleue"],
            },
            {
                "nom": "comparaison_informatique",
                "resultat": _poser(
                    cours_info,
                    "Compare Atlas et Boréal selon leurs fonctions et leurs périodes.",
                ),
                "requis": ["class", "anomal", "2012", "2019"],
                "interdits": ["2026", "traduction automatique"],
            },
            {
                "nom": "metadonnees_histoire",
                "resultat": _poser(
                    cours_histoire,
                    "En quelle année le traité d'Orme a-t-il été signé ?",
                ),
                "requis": ["1919"],
                "interdits": ["signé en 2024"],
            },
            {
                "nom": "hors_sujet",
                "resultat": _poser(
                    cours_histoire,
                    "Quel temps fera-t-il demain à Toulouse ?",
                ),
                "requis": ["pas présente dans les ressources du cours"],
                "interdits": [],
            },
            {
                "nom": "salutation",
                "resultat": _poser(cours_histoire, "Bonjour, qui es-tu ?"),
                "requis": ["assistant ia"],
                "interdits": ["extrait"],
            },
        ])

        if rag.index_existe("3"):
            scenarios.extend([
                {
                "nom": "goiffieux_preuves",
                "resultat": _poser(
                    "3",
                    "Quelles preuves archéologiques indiquent la présence d’un vignoble à Goiffieux ?",
                ),
                "requis": ["tranch", "provign"],
                "un_parmi": [["pollens", "empreinte", "tuteur", "ceps"]],
                "interdits": ["aucune autre preuve", "suggérant une activité agricole"],
                },
                {
                "nom": "goiffieux_comparaison",
                "resultat": _poser(
                    "3",
                    "Compare la villa de Goiffieux et la Casa di Sallustio sans confondre leurs localisations, leurs fonctions et leurs périodes.",
                ),
                "requis": [
                    "localisation", "fonction", "période",
                    "goiffieux", "sallustio", "au moins au ii",
                ],
                "interdits": ["79 ap. j.-c.", "réaménagements en 2005"],
                },
            ])

    reussis = 0
    rapport = []
    for scenario in scenarios:
        texte = scenario["resultat"]["reponse"]
        ok, erreurs = _evaluer(
            texte,
            scenario["requis"],
            scenario["interdits"],
            scenario.get("un_parmi"),
        )
        reussis += int(ok)
        rapport.append({
            "nom": scenario["nom"],
            "ok": ok,
            "erreurs": erreurs,
            "reponse": texte,
            "sources": scenario["resultat"].get("sources", []),
            "tokens": scenario["resultat"].get("tokens", 0),
        })

    sortie = os.path.join(DATA_DIR, "prompt_benchmark_last.json")
    with open(sortie, "w", encoding="utf-8") as fichier:
        json.dump(
            {"score": reussis, "total": len(rapport), "scenarios": rapport},
            fichier,
            ensure_ascii=False,
            indent=2,
        )

    _supprimer_cours_crees(cours_crees)

    print(f"Score : {reussis}/{len(rapport)}")
    for resultat in rapport:
        statut = "OK" if resultat["ok"] else "ECHEC"
        print(f"{statut} - {resultat['nom']}")
        for erreur in resultat["erreurs"]:
            print(f"  {erreur}")
    print(f"Rapport : {sortie}")
    return 0 if reussis == len(rapport) else 1


if __name__ == "__main__":
    raise SystemExit(main())
