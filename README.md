<h1 align="center">🤖 Assistant RAG pour Moodle</h1>

<p align="center">
  <strong>Interroger les PDF d'un cours directement dans Moodle, avec des réponses sourcées et une maîtrise des données.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Moodle-F98012?style=for-the-badge&logo=moodle&logoColor=white" alt="Moodle">
  <img src="https://img.shields.io/badge/FAISS-Search-4B8BBE?style=for-the-badge" alt="FAISS">
  <img src="https://img.shields.io/badge/BM25%20%2B%20RRF-Hybrid_Search-6C63FF?style=for-the-badge" alt="BM25 et RRF">
</p>

## 🎓 Le projet

Ce prototype académique a été réalisé pendant mon stage de **Licence 3 MIASHS** pour étudier l'intégration d'un assistant RAG dans Moodle.

Le plugin permet à un étudiant de poser une question sur son cours. Le backend recherche les passages les plus pertinents dans les PDF, demande au modèle de construire une réponse et renvoie les sources utilisées.

L'objectif n'est pas seulement de faire fonctionner une IA : le projet cherche aussi à **limiter les hallucinations**, à rendre les réponses vérifiables et à conserver les documents et l'indexation sur une infrastructure maîtrisée.

## ✨ Points forts

- 🔌 intégration sous forme de bloc Moodle ;
- ⚡ backend FastAPI ;
- 🧠 embeddings calculés localement ;
- 🔍 recherche hybride FAISS et BM25 ;
- 🏆 fusion des résultats avec RRF ;
- 📚 réponses accompagnées de sources ;
- ♻️ cache SHA-256 des PDF inchangés ;
- ⏳ indexation asynchrone avec suivi de progression ;
- 👋 traitement direct des salutations sans appel au modèle ;
- 🧭 absence de sources affichées quand la réponse indique que l'information n'est pas dans le cours ;
- 🗣️ format de réponse plus lisible : paragraphes courts, titres utiles et gras limité ;
- 👁️ OCR optionnel avec Tesseract ;
- 🛡️ contrôle renforcé des preuves, comparaisons et affirmations numériques ;
- 🧪 50 tests automatisés.

## 🔄 Comment fonctionne le pipeline ?

```text
PDF du cours
     │
     ▼
Extraction du texte ──► OCR optionnel
     │
     ▼
Découpage en passages
     │
     ├──► Embeddings locaux ──► FAISS
     │
     └──► Recherche lexicale ─► BM25
                              │
                              ▼
                       Fusion avec RRF
                              │
                              ▼
                 Passages les plus pertinents
                              │
                              ▼
                     ILAAS ou Ollama
                              │
                              ▼
                  Réponse + sources dans Moodle
```

### Étapes détaillées

1. L'enseignant lance l'indexation depuis Moodle.
2. Le plugin transmet les PDF au backend.
3. FastAPI accepte la demande et construit l'index en arrière-plan.
4. Moodle suit la progression et conserve le dernier résultat par cours.
5. Les PDF inchangés réutilisent leur cache.
6. PyMuPDF extrait le texte et Tesseract peut traiter les pages scannées.
7. `sentence-transformers` produit les embeddings localement.
8. FAISS recherche les passages proches par le sens.
9. BM25 recherche les mots et expressions précis.
10. RRF fusionne les deux classements.
11. Les meilleurs extraits sont envoyés au modèle avec la question.
12. Les réponses sensibles peuvent être vérifiées lors d'une seconde passe.
13. Le backend filtre les affirmations non justifiées.
14. Si l'information est absente du cours, le backend renvoie une réponse sans sources.
15. Moodle affiche la réponse et les sources réellement retenues.

## 🧱 Architecture

```text
backend/                         # API FastAPI et moteur RAG
backend/prompt_benchmark.py      # Benchmark du prompt et de la recherche
backend/rag_rules.py             # Marqueurs de langage et réponses locales
moodle/block_ragchat/            # Plugin Moodle
ARCHITECTURE.md                  # Documentation technique détaillée
```

Les rapports, PDF de cours, index FAISS, clés API et documents de travail ne sont pas versionnés.

## 🚀 Installer le backend

Prérequis : Python 3.11 ou une version compatible avec les dépendances.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Exemple de configuration :

```powershell
$env:RAG_PROVIDER = "ilaas"
$env:ILAAS_API_KEY = "votre_cle"
$env:RAG_SHARED_TOKEN = "secret_partage"
uvicorn main:app --host 127.0.0.1 --port 8000
```

Vérification :

```text
GET http://127.0.0.1:8000/health
```

## 🧩 Installer le plugin Moodle

1. Copier `moodle/block_ragchat` dans `MOODLE_ROOT/blocks/`.
2. Terminer l'installation depuis l'administration Moodle.
3. Configurer `http://127.0.0.1:8000` comme URL du backend.
4. Utiliser le même secret dans `RAG_SHARED_TOKEN` et `backendtoken`.
5. Ajouter le bloc à un cours.
6. Lancer la réindexation avec un compte enseignant.

Le plugin est un prototype alpha testé avec Moodle 5.1.

## 🧠 Fournisseurs de génération

| Fournisseur | Fonctionnement |
|---|---|
| `ilaas` | Embeddings locaux et génération via l'infrastructure ILAAS |
| `ollama` | Embeddings et génération entièrement en local |

Le fournisseur est sélectionné avec `RAG_PROVIDER`.

## 🧪 Tests et benchmark

Lancer les tests depuis la racine :

```powershell
python -B -m unittest discover -s backend/tests -v
```

Lancer le benchmark avec FastAPI démarré et le fournisseur configuré :

```powershell
python backend/prompt_benchmark.py
```

Le rapport local est écrit dans `backend/data/prompt_benchmark_last.json`. Avec ILAAS, le benchmark effectue de vrais appels au modèle.

## 🔐 Souveraineté et confidentialité

Les documents complets, l'extraction, les embeddings et la recherche restent sur le backend.

Avec ILAAS, seuls les éléments nécessaires à la génération sont transmis :

- la question ;
- un historique court ;
- les extraits sélectionnés.

Le plugin peut envoyer un jeton partagé dans l'en-tête `X-RAG-Token`, vérifié par FastAPI avant le traitement de la requête.

Ce dépôt ne contient **aucune clé API** ni ressource pédagogique utilisée pendant les tests.

## 🚧 Limites actuelles

- l'OCR demande Tesseract avec les langues `fra` et `eng` ;
- les numéros de pages peuvent être approximatifs lorsqu'un passage traverse plusieurs pages ;
- l'index global du cours est encore reconstruit après la réutilisation des caches ;
- certains suivis très vagues ou éloignés du sujet restent difficiles à interpréter ;
- les questions de preuve ou de comparaison peuvent nécessiter deux appels au modèle ;
- les règles textuelles explicites sont regroupées dans `backend/rag_rules.py` pour garder `rag.py` plus lisible ;
- le jeton partagé ne remplace pas HTTPS, un pare-feu et une authentification complète en production.

## 👨‍💻 Auteur

**Hyacinthe Waboe**<br>
Projet de stage de Licence 3 MIASHS.

<p align="center"><em>Une IA utile n'est pas seulement une IA qui répond : c'est une IA dont on peut comprendre et vérifier la réponse. ✨</em></p>
