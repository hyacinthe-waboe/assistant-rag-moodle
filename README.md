# Assistant RAG pour Moodle

Prototype réalisé dans le cadre d'un stage de Licence 3 MIASHS Informatique à
l'Université Toulouse Jean Jaurès.

Le projet ajoute à Moodle un assistant capable de répondre à partir des PDF
d'un cours. Il combine un plugin Moodle, un backend FastAPI et un pipeline RAG
avec recherche hybride.

## Fonctionnement

1. Un enseignant lance l'indexation depuis Moodle.
2. Le plugin transmet les PDF du cours au backend.
3. Le backend répond immédiatement, puis construit l'index en arrière-plan.
4. Moodle interroge la progression et affiche l'étape en cours. Le dernier
   résultat est sauvegardé par cours et reste visible après un rechargement ou
   un redémarrage du backend.
5. Le backend réutilise les caches des PDF inchangés quand c'est possible.
6. PyMuPDF extrait le texte des fichiers nouveaux ou modifiés.
7. `sentence-transformers` produit localement les embeddings.
8. FAISS effectue une recherche sémantique.
9. BM25 recherche les mots précis.
10. RRF fusionne les deux classements.
11. Les meilleurs extraits et la question sont envoyés au modèle de génération.
12. Pour les questions de preuve ou de comparaison, une seconde passe vérifie
    la réponse à partir des extraits sélectionnés.
13. Le backend retire les affirmations numériques non justifiées, les
    pseudo-preuves et les éléments de comparaison sans référence.
14. Moodle affiche la réponse et les sources réellement citées. Si le modèle
    ne produit aucune référence, les passages sélectionnés sont conservés
    comme repli de sécurité.

Les salutations et messages courants reconnus sont traités directement par le
backend, sans recherche documentaire et sans appel au modèle. À la fin d'une
indexation, l'interface distingue les PDF reçus des PDF contenant réellement
du texte exploitable. Les pages sans texte natif peuvent être reconnues par
OCR lorsque Tesseract et les langues `fra` et `eng` sont installés.

## Structure

```text
backend/                         API FastAPI et moteur RAG
backend/prompt_benchmark.py      Benchmark réel du prompt et de la recherche
Plug in/source/block_ragchat/    Code source du plugin Moodle
ARCHITECTURE.md                  Description technique
CONTEXTE_REPRISE_CODEX.md        Guide de reprise sur un autre poste
rapport_stage.docx               Journal et rapport de stage
rag_moodle.py                    Prototype historique en ligne de commande
```

Le rapport principal est versionné afin de faciliter le travail sur plusieurs
postes. Les PDF de cours, les index FAISS, les clés API et les autres documents
de stage ne sont pas versionnés.

## Installation du backend

Prérequis : Python 3.11 ou une version compatible avec les dépendances.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Les variables peuvent être chargées depuis le terminal :

```powershell
$env:RAG_PROVIDER = "ilaas"
$env:ILAAS_API_KEY = "votre_cle"
$env:RAG_SHARED_TOKEN = "secret_partage"
uvicorn main:app --host 127.0.0.1 --port 8000
```

Le service peut être vérifié avec `GET http://127.0.0.1:8000/health`.

## Installation du plugin Moodle

1. Copier `Plug in/source/block_ragchat` dans `MOODLE_ROOT/blocks/`.
2. Terminer l'installation depuis l'administration Moodle.
3. Configurer l'URL du backend dans les paramètres du bloc.
4. Configurer le même jeton partagé que `RAG_SHARED_TOKEN` dans le champ
   `backendtoken`.
5. Ajouter le bloc à un cours.
6. Utiliser le bouton de réindexation avec un compte enseignant.

Le plugin est actuellement un prototype alpha testé avec Moodle 5.1.

## Fournisseurs

- `ilaas` : embeddings locaux et génération via l'infrastructure ILAAS.
- `ollama` : embeddings et génération entièrement en local.

Le fournisseur est sélectionné avec la variable `RAG_PROVIDER`.

## Vérification

Les tests unitaires s'exécutent depuis la racine du dépôt :

```powershell
python -B -m unittest discover -s backend/tests -v
```

Le benchmark nécessite FastAPI sur `http://127.0.0.1:8000` avec les identifiants
du fournisseur chargés. Il crée des cours temporaires et teste aussi le cours
Moodle `3` lorsqu'il est présent localement :

```powershell
python backend/prompt_benchmark.py
```

Le dernier rapport est écrit dans
`backend/data/prompt_benchmark_last.json`, qui n'est pas versionné.
Avec ILAAS, ce benchmark effectue de vrais appels au modèle.

## Limites actuelles

- L'OCR des PDF scannés nécessite l'installation système de Tesseract avec les
  données linguistiques françaises et anglaises.
- Les numéros de pages sont approximatifs lorsqu'un passage couvre plusieurs
  pages.
- Une réindexation réutilise maintenant le cache des PDF inchangés, mais
  reconstruit encore l'index global du cours.
- Quelques questions de suivi vagues, comme « Peux-tu préciser ? », sont
  complétées localement avec la dernière question de l'étudiant avant la
  recherche. Les suivis plus ambigus après une longue digression restent une
  limite du prototype.
- Les questions de preuve et de comparaison peuvent nécessiter deux appels au
  modèle, donc être plus lentes et consommer davantage de jetons.
- Le jeton partagé protège les appels entre Moodle et FastAPI, mais ne remplace
  pas HTTPS, un pare-feu et une gestion complète des identités en production.

## Confidentialité

Les documents complets, l'extraction, les embeddings et la recherche restent
sur le backend. Avec ILAAS, seuls la question, l'historique court et les
extraits sélectionnés sont envoyés au modèle de génération.

Quand un jeton partagé est configuré, le plugin Moodle l'envoie dans l'en-tête
`X-RAG-Token` et le backend le vérifie avant de traiter la requête.

Ce dépôt ne contient aucune clé API ni ressource pédagogique utilisée pendant
les tests.
