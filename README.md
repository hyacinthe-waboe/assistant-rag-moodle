# Assistant RAG pour Moodle

Prototype réalisé dans le cadre d'un stage de Licence 3 MIASHS Informatique à
l'Université Toulouse Jean Jaurès.

Le projet ajoute à Moodle un assistant capable de répondre à partir des PDF
d'un cours. Il combine un plugin Moodle, un backend FastAPI et un pipeline RAG
avec recherche hybride.

## Fonctionnement

1. Un enseignant lance l'indexation depuis Moodle.
2. Le plugin transmet les PDF du cours au backend.
3. PyMuPDF extrait le texte et le découpe en passages.
4. `sentence-transformers` produit localement les embeddings.
5. FAISS effectue une recherche sémantique.
6. BM25 recherche les mots précis.
7. RRF fusionne les deux classements.
8. Les meilleurs extraits et la question sont envoyés au modèle de génération.
9. Moodle affiche la réponse, les sources et les extraits utilisés.

## Structure

```text
backend/                         API FastAPI et moteur RAG
Plug in/source/block_ragchat/    Code source du plugin Moodle
ARCHITECTURE.md                  Description technique
rag_moodle.py                    Prototype historique en ligne de commande
```

Les PDF de cours, les index FAISS, les clés API et les documents de stage ne
sont pas versionnés.

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
uvicorn main:app --host 127.0.0.1 --port 8000
```

Le service peut être vérifié avec `GET http://127.0.0.1:8000/health`.

## Installation du plugin Moodle

1. Copier `Plug in/source/block_ragchat` dans `MOODLE_ROOT/blocks/`.
2. Terminer l'installation depuis l'administration Moodle.
3. Configurer l'URL du backend dans les paramètres du bloc.
4. Ajouter le bloc à un cours.
5. Utiliser le bouton de réindexation avec un compte enseignant.

Le plugin est actuellement un prototype alpha testé avec Moodle 5.1.

## Fournisseurs

- `ilaas` : embeddings locaux et génération via l'infrastructure ILAAS.
- `ollama` : embeddings et génération entièrement en local.

Le fournisseur est sélectionné avec la variable `RAG_PROVIDER`.

## Limites actuelles

- Les PDF scannés nécessitent un OCR.
- Les numéros de pages sont approximatifs lorsqu'un passage couvre plusieurs
  pages.
- Une réindexation reconstruit actuellement tout l'index du cours.
- Une question de suivi vague peut échouer après une digression, car
  l'historique n'est pas encore utilisé pour reformuler la requête de recherche.
- Le backend doit rester sur un réseau protégé tant qu'une authentification
  dédiée n'a pas été ajoutée.

## Confidentialité

Les documents complets, l'extraction, les embeddings et la recherche restent
sur le backend. Avec ILAAS, seuls la question et les extraits sélectionnés sont
envoyés au modèle de génération.

Ce dépôt ne contient aucune clé API ni ressource pédagogique utilisée pendant
les tests.

