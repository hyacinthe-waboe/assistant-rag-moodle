# Architecture du chatbot RAG Moodle

## Les trois parties

1. **Plugin Moodle (`block_ragchat`)**
   - affiche le chat dans le cours ;
   - vérifie les droits de l'utilisateur ;
   - récupère les PDF du cours ;
   - communique avec le backend FastAPI.

2. **Backend FastAPI (`backend/`)**
   - reçoit les PDF et les questions ;
   - construit un index séparé pour chaque cours ;
   - cherche les extraits pertinents ;
   - envoie au modèle uniquement la question et ces extraits.

3. **Fournisseur IA**
   - ILAAS : embeddings locaux, génération sur l'infrastructure UT2J ;
   - Ollama : embeddings et génération entièrement en local.

## Indexation d'un cours

```text
PDF Moodle
  -> extraction du texte page par page
  -> regroupement et découpage en chunks
  -> transformation des chunks en vecteurs
  -> sauvegarde de l'index FAISS et des chunks
```

L'indexation est lancée manuellement par l'enseignant. Chaque cours possède son
propre dossier dans `backend/data/<course_id>/`.

## Réponse à une question

```text
Question de l'étudiant
  -> recherche sémantique FAISS
  -> recherche par mots-clés BM25
  -> fusion des deux classements avec RRF
  -> sélection des meilleurs extraits
  -> génération de la réponse par le modèle
  -> affichage de la réponse, des sources et des extraits dans Moodle
```

L'historique contient au maximum trois échanges. Il aide le modèle à comprendre
les questions de suivi, mais la recherche des extraits utilise la question
courante.

## Rôle des fichiers Python

- `config.py` : paramètres et prompt système.
- `providers.py` : connexion à ILAAS ou Ollama.
- `rag.py` : extraction, indexation, recherche et génération.
- `main.py` : endpoints HTTP utilisés par le plugin Moodle.
- `rag_moodle.py` : prototype historique en ligne de commande.

## Limites actuelles

- seuls les PDF contenant du texte sont indexés ;
- les documents scannés nécessitent un OCR ;
- la page affichée correspond au début approximatif du chunk ;
- la réindexation reconstruit tout l'index du cours.
