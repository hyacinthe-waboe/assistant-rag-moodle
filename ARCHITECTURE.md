# Architecture du chatbot RAG Moodle

## Les trois parties

1. **Plugin Moodle (`block_ragchat`)**
   - affiche le chat dans le cours ;
   - vérifie les droits de l'utilisateur ;
   - récupère les PDF du cours ;
   - communique avec le backend FastAPI et envoie un jeton partagé si configuré.

2. **Backend FastAPI (`backend/`)**
   - reçoit les PDF et les questions ;
   - construit un index séparé pour chaque cours ;
   - réutilise un cache par PDF inchangé pour éviter les recalculs inutiles ;
   - cherche les extraits pertinents ;
   - envoie au modèle uniquement la question, un historique court et les extraits sélectionnés ;
   - nettoie la réponse avant l'affichage.

3. **Fournisseur IA**
   - ILAAS : embeddings locaux, génération sur l'infrastructure UT2J ;
   - Ollama : embeddings et génération entièrement en local.

## Indexation d'un cours

```text
PDF Moodle
  -> dépôt temporaire sur le backend
  -> réponse HTTP 202 à Moodle
  -> traitement en arrière-plan
  -> empreinte SHA-256 et réutilisation du cache des PDF inchangés
  -> extraction du texte page par page, avec OCR optionnel
  -> regroupement et découpage en chunks
  -> transformation des chunks en vecteurs
  -> sauvegarde atomique de l'index FAISS et des chunks
  -> sauvegarde de l'état final de l'indexation
```

L'indexation est lancée manuellement par l'enseignant. Chaque cours possède son
propre dossier dans `backend/data/<course_id>/`. Le plugin interroge
`GET /index/<course_id>/status` toutes les trois secondes et affiche l'étape
ainsi que le pourcentage de progression.

Depuis la dernière mise à jour, le backend conserve aussi un cache par fichier
PDF. Lorsqu'un document n'a pas changé, son extraction et sa vectorisation sont
réutilisées au lieu d'être recalculées.

L'état courant est conservé en mémoire pendant le traitement et le dernier
résultat est enregistré dans `backend/data/<course_id>/index_status.json`.
L'interface peut donc réafficher la dernière indexation terminée après un
rechargement de Moodle ou un redémarrage de FastAPI.

## Réponse à une question

```text
Question de l'étudiant
  -> réponse locale si c'est une salutation ou un remerciement simple
  -> recherche sémantique FAISS
  -> recherche par mots-clés BM25
  -> fusion des deux classements avec RRF
  -> diversification par entités et axes de comparaison
  -> sélection des meilleurs extraits
  -> génération de la réponse par le modèle
  -> seconde vérification pour les preuves et comparaisons
  -> nettoyage des affirmations non justifiées
  -> absence de sources affichées si l'information n'est pas documentée
  -> affichage de la réponse et des extraits cités dans Moodle
```

L'historique contient au maximum trois échanges. Il aide le modèle à comprendre
les questions de suivi. Pour quelques formulations vagues reconnues, la
recherche réutilise aussi la dernière question de l'étudiant, sans appel
supplémentaire au modèle.

## Rôle des fichiers Python

- `config.py` : paramètres et prompt système.
- `providers.py` : connexion à ILAAS ou Ollama.
- `rag.py` : extraction, indexation, recherche et génération.
- `rag_rules.py` : réponses locales et marqueurs de langage utilisés par `rag.py`.
- `main.py` : endpoints HTTP utilisés par le plugin Moodle.
- `prompt_benchmark.py` : scénarios réels de non-régression du prompt.

## Limites actuelles

- seuls les PDF sont actuellement pris en charge ;
- l'OCR des documents scannés nécessite Tesseract sur le backend ;
- la page affichée correspond au début approximatif du chunk ;
- la réindexation reconstruit encore l'index global du cours, même si les PDF
  inchangés réutilisent leur cache.
- le plugin et le backend utilisent un jeton partagé `X-RAG-Token` quand il
  est configuré.
- les questions complexes de preuve et de comparaison utilisent une seconde
  génération de contrôle.
- certains marqueurs de langage restent volontairement listés dans
  `rag_rules.py` pour traiter les cas simples sans alourdir le prompt.
