# Plugin block_ragchat — Installation et test

Assistant IA de cours (RAG souverain) pour Moodle 5.1.
Stage Pilote IA Moodle 5.1 — MIN / UT2J.

## Prérequis
Le backend FastAPI doit tourner et être accessible depuis le serveur Moodle :

    cd backend
    pip install -r requirements.txt
    $env:RAG_PROVIDER="ilaas"
    $env:ILAAS_API_KEY="votre_cle_Moodle"
    uvicorn main:app --host 127.0.0.1 --port 8000

## Installation du plugin
1. Copier le dossier `block_ragchat/` dans `MOODLE_ROOT/blocks/`.
   (ou installer le ZIP via Administration du site > Plugins > Installer des plugins)
2. Se connecter en admin : Moodle détecte le plugin et propose la mise à jour
   de la base. Valider.
3. Régler l'URL du backend :
   Administration du site > Plugins > Blocs > Assistant IA du cours
   -> champ "URL du backend RAG" (défaut : http://localhost:8000)

## Utilisation
### Enseignant
1. Activer le mode édition dans un cours.
2. Ajouter le bloc "Assistant IA du cours".
3. Cliquer sur "Réindexer le cours" : le plugin récupère tous les PDF du cours
   et les envoie au backend. Un message confirme (fichiers, chunks, tokens).

### Étudiant
- Taper une question dans le bloc et appuyer sur Entrée ou "Envoyer".
- La réponse s'affiche, ancrée dans le cours, avec les sources citées.

## Points clés (rapport)
- L'URL du backend reste côté serveur : le navigateur ne l'atteint jamais.
- Toute requête passe par les services web Moodle (authentification + droits respectés).
- La récupération des PDF utilise la File API interne (pas d'accès brut au disque ni à la BDD).
- Bascule souveraine : côté backend, `RAG_PROVIDER=ollama` suffit pour passer en local.
- RGPD : la transmission de la question au backend est déclarée (privacy provider).

## Limites connues (prototype)
- Seuls les PDF sont indexés (pas encore Pages/Livres Moodle, ni OCR des scans).
- L'index est reconstruit entièrement à chaque réindexation (pas d'ajout incrémental).
- Réindexation manuelle (pas encore déclenchée automatiquement à l'ajout d'un fichier).
