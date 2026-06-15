# Plugin block_ragchat — Installation et test

Assistant IA de cours (RAG souverain) pour Moodle 5.1.
Stage Pilote IA Moodle 5.1 — MIN / UT2J.

## Prérequis
Le backend FastAPI doit tourner et être accessible depuis le serveur Moodle :

    cd backend
    pip install -r requirements.txt
    $env:RAG_PROVIDER="ilaas"
    $env:ILAAS_API_KEY="votre_cle_Moodle"
    $env:RAG_SHARED_TOKEN="secret_partage"
    uvicorn main:app --host 127.0.0.1 --port 8000

## Installation du plugin
1. Copier le dossier `block_ragchat/` dans `MOODLE_ROOT/blocks/`.
   (ou installer le ZIP via Administration du site > Plugins > Installer des plugins)
2. Se connecter en admin : Moodle détecte le plugin et propose la mise à jour
   de la base. Valider.
3. Régler l'URL du backend :
   Administration du site > Plugins > Blocs > Assistant IA du cours
   -> champ "URL du backend RAG" (défaut : http://127.0.0.1:8000)
4. Renseigner le jeton d'authentification partagé si le backend l'exige.

## Utilisation
### Enseignant
1. Activer le mode édition dans un cours.
2. Ajouter le bloc "Assistant IA du cours".
3. Cliquer sur "Réindexer le cours" : le plugin récupère tous les PDF du cours
   et les envoie au backend. La progression est affichée pendant leur
   traitement en arrière-plan, puis un message distingue les PDF reçus, les
   PDF contenant du texte exploitable et le nombre de passages générés.

### Étudiant
- Taper une question dans le bloc et appuyer sur Entrée ou "Envoyer".
- La réponse s'affiche, ancrée dans le cours, avec les sources citées.
- Les salutations simples sont traitées localement, sans recherche ni appel au
  modèle de génération.

## Points clés (rapport)
- L'URL du backend reste côté serveur : le navigateur ne l'atteint jamais.
- Toute requête passe par les services web Moodle (authentification + droits respectés).
- Les appels au backend sont protégés par un jeton partagé côté serveur.
- La récupération des PDF utilise la File API interne (pas d'accès brut au disque ni à la BDD).
- Bascule souveraine : côté backend, `RAG_PROVIDER=ollama` suffit pour passer en local.
- RGPD : le privacy provider déclare les PDF indexés, l'identifiant du cours,
  la question et l'historique court transmis au backend.

## Limites connues (prototype)
- Seuls les PDF sont indexés (pas encore Pages/Livres Moodle).
- Les scans nécessitent Tesseract sur la machine du backend pour activer l'OCR.
- Les PDF inchangés réutilisent leur cache, mais l'index global du cours est
  encore reconstruit à chaque réindexation.
- Réindexation manuelle (pas encore déclenchée automatiquement à l'ajout d'un fichier).
