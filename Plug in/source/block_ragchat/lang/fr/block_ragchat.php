<?php
// Chaînes de langue françaises du plugin block_ragchat.

defined('MOODLE_INTERNAL') || die();

$string['pluginname'] = 'Assistant IA du cours';

// Capacités.
$string['ragchat:addinstance'] = 'Ajouter un bloc Assistant IA';
$string['ragchat:use'] = 'Utiliser l\'assistant IA du cours';
$string['ragchat:reindex'] = 'Réindexer les ressources du cours';

// Réglages.
$string['backendurl'] = 'URL du backend RAG';
$string['backendurl_desc'] = 'Adresse du service RAG (FastAPI). Exemple : http://127.0.0.1:8000. '
    . 'Cette URL reste côté serveur et n\'est jamais transmise au navigateur.';

// Interface.
$string['askplaceholder'] = 'Posez une question sur le cours…';
$string['send'] = 'Envoyer';
$string['reindex'] = 'Réindexer le cours';
$string['coursecontextonly'] = 'Cet assistant n\'est disponible qu\'à l\'intérieur d\'un cours.';

// Erreurs.
$string['error_noconfig'] = 'Le backend RAG n\'est pas configuré. Contactez l\'administrateur.';
$string['error_backend'] = 'Le service IA est momentanément indisponible.';
$string['error_invalidquestion'] = 'La question est vide, trop longue ou invalide.';
$string['error_noindex'] = 'Les ressources de ce cours ne sont pas encore indexées.';
$string['error_nopdf'] = 'Aucun fichier PDF trouvé dans ce cours à indexer.';

// Confidentialité (RGPD).
$string['privacy:metadata:ragbackend'] = 'Le backend RAG indexe les PDF du cours et traite les questions. '
    . 'Selon le fournisseur configuré, la question, l\'historique court et les extraits sélectionnés '
    . 'peuvent être transmis au service de génération. Les PDF complets et l\'index ne sont pas '
    . 'transmis à ILAAS.';
$string['privacy:metadata:ragbackend:courseid'] = 'L\'identifiant du cours concerné.';
$string['privacy:metadata:ragbackend:documents'] = 'Les PDF du cours transmis au backend pour '
    . 'l\'extraction et l\'indexation.';
$string['privacy:metadata:ragbackend:question'] = 'La question saisie par l\'utilisateur, '
    . 'envoyée au service pour générer une réponse.';
$string['privacy:metadata:ragbackend:history'] = 'Les trois derniers échanges de la conversation, '
    . 'utilisés pour comprendre les questions de suivi.';
