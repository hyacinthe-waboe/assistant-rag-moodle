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
$string['backendurl_desc'] = 'Adresse du service RAG (FastAPI). Exemple : http://localhost:8000. '
    . 'Cette URL reste côté serveur et n\'est jamais transmise au navigateur.';

// Interface.
$string['askplaceholder'] = 'Posez une question sur le cours…';
$string['send'] = 'Envoyer';
$string['reindex'] = 'Réindexer le cours';
$string['coursecontextonly'] = 'Cet assistant n\'est disponible qu\'à l\'intérieur d\'un cours.';

// Erreurs.
$string['error_noconfig'] = 'Le backend RAG n\'est pas configuré. Contactez l\'administrateur.';
$string['error_backend'] = 'Le service IA est momentanément indisponible.';
$string['error_nopdf'] = 'Aucun fichier PDF trouvé dans ce cours à indexer.';

// Confidentialité (RGPD).
$string['privacy:metadata:ragbackend'] = 'Pour répondre, la question est transmise à un service RAG '
    . 'qui la traite à partir des ressources du cours.';
$string['privacy:metadata:ragbackend:question'] = 'La question saisie par l\'utilisateur, '
    . 'envoyée au service pour générer une réponse.';
