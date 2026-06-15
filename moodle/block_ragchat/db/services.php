<?php
// Déclaration des fonctions externes appelées en AJAX par le module JS.

defined('MOODLE_INTERNAL') || die();

$functions = [

    // Poser une question -> backend /ask.
    'block_ragchat_ask' => [
        'classname'   => 'block_ragchat\\external\\ask',
        'description' => 'Pose une question au chatbot RAG du cours.',
        'type'        => 'read',
        'ajax'        => true,        // appelable depuis le navigateur via core/ajax
        'capabilities'=> 'block/ragchat:use',
    ],

    // Réindexer les ressources du cours -> backend /index/{courseid}.
    'block_ragchat_reindex' => [
        'classname'   => 'block_ragchat\\external\\reindex',
        'description' => 'Récupère les PDF du cours et (re)construit l\'index RAG.',
        'type'        => 'write',
        'ajax'        => true,
        'capabilities'=> 'block/ragchat:reindex',
    ],

    // Consulter la progression de l'indexation en arrière-plan.
    'block_ragchat_index_status' => [
        'classname'   => 'block_ragchat\\external\\index_status',
        'description' => 'Renvoie la progression de l\'indexation RAG du cours.',
        'type'        => 'read',
        'ajax'        => true,
        'capabilities'=> 'block/ragchat:reindex',
    ],
];
