<?php
// Réglages globaux du plugin (Administration du site > Plugins > Blocs > RAG Chat).

defined('MOODLE_INTERNAL') || die();

if ($ADMIN->fulltree) {

    // URL de base du backend FastAPI. L'URL reste côté serveur : jamais exposée au navigateur.
    $settings->add(new admin_setting_configtext(
        'block_ragchat/backendurl',
        get_string('backendurl', 'block_ragchat'),
        get_string('backendurl_desc', 'block_ragchat'),
        'http://localhost:8000',
        PARAM_URL
    ));
}
