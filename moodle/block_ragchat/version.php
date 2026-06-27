<?php
// Métadonnées du plugin block_ragchat.
// Prototype d'assistant RAG pour Moodle.

defined('MOODLE_INTERNAL') || die();

$plugin->component = 'block_ragchat';   // nom canonique du plugin (type_nom)
$plugin->version   = 2026062301;        // AAAAMMJJXX — à incrémenter à chaque màj
$plugin->requires  = 2024042200;        // Moodle 4.4+ (external API namespacée)
$plugin->maturity  = MATURITY_ALPHA;    // prototype de stage
$plugin->release   = '0.9.4';
