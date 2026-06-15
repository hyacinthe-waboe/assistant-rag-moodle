<?php
// Helpers du plugin block_ragchat.

defined('MOODLE_INTERNAL') || die();

function block_ragchat_backend_headers(bool $json = true): array {
    $headers = [];
    $token = trim((string) get_config('block_ragchat', 'backendtoken'));
    if ($token !== '') {
        $headers[] = 'X-RAG-Token: ' . $token;
    }
    if ($json) {
        $headers[] = 'Content-Type: application/json';
    }
    return $headers;
}
