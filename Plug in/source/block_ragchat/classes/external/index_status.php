<?php
// Fonction externe : consulter la progression d'une indexation.

namespace block_ragchat\external;

defined('MOODLE_INTERNAL') || die();

use core_external\external_api;
use core_external\external_function_parameters;
use core_external\external_single_structure;
use core_external\external_value;

class index_status extends external_api {

    public static function execute_parameters(): external_function_parameters {
        return new external_function_parameters([
            'courseid' => new external_value(PARAM_INT, 'Identifiant du cours'),
        ]);
    }

    public static function execute(int $courseid): array {
        $params = self::validate_parameters(self::execute_parameters(), ['courseid' => $courseid]);
        $context = \context_course::instance($params['courseid']);
        self::validate_context($context);
        require_capability('block/ragchat:reindex', $context);

        $backendurl = trim((string) get_config('block_ragchat', 'backendurl'));
        if ($backendurl === '') {
            throw new \moodle_exception('error_noconfig', 'block_ragchat');
        }

        $ch = curl_init(rtrim($backendurl, '/') . '/index/' . $params['courseid'] . '/status');
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, 10);
        $raw = curl_exec($ch);
        $code = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $err = curl_error($ch);
        curl_close($ch);

        if ($err || $code !== 200) {
            throw new \moodle_exception('error_backend', 'block_ragchat');
        }

        $data = json_decode($raw, true);
        if (!is_array($data) || !isset($data['status'])) {
            throw new \moodle_exception('error_backend', 'block_ragchat');
        }

        $result = is_array($data['result'] ?? null) ? $data['result'] : [];
        return [
            'status' => (string) $data['status'],
            'stage' => (string) ($data['stage'] ?? ''),
            'progress' => (int) ($data['progress'] ?? 0),
            'message' => (string) ($data['message'] ?? ''),
            'error' => (string) ($data['error'] ?? ''),
            'fichiers' => (int) ($result['fichiers'] ?? $data['files'] ?? 0),
            'chunks' => (int) ($result['chunks'] ?? 0),
            'tokens' => (int) ($result['tokens_embedding'] ?? 0),
        ];
    }

    public static function execute_returns(): external_single_structure {
        return new external_single_structure([
            'status' => new external_value(PARAM_ALPHA, 'État du travail'),
            'stage' => new external_value(PARAM_ALPHANUMEXT, 'Étape en cours'),
            'progress' => new external_value(PARAM_INT, 'Progression de 0 à 100'),
            'message' => new external_value(PARAM_TEXT, 'Détail de la progression'),
            'error' => new external_value(PARAM_TEXT, 'Erreur éventuelle'),
            'fichiers' => new external_value(PARAM_INT, 'Nombre de PDF'),
            'chunks' => new external_value(PARAM_INT, 'Nombre de passages'),
            'tokens' => new external_value(PARAM_INT, 'Tokens d\'embedding'),
        ]);
    }
}
