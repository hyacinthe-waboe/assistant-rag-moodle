<?php
// Fonction externe : consulter la progression d'une indexation.

namespace block_ragchat\external;

defined('MOODLE_INTERNAL') || die();

require_once(__DIR__ . '/../../lib.php');

use core_external\external_api;
use core_external\external_function_parameters;
use core_external\external_single_structure;
use core_external\external_value;

class index_status extends external_api {

    public static function execute_parameters(): external_function_parameters {
        return new external_function_parameters([
            'courseid' => new external_value(PARAM_INT, 'Identifiant du cours'),
            'include_finished' => new external_value(
                PARAM_BOOL,
                'Retourne aussi le dernier état terminé',
                VALUE_DEFAULT,
                false
            ),
        ]);
    }

    public static function execute(int $courseid, bool $include_finished = false): array {
        $params = self::validate_parameters(self::execute_parameters(), [
            'courseid' => $courseid,
            'include_finished' => $include_finished,
        ]);
        $context = \context_course::instance($params['courseid']);
        self::validate_context($context);
        require_capability('block/ragchat:reindex', $context);

        $backendurl = trim((string) get_config('block_ragchat', 'backendurl'));
        if ($backendurl === '') {
            throw new \moodle_exception('error_noconfig', 'block_ragchat');
        }

        $url = rtrim($backendurl, '/') . '/index/' . $params['courseid'] . '/status';
        if (!empty($params['include_finished'])) {
            $url .= '?include_finished=1';
        }

        $ch = curl_init($url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_HTTPHEADER, block_ragchat_backend_headers(false));
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
            'started_at' => (string) ($data['started_at'] ?? ''),
            'updated_at' => (string) ($data['updated_at'] ?? ''),
            'last_status' => (string) ($data['last_status'] ?? ''),
            'last_stage' => (string) ($data['last_stage'] ?? ''),
            'last_progress' => (int) ($data['last_progress'] ?? 0),
            'last_message' => (string) ($data['last_message'] ?? ''),
            'last_error' => (string) ($data['last_error'] ?? ''),
            'last_started_at' => (string) ($data['last_started_at'] ?? ''),
            'last_updated_at' => (string) ($data['last_updated_at'] ?? ''),
            'fichiers' => (int) ($result['fichiers'] ?? $data['fichiers'] ?? $data['files'] ?? 0),
            'fichiersexploitables' => (int) (
                $result['fichiers_exploitables'] ?? $data['fichiersexploitables'] ?? 0
            ),
            'fichiersocr' => (int) ($result['fichiers_ocr'] ?? $data['fichiers_ocr'] ?? 0),
            'pagesocr' => (int) ($result['pages_ocr'] ?? $data['pages_ocr'] ?? 0),
            'chunks' => (int) ($result['chunks'] ?? $data['chunks'] ?? 0),
            'tokens' => (int) ($result['tokens_embedding'] ?? $data['tokens'] ?? 0),
        ];
    }

    public static function execute_returns(): external_single_structure {
        return new external_single_structure([
            'status' => new external_value(PARAM_ALPHA, 'État du travail'),
            'stage' => new external_value(PARAM_ALPHANUMEXT, 'Étape en cours'),
            'progress' => new external_value(PARAM_INT, 'Progression de 0 à 100'),
            'message' => new external_value(PARAM_TEXT, 'Détail de la progression'),
            'error' => new external_value(PARAM_TEXT, 'Erreur éventuelle'),
            'started_at' => new external_value(PARAM_TEXT, 'Date de démarrage UTC'),
            'updated_at' => new external_value(PARAM_TEXT, 'Dernière mise à jour UTC'),
            'last_status' => new external_value(PARAM_TEXT, 'Dernier état connu'),
            'last_stage' => new external_value(PARAM_TEXT, 'Dernière étape connue'),
            'last_progress' => new external_value(PARAM_INT, 'Dernière progression connue'),
            'last_message' => new external_value(PARAM_TEXT, 'Dernier message connu'),
            'last_error' => new external_value(PARAM_TEXT, 'Dernière erreur connue'),
            'last_started_at' => new external_value(PARAM_TEXT, 'Dernier démarrage UTC'),
            'last_updated_at' => new external_value(PARAM_TEXT, 'Dernière mise à jour UTC'),
            'fichiers' => new external_value(PARAM_INT, 'Nombre de PDF'),
            'fichiersexploitables' => new external_value(
                PARAM_INT,
                'Nombre de PDF contenant du texte exploitable'
            ),
            'fichiersocr' => new external_value(PARAM_INT, 'Nombre de PDF traités par OCR'),
            'pagesocr' => new external_value(PARAM_INT, 'Nombre de pages reconnues par OCR'),
            'chunks' => new external_value(PARAM_INT, 'Nombre de passages'),
            'tokens' => new external_value(PARAM_INT, 'Tokens d\'embedding'),
        ]);
    }
}
