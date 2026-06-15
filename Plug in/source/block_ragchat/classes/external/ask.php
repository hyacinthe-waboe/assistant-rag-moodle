<?php
namespace block_ragchat\external;

defined('MOODLE_INTERNAL') || die();

require_once(__DIR__ . '/../../lib.php');

use core_external\external_api;
use core_external\external_function_parameters;
use core_external\external_single_structure;
use core_external\external_multiple_structure;
use core_external\external_value;

class ask extends external_api {

    public static function execute_parameters(): external_function_parameters {
        return new external_function_parameters([
            'courseid' => new external_value(PARAM_INT,  'Identifiant du cours'),
            'question' => new external_value(PARAM_TEXT, 'Question de l\'étudiant'),
            'history'  => new external_value(PARAM_RAW,  'Historique JSON', VALUE_DEFAULT, '[]'),
        ]);
    }

    public static function execute(int $courseid, string $question, string $history = '[]'): array {
        $params = self::validate_parameters(self::execute_parameters(), [
            'courseid' => $courseid,
            'question' => $question,
            'history'  => $history,
        ]);

        $context = \context_course::instance($params['courseid']);
        self::validate_context($context);
        require_capability('block/ragchat:use', $context);

        $backendurl = trim((string) get_config('block_ragchat', 'backendurl'));
        if ($backendurl === '') {
            return ['reponse' => get_string('error_noconfig', 'block_ragchat'),
                    'sources' => [], 'passages' => [], 'tokens' => 0];
        }

        // Décode et valide l'historique JSON
        $historyArray = json_decode($params['history'], true);
        if (!is_array($historyArray)) {
            $historyArray = [];
        }

        $payload = json_encode([
            'course_id' => (string) $params['courseid'],
            'question'  => $params['question'],
            'history'   => $historyArray,
        ]);

        $ch = curl_init(rtrim($backendurl, '/') . '/ask');
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
        curl_setopt($ch, CURLOPT_HTTPHEADER, block_ragchat_backend_headers(true));
        curl_setopt($ch, CURLOPT_TIMEOUT, 120);
        $raw  = curl_exec($ch);
        $code = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $err  = curl_error($ch);
        curl_close($ch);

        if ($code === 404) {
            return ['reponse' => get_string('error_noindex', 'block_ragchat'),
                    'sources' => [], 'passages' => [], 'tokens' => 0];
        }

        if ($code === 422) {
            return ['reponse' => get_string('error_invalidquestion', 'block_ragchat'),
                    'sources' => [], 'passages' => [], 'tokens' => 0];
        }

        if ($err || $code !== 200) {
            return ['reponse' => get_string('error_backend', 'block_ragchat'),
                    'sources' => [], 'passages' => [], 'tokens' => 0];
        }

        $data = json_decode($raw, true);
        if (!is_array($data) || !isset($data['reponse'])) {
            return ['reponse' => get_string('error_backend', 'block_ragchat'),
                    'sources' => [], 'passages' => [], 'tokens' => 0];
        }

        $passages = [];
        foreach (($data['passages'] ?? []) as $p) {
            $passages[] = [
                'source' => (string) ($p['source'] ?? ''),
                'page'   => (string) ($p['page']   ?? ''),
                'texte'  => (string) ($p['texte']  ?? ''),
            ];
        }

        return [
            'reponse'  => $data['reponse'],
            'sources'  => $data['sources']  ?? [],
            'passages' => $passages,
            'tokens'   => (int) ($data['tokens'] ?? 0),
        ];
    }

    public static function execute_returns(): external_single_structure {
        return new external_single_structure([
            'reponse'  => new external_value(PARAM_RAW,  'Réponse'),
            'sources'  => new external_multiple_structure(
                new external_value(PARAM_TEXT, 'Source')
            ),
            'passages' => new external_multiple_structure(
                new external_single_structure([
                    'source' => new external_value(PARAM_TEXT, 'Fichier source'),
                    'page'   => new external_value(PARAM_TEXT, 'Page'),
                    'texte'  => new external_value(PARAM_RAW,  'Extrait'),
                ])
            ),
            'tokens'   => new external_value(PARAM_INT, 'Tokens'),
        ]);
    }
}
