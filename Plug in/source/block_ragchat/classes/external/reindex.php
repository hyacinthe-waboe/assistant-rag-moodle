<?php
// Fonction externe : (ré)indexer les ressources du cours.
// Récupère les PDF du cours via la File API interne de Moodle, puis les
// envoie au backend /index/{courseid} qui (re)construit l'index vectoriel.

namespace block_ragchat\external;

defined('MOODLE_INTERNAL') || die();

use core_external\external_api;
use core_external\external_function_parameters;
use core_external\external_single_structure;
use core_external\external_value;

class reindex extends external_api {

    public static function execute_parameters(): external_function_parameters {
        return new external_function_parameters([
            'courseid' => new external_value(PARAM_INT, 'Identifiant du cours'),
        ]);
    }

    public static function execute(int $courseid): array {
        global $DB;

        // 1. Validation paramètres + droits (réservé enseignant/gestionnaire).
        $params = self::validate_parameters(self::execute_parameters(), ['courseid' => $courseid]);
        $context = \context_course::instance($params['courseid']);
        self::validate_context($context);
        require_capability('block/ragchat:reindex', $context);

        $backendurl = trim((string) get_config('block_ragchat', 'backendurl'));
        if ($backendurl === '') {
            throw new \moodle_exception('error_noconfig', 'block_ragchat');
        }

        // 2. Collecte des PDF du cours.
        // On récupère les fichiers PDF dans le contexte du cours ET dans tous
        // ses contextes enfants (modules) via le chemin de contexte. On passe
        // par $DB (table officielle {files}) puis par file_storage : aucune
        // lecture brute du système de fichiers, on reste sur les API Moodle.
        $sql = "SELECT f.id
                  FROM {files} f
                  JOIN {context} ctx ON ctx.id = f.contextid
                 WHERE (ctx.id = :ctxid OR " . $DB->sql_like('ctx.path', ':ctxpath') . ")
                   AND f.mimetype = :mime
                   AND f.filename <> '.'";
        $sqlparams = [
            'ctxid'   => $context->id,
            'ctxpath' => $context->path . '/%',
            'mime'    => 'application/pdf',
        ];
        $ids = $DB->get_fieldset_sql($sql, $sqlparams);

        $fs = get_file_storage();
        $tmpdir = make_request_directory();   // dossier temporaire auto-nettoyé
        $postdata = [];
        $index = 0;

        foreach ($ids as $id) {
            $file = $fs->get_file_by_id($id);
            if (!$file || $file->is_directory()) {
                continue;
            }
            // Copie du contenu (stocké en base/filedir) vers un fichier temporaire
            // pour pouvoir le joindre à la requête multipart.
            $abspath = $tmpdir . '/' . $file->get_id() . '_' . $file->get_filename();
            $file->copy_content_to($abspath);
            $postdata['files[' . $index . ']'] =
                curl_file_create($abspath, 'application/pdf', $file->get_filename());
            $index++;
        }

        if ($index === 0) {
            throw new \moodle_exception('error_nopdf', 'block_ragchat');
        }

        // 3. Envoi multipart via curl PHP natif (contourne le blocage localhost de Moodle).
        $ch = curl_init(rtrim($backendurl, '/') . '/index/' . $params['courseid']);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $postdata);
        curl_setopt($ch, CURLOPT_TIMEOUT, 120);
        $raw  = curl_exec($ch);
        $code = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $err  = curl_error($ch);
        curl_close($ch);

        if ($err || $code !== 200) {
            throw new \moodle_exception('error_backend', 'block_ragchat');
        }

        $data = json_decode($raw, true);
        if (!is_array($data) || !isset($data['chunks'])) {
            throw new \moodle_exception('error_backend', 'block_ragchat');
        }

        return [
            'fichiers' => (int) ($data['fichiers'] ?? $index),
            'chunks'   => (int) $data['chunks'],
            'tokens'   => (int) ($data['tokens_embedding'] ?? 0),
        ];
    }

    public static function execute_returns(): external_single_structure {
        return new external_single_structure([
            'fichiers' => new external_value(PARAM_INT, 'Nombre de PDF indexés'),
            'chunks'   => new external_value(PARAM_INT, 'Nombre de chunks générés'),
            'tokens'   => new external_value(PARAM_INT, 'Tokens d\'embedding consommés'),
        ]);
    }
}
