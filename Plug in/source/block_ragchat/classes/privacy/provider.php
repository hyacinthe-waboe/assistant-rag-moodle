<?php
// Fournisseur de confidentialité (RGPD).
// Ce plugin ne STOCKE aucune donnée personnelle dans Moodle, mais il
// TRANSMET la question de l'étudiant à un backend externe. On le déclare
// explicitement : c'est une exigence RGPD et un point central du pilote
// "IA souveraine" (en déploiement Ollama local, le backend est interne UT2J).

namespace block_ragchat\privacy;

defined('MOODLE_INTERNAL') || die();

use core_privacy\local\metadata\collection;

class provider implements \core_privacy\local\metadata\provider {

    public static function get_metadata(collection $collection): collection {
        $collection->add_external_location_link(
            'ragbackend',
            ['question' => 'privacy:metadata:ragbackend:question'],
            'privacy:metadata:ragbackend'
        );
        return $collection;
    }
}
