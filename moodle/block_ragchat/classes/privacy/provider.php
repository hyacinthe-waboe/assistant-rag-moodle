<?php
// Fournisseur de confidentialité (RGPD).
// Le plugin ne stocke pas les conversations dans Moodle. Il transmet au
// backend les ressources à indexer, la question et l'historique court.

namespace block_ragchat\privacy;

defined('MOODLE_INTERNAL') || die();

use core_privacy\local\metadata\collection;

class provider implements \core_privacy\local\metadata\provider {

    public static function get_metadata(collection $collection): collection {
        $collection->add_external_location_link(
            'ragbackend',
            [
                'courseid' => 'privacy:metadata:ragbackend:courseid',
                'documents' => 'privacy:metadata:ragbackend:documents',
                'question' => 'privacy:metadata:ragbackend:question',
                'history' => 'privacy:metadata:ragbackend:history',
            ],
            'privacy:metadata:ragbackend'
        );
        return $collection;
    }
}
