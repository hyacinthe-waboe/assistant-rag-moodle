<?php
defined('MOODLE_INTERNAL') || die();

class block_ragchat extends block_base {

    public function init() {
        $this->title = get_string('pluginname', 'block_ragchat');
    }

    public function applicable_formats() {
        return ['course-view' => true, 'site' => false, 'my' => false];
    }

    public function instance_allow_multiple() {
        return false;
    }

    public function get_content() {
        global $PAGE;

        if ($this->content !== null) {
            return $this->content;
        }

        $this->content         = new stdClass();
        $this->content->footer = '';

        if (!$this->page->course || $this->page->course->id == SITEID) {
            $this->content->text = get_string('coursecontextonly', 'block_ragchat');
            return $this->content;
        }

        $courseid   = (int) $this->page->course->id;
        $context    = context_course::instance($courseid);
        $canreindex = has_capability('block/ragchat:reindex', $context);

        if (!has_capability('block/ragchat:use', $context)) {
            $this->content->text = '';
            return $this->content;
        }

        $icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                      stroke="currentColor" stroke-width="1.8"
                      stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2a4 4 0 0 1 4 4v1h1a3 3 0 0 1 3 3v6a3 3 0 0 1-3 3H7
                             a3 3 0 0 1-3-3v-6a3 3 0 0 1 3-3h1V6a4 4 0 0 1 4-4z"/>
                    <circle cx="9" cy="13" r="1" fill="currentColor" stroke="none"/>
                    <circle cx="15" cy="13" r="1" fill="currentColor" stroke="none"/>
                    <path d="M9 17s1 1 3 1 3-1 3-1"/>
                 </svg>';

        // Le contenu HTML reste simple ; toute la présentation est dans styles.css.
        $html  = '<div class="brc_launcher">';
        $html .= '<button id="brc-open" class="brc_open_btn" aria-label="Ouvrir l\'assistant IA">';
        $html .=   '<span class="brc_open_ico">' . $icon . '</span>';
        $html .=   '<span class="brc_open_label">';
        $html .=     '<span class="brc_open_title">' . get_string('pluginname', 'block_ragchat') . '</span>';
        $html .=     '<span class="brc_open_sub">Ancré sur les ressources du cours</span>';
        $html .=   '</span>';
        $html .=   '<span class="brc_open_arrow">';
        $html .=     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                           stroke="currentColor" stroke-width="2"
                           stroke-linecap="round" stroke-linejoin="round">
                         <polyline points="9 18 15 12 9 6"/>
                      </svg>';
        $html .=   '</span>';
        $html .= '</button>';
        $html .= '</div>';

        $this->content->text = $html;

        $coursename = format_string($this->page->course->fullname);
        $PAGE->requires->js_call_amd('block_ragchat/chat', 'init',
            [$courseid, $canreindex, $coursename]);

        return $this->content;
    }

    public function has_config() {
        return true;
    }
}
