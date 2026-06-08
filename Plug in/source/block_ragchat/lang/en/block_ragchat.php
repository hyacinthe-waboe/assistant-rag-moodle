<?php
// English language strings for block_ragchat (default language, required by Moodle).

defined('MOODLE_INTERNAL') || die();

$string['pluginname'] = 'Course AI Assistant';

// Capabilities.
$string['ragchat:addinstance'] = 'Add an AI Assistant block';
$string['ragchat:use'] = 'Use the course AI assistant';
$string['ragchat:reindex'] = 'Reindex course resources';

// Settings.
$string['backendurl'] = 'RAG backend URL';
$string['backendurl_desc'] = 'Address of the RAG service (FastAPI). Example: http://localhost:8000. '
    . 'This URL stays server-side and is never sent to the browser.';

// Interface.
$string['askplaceholder'] = 'Ask a question about the course…';
$string['send'] = 'Send';
$string['reindex'] = 'Reindex course';
$string['coursecontextonly'] = 'This assistant is only available inside a course.';

// Errors.
$string['error_noconfig'] = 'The RAG backend is not configured. Please contact the administrator.';
$string['error_backend'] = 'The AI service is temporarily unavailable.';
$string['error_nopdf'] = 'No PDF file found to index in this course.';

// Privacy (GDPR).
$string['privacy:metadata:ragbackend'] = 'In order to answer, the question is sent to a RAG service '
    . 'that processes it against the course resources.';
$string['privacy:metadata:ragbackend:question'] = 'The question entered by the user, '
    . 'sent to the service to generate an answer.';
