define(['core/ajax', 'core/str'], function(Ajax, Str) {
    'use strict';

    var SVG_BOT = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"' +
        ' stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M12 2a4 4 0 0 1 4 4v1h1a3 3 0 0 1 3 3v6a3 3 0 0 1-3 3H7' +
        'a3 3 0 0 1-3-3v-6a3 3 0 0 1 3-3h1V6a4 4 0 0 1 4-4z"/>' +
        '<circle cx="9" cy="13" r="1" fill="currentColor" stroke="none"/>' +
        '<circle cx="15" cy="13" r="1" fill="currentColor" stroke="none"/>' +
        '<path d="M9 17s1 1 3 1 3-1 3-1"/></svg>';

    var SVG_FILE = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none"' +
        ' stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>' +
        '<polyline points="14 2 14 8 20 8"/></svg>';

    var SVG_SEND = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none"' +
        ' stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<line x1="22" y1="2" x2="11" y2="13"/>' +
        '<polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';

    var SVG_EXPAND = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none"' +
        ' stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/>' +
        '<line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>';

    var SVG_SHRINK = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none"' +
        ' stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/>' +
        '<line x1="10" y1="14" x2="3" y2="21"/><line x1="21" y1="3" x2="14" y2="10"/></svg>';

    var SVG_CLOSE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"' +
        ' stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';

    var SVG_NEWCHAT = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none"' +
        ' stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>' +
        '<path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>';

    var SVG_RELOAD = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none"' +
        ' stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<polyline points="1 4 1 10 7 10"/>' +
        '<path d="M3.51 15a9 9 0 1 0 .49-4.5"/></svg>';

    // ── Nettoyeur markdown ────────────────────────────────────────────────
    function nettoyerMarkdown(texte) {
        return texte
            .replace(/\*\*(.*?)\*\*/g, '$1')
            .replace(/\*(.*?)\*/g, '$1')
            .replace(/^#{1,6}\s+/gm, '')
            .replace(/^[-•]\s+/gm, '- ')
            .replace(/`([^`]+)`/g, '$1');
    }

    // ── État ──────────────────────────────────────────────────────────────
    var panel = null, overlay = null, isOpen = false, isFull = false;
    var welcomeHTML = '';
    var indexPollTimer = null;
    var indexStatusBubble = null;

    // Historique de conversation : max 6 messages (3 questions + 3 réponses)
    // Envoyé au backend à chaque nouvelle question pour donner la mémoire au modèle.
    var historique = [];

    // ── Construit le panel ────────────────────────────────────────────────
    function buildPanel(courseid, canreindex, coursename) {

        overlay = document.createElement('div');
        overlay.className = 'brc_overlay';
        overlay.addEventListener('click', closePanel);
        document.body.appendChild(overlay);

        panel = document.createElement('div');
        panel.className = 'brc_panel';

        panel.innerHTML =
            '<div class="brc_panel_header">' +
            '  <div class="brc_panel_avatar">' + SVG_BOT + '</div>' +
            '  <div class="brc_panel_info">' +
            '    <span class="brc_panel_title">Assistant IA du cours</span>' +
            '    <span class="brc_panel_course">' + escHtml(coursename) + '</span>' +
            '  </div>' +
            '  <div class="brc_panel_actions">' +
            '    <button id="brc-newchat" class="brc_icon_btn" title="Nouvelle conversation">' + SVG_NEWCHAT + '</button>' +
            '    <button id="brc-fullscreen" class="brc_icon_btn" title="Plein ecran">' + SVG_EXPAND + '</button>' +
            '    <button id="brc-close" class="brc_icon_btn" title="Fermer">' + SVG_CLOSE + '</button>' +
            '  </div>' +
            '</div>' +

            '<div id="brc-messages" class="brc_messages">' +
            '  <div class="brc_welcome" id="brc-welcome">' +
            '    <div class="brc_welcome_ico">' + SVG_BOT + '</div>' +
            '    <h3>Comment puis-je vous aider ?</h3>' +
            '    <p>Posez-moi une question sur ce cours.<br>' +
            '    Je reponds uniquement a partir des ressources indexees.</p>' +
            '  </div>' +
            '</div>' +

            '<div class="brc_footer">' +
            (canreindex ?
                '<div class="brc_reindex_bar">' +
                '<button id="brc-reindex" class="brc_reindex_btn">' +
                SVG_RELOAD + ' Reindexer le cours</button></div>' : '') +
            '  <div class="brc_input_wrap">' +
            '    <textarea id="brc-input" class="brc_textarea" rows="1"' +
            '      placeholder="Posez une question sur le cours..."></textarea>' +
            '    <button id="brc-send" class="brc_send_btn" aria-label="Envoyer">' + SVG_SEND + '</button>' +
            '  </div>' +
            '  <p class="brc_hint">Entree pour envoyer - Maj+Entree pour un saut de ligne</p>' +
            '</div>';

        document.body.appendChild(panel);
        welcomeHTML = document.getElementById('brc-welcome').outerHTML;

        document.getElementById('brc-close').addEventListener('click', closePanel);
        document.getElementById('brc-fullscreen').addEventListener('click', toggleFull);
        document.getElementById('brc-newchat').addEventListener('click', resetChat);
        document.getElementById('brc-send').addEventListener('click', function() { envoyer(courseid); });
        document.getElementById('brc-input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); envoyer(courseid); }
        });
        document.getElementById('brc-input').addEventListener('input', function() {
            autoResize(this);
        });

        if (canreindex) {
            var rBtn = document.getElementById('brc-reindex');
            if (rBtn) rBtn.addEventListener('click', function() { reindexer(courseid); });
        }

        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && isOpen) closePanel();
        });
    }

    function escHtml(s) {
        return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function autoResize(el) {
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 140) + 'px';
    }

    // ── Nouveau chat — efface historique et messages ──────────────────────
    function resetChat() {
        var zone = document.getElementById('brc-messages');
        zone.innerHTML = welcomeHTML;
        document.getElementById('brc-input').value = '';
        autoResize(document.getElementById('brc-input'));
        historique = [];  // vide la mémoire de conversation
    }

    // ── Ouvrir / fermer ───────────────────────────────────────────────────
    function openPanel() {
        isOpen = true;
        overlay.classList.add('brc_visible');
        panel.classList.add('brc_open');
        setTimeout(function() {
            var inp = document.getElementById('brc-input');
            if (inp) inp.focus();
        }, 280);
    }

    function closePanel() {
        isOpen = false;
        overlay.classList.remove('brc_visible');
        panel.classList.remove('brc_open');
    }

    function toggleFull() {
        isFull = !isFull;
        panel.classList.toggle('brc_fullscreen', isFull);
        var btn = document.getElementById('brc-fullscreen');
        btn.innerHTML = isFull ? SVG_SHRINK : SVG_EXPAND;
        btn.title = isFull ? 'Reduire' : 'Plein ecran';
    }

    function hideWelcome() {
        var w = document.getElementById('brc-welcome');
        if (w) w.style.display = 'none';
    }

    function scrollBas() {
        var z = document.getElementById('brc-messages');
        if (z) z.scrollTop = z.scrollHeight;
    }

    function bulleUser(texte) {
        var m = document.createElement('div');
        m.className = 'brc_msg brc_msg_user';
        m.innerHTML = '<div class="brc_msg_row">' +
            '<div class="brc_bubble">' + escHtml(texte) + '</div></div>';
        return m;
    }

    function typingIndicator() {
        var m = document.createElement('div');
        m.className = 'brc_msg brc_msg_bot';
        m.id = 'brc-typing';
        m.innerHTML = '<div class="brc_msg_row">' +
            '<div class="brc_av">' + SVG_BOT + '</div>' +
            '<div class="brc_bubble brc_typing"><span></span><span></span><span></span></div>' +
            '</div>';
        return m;
    }

    function bulleBot(texte, passages) {
        passages = passages || [];
        var m = document.createElement('div');
        m.className = 'brc_msg brc_msg_bot';

        // Nettoyage markdown avant affichage
        var texteNet = nettoyerMarkdown(texte);

        var row = '<div class="brc_msg_row">' +
            '<div class="brc_av">' + SVG_BOT + '</div>' +
            '<div class="brc_bubble">' + escHtml(texteNet) + '</div></div>';

        var acc = '';
        if (passages.length > 0) {
            var label = passages.length + ' extrait' + (passages.length > 1 ? 's' : '') +
                        ' utilise' + (passages.length > 1 ? 's' : '');
            var chunksHtml = passages.map(function(p) {
                var nom = (p.source || '').replace(/^.*[/\\]/, '');
                return '<div class="brc_chunk">' +
                    '<div class="brc_chunk_h">' + SVG_FILE + ' ' + escHtml(nom) +
                    ' &mdash; p.' + escHtml(String(p.page)) + '</div>' +
                    '<div class="brc_chunk_b">' + escHtml(p.texte) + '</div></div>';
            }).join('');

            acc = '<div class="brc_acc" style="padding-left:37px">' +
                '<button class="brc_acc_btn">' + SVG_FILE +
                ' <span>' + escHtml(label) + ' - voir</span></button>' +
                '<div class="brc_chunks" style="display:none">' + chunksHtml + '</div></div>';
        }

        m.innerHTML = row + acc;

        var boutonExtraits = m.querySelector('.brc_acc_btn');
        if (boutonExtraits) {
            boutonExtraits.addEventListener('click', function() {
                var contenu = this.nextElementSibling;
                var ouvert = contenu.style.display !== 'none';
                contenu.style.display = ouvert ? 'none' : 'flex';
                this.querySelector('span').textContent =
                    label + ' - ' + (ouvert ? 'voir' : 'masquer');
            });
        }

        return m;
    }

    function afficherStatutIndexation(texte) {
        hideWelcome();
        var zone = document.getElementById('brc-messages');
        if (!indexStatusBubble || !document.body.contains(indexStatusBubble)) {
            indexStatusBubble = bulleBot(texte, []);
            indexStatusBubble.classList.add('brc_index_status');
            zone.appendChild(indexStatusBubble);
        } else {
            indexStatusBubble.querySelector('.brc_bubble').textContent = texte;
        }
        scrollBas();
    }

    function arreterSuiviIndexation() {
        if (indexPollTimer) {
            window.clearTimeout(indexPollTimer);
            indexPollTimer = null;
        }
    }

    function suivreIndexation(courseid, afficherSiInactive) {
        arreterSuiviIndexation();
        Ajax.call([{
            methodname: 'block_ragchat_index_status',
            args: { courseid: courseid },
        }])[0]
        .then(function(r) {
            var btn = document.getElementById('brc-reindex');
            if (r.status === 'queued' || r.status === 'running') {
                if (btn) btn.disabled = true;
                afficherStatutIndexation(
                    'Indexation en cours : ' + r.progress + ' %' +
                    (r.message ? ' - ' + r.message : '')
                );
                indexPollTimer = window.setTimeout(function() {
                    suivreIndexation(courseid, true);
                }, 3000);
                return;
            }

            if (btn) btn.disabled = false;
            if (r.status === 'completed' && afficherSiInactive) {
                afficherStatutIndexation(
                    'Indexation terminée : ' + r.fichiers + ' PDF, ' +
                    r.chunks + ' passages.'
                );
            } else if (r.status === 'failed' && afficherSiInactive) {
                afficherStatutIndexation(
                    'Échec de l’indexation' + (r.error ? ' : ' + r.error : '.')
                );
            } else if (afficherSiInactive) {
                afficherStatutIndexation('Aucune indexation en cours.');
            }
        })
        .catch(function() {
            var btn = document.getElementById('brc-reindex');
            if (btn) btn.disabled = false;
            if (afficherSiInactive) {
                afficherStatutIndexation(
                    'Impossible de consulter la progression de l’indexation.'
                );
            }
        });
    }

    // ── Envoi ─────────────────────────────────────────────────────────────
    function envoyer(courseid) {
        var inp  = document.getElementById('brc-input');
        var send = document.getElementById('brc-send');
        var q    = inp.value.trim();
        if (!q) return;

        hideWelcome();
        var zone = document.getElementById('brc-messages');
        zone.appendChild(bulleUser(q));
        inp.value = ''; autoResize(inp); send.disabled = true;

        var t = typingIndicator();
        zone.appendChild(t); scrollBas();

        // Historique envoyé sans inclure la question courante
        var historiqueEnvoi = historique.slice();

        Ajax.call([{
            methodname: 'block_ragchat_ask',
            args: {
                courseid: courseid,
                question: q,
                history:  JSON.stringify(historiqueEnvoi),
            },
        }])[0]
        .then(function(r) {
            t.remove();
            var reponse = r.reponse;
            zone.appendChild(bulleBot(reponse, r.passages || []));
            scrollBas();

            // Mémorise l'échange pour la prochaine question
            historique.push({role: 'user',      content: q});
            historique.push({role: 'assistant', content: reponse});

            // Cap à 6 messages (3 échanges)
            if (historique.length > 6) {
                historique = historique.slice(-6);
            }
        })
        .catch(function() {
            t.remove();
            Str.get_string('error_backend', 'block_ragchat').done(function(msg) {
                zone.appendChild(bulleBot(msg, [])); scrollBas();
            });
        })
        .always(function() { send.disabled = false; });
    }

    // ── Reindexation ─────────────────────────────────────────────────────
    function reindexer(courseid) {
        var btn = document.getElementById('brc-reindex');
        if (btn) btn.disabled = true;
        afficherStatutIndexation('Envoi des PDF au service d’indexation...');

        Ajax.call([{
            methodname: 'block_ragchat_reindex',
            args: { courseid: courseid },
        }])[0]
        .then(function(r) {
            afficherStatutIndexation(
                r.fichiers + ' PDF reçus. Préparation de l’indexation...'
            );
            suivreIndexation(courseid, true);
        })
        .catch(function() {
            afficherStatutIndexation(
                'Le lancement a échoué. Vérifiez que le service IA est disponible.'
            );
            if (btn) btn.disabled = false;
        });
    }

    // ── Init ─────────────────────────────────────────────────────────────
    return {
        init: function(courseid, canreindex, coursename) {
            buildPanel(courseid, canreindex, coursename || '');
            if (canreindex) {
                suivreIndexation(courseid, false);
            }

            var openBtn = document.getElementById('brc-open');
            if (openBtn) {
                openBtn.addEventListener('click', openPanel);
            }
        }
    };
});
