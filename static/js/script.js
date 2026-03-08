/**
 * static/js/script.js
 * Loaded by base.html on every page.
 * Features: chart lightbox, copy-to-clipboard toast, auto-dismiss alerts.
 * Table sorting is handled inline in results.html (simpler per-page control).
 */

document.addEventListener('DOMContentLoaded', function () {
    _initLightbox();
    _initCopyToast();
});

/* ── CHART LIGHTBOX ─────────────────────────────────────────────
   Adds click-to-zoom to any <img> with class .js-chart-zoom.
   Creates overlay on first call, reuses it afterwards.
   ──────────────────────────────────────────────────────────────── */
function _initLightbox() {
    // Build overlay once
    const overlay = document.createElement('div');
    overlay.id = 'lb-overlay';

    const img = document.createElement('img');
    img.id = 'lb-img'; img.alt = 'Chart full view';

    const closeBtn = document.createElement('button');
    closeBtn.id = 'lb-close'; closeBtn.innerHTML = '&times;';
    closeBtn.setAttribute('aria-label', 'Close');

    overlay.appendChild(closeBtn);
    overlay.appendChild(img);
    document.body.appendChild(overlay);

    function open(src) {
        img.src = src;
        overlay.classList.add('on');
        document.body.style.overflow = 'hidden';
    }
    function close() {
        overlay.classList.remove('on');
        document.body.style.overflow = '';
        img.src = '';
    }

    // Attach to existing .js-chart-zoom images
    document.querySelectorAll('img.js-chart-zoom').forEach(el => {
        el.addEventListener('click', () => open(el.src));
    });

    overlay.addEventListener('click', e => { if (e.target === overlay || e.target === closeBtn) close(); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && overlay.classList.contains('on')) close(); });
}

/* ── COPY TO CLIPBOARD TOAST ────────────────────────────────────
   Any element with data-copy="true" copies its text on click
   and shows a brief toast notification.
   ──────────────────────────────────────────────────────────────── */
function _initCopyToast() {
    const toast = document.createElement('div');
    toast.id = 'copy-toast';
    toast.textContent = '✓ Copied';
    document.body.appendChild(toast);

    document.querySelectorAll('[data-copy="true"]').forEach(el => {
        el.style.cursor = 'pointer';
        el.addEventListener('click', () => copyToClipboard(el.innerText.trim()));
    });
}

/**
 * Copy text to clipboard and show a toast.
 * Can also be called directly from template JS.
 * @param {string} text
 */
function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).catch(console.error);
    } else {
        const ta = document.createElement('textarea');
        ta.value = text; ta.style.cssText = 'position:fixed;opacity:0;';
        document.body.appendChild(ta); ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    }
    const toast = document.getElementById('copy-toast');
    if (toast) {
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 2000);
    }
}