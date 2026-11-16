/**
 * CatSense AI / MeowMind Theme Controller
 * Handles light/dark switching, localStorage persistence, and event dispatch.
 */
(function() {
    function getPreferredTheme() {
        const saved = localStorage.getItem('catsense_theme');
        if (saved) return saved;
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('catsense_theme', theme);
        updateToggleIcons(theme);
        window.dispatchEvent(new CustomEvent('catsense:theme-changed', { detail: { theme } }));
    }

    function updateToggleIcons(theme) {
        const toggles = document.querySelectorAll('.theme-toggle-btn');
        toggles.forEach(btn => {
            const sunIcon = btn.querySelector('.icon-sun');
            const moonIcon = btn.querySelector('.icon-moon');
            if (sunIcon && moonIcon) {
                if (theme === 'dark') {
                    sunIcon.style.display = 'block';
                    moonIcon.style.display = 'none';
                    btn.setAttribute('aria-label', 'Switch to light theme');
                } else {
                    sunIcon.style.display = 'none';
                    moonIcon.style.display = 'block';
                    btn.setAttribute('aria-label', 'Switch to dark theme');
                }
            }
        });
    }

    // Initialize toggle listener
    document.addEventListener('DOMContentLoaded', () => {
        // If page has lock_dark, force dark
        if (document.documentElement.getAttribute('data-lock-dark') === 'true') {
            document.documentElement.setAttribute('data-theme', 'dark');
            return;
        }

        const currentTheme = document.documentElement.getAttribute('data-theme') || getPreferredTheme();
        updateToggleIcons(currentTheme);

        document.querySelectorAll('.theme-toggle-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const active = document.documentElement.getAttribute('data-theme') || 'dark';
                const next = active === 'dark' ? 'light' : 'dark';
                applyTheme(next);
            });
        });
    });

    // Listen to system preference changes if user hasn't explicitly set a theme
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!localStorage.getItem('catsense_theme') && document.documentElement.getAttribute('data-lock-dark') !== 'true') {
            applyTheme(e.matches ? 'dark' : 'light');
        }
    });
})();
