/* =====================================================
   MeowMind AI — Main JS v3.0
   Vanilla JS | No Frameworks | Flask-integrated
   ===================================================== */

// ────────────────────────────────────────────────────
// PAGE TRANSITION OVERLAY (Light SaaS Style)
// ────────────────────────────────────────────────────
function initPageTransition() {
    const overlay = document.createElement('div');
    overlay.id = 'page-overlay';
    overlay.style.cssText = `
        position: fixed; inset: 0; background: var(--bg-base, #0B0F17); z-index: 99999;
        pointer-events: none; opacity: 0;
        transition: opacity 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    `;
    document.body.appendChild(overlay);

    window.addEventListener('load', () => {
        overlay.style.opacity = '0';
    });

    document.querySelectorAll('a[href]:not([target]):not([href^="#"]):not([href^="mailto"])').forEach(link => {
        link.addEventListener('click', e => {
            const href = link.getAttribute('href');
            if (!href || href.startsWith('http') || href.startsWith('//')) return;
            e.preventDefault();
            overlay.style.opacity = '1';
            overlay.style.pointerEvents = 'all';
            setTimeout(() => { window.location.href = href; }, 300);
        });
    });
}

// ────────────────────────────────────────────────────
// NAVBAR & MOBILE MENU
// ────────────────────────────────────────────────────
function initNavbar() {
    const navbar = document.querySelector('.navbar');
    const toggle = document.querySelector('.nav-toggle');
    const links = document.querySelector('.nav-links');

    if (!navbar) return;

    // Scroll effect
    const handleScroll = () => {
        if (window.scrollY > 20) {
            navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.05)';
        } else {
            navbar.style.boxShadow = 'none';
        }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();

    // Mobile menu toggle
    if (toggle && links) {
        toggle.addEventListener('click', () => {
            const isOpen = links.classList.toggle('open');
            toggle.classList.toggle('active', isOpen);
            toggle.setAttribute('aria-expanded', isOpen);
        });

        // Close when clicking outside
        document.addEventListener('click', e => {
            if (!navbar.contains(e.target)) {
                links.classList.remove('open');
                toggle.classList.remove('active');
            }
        });
    }
}

// ────────────────────────────────────────────────────
// SCROLL REVEAL (Intersection Observer)
// ────────────────────────────────────────────────────
function initScrollReveal() {
    const io = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                io.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -20px 0px' });

    document.querySelectorAll('.anim-up, .anim-scale, .reveal').forEach(el => {
        // Only observe elements that are actually in the DOM
        io.observe(el);
    });
}

// ────────────────────────────────────────────────────
// TOAST NOTIFICATIONS (For Flask Flash Messages)
// ────────────────────────────────────────────────────
function showToast(message, type = 'success') {
    let toast = document.querySelector('.toast');
    if (!toast) {
        toast = document.createElement('div');
        document.body.appendChild(toast);
    }
    
    // Modern SaaS toast styling overrides
    toast.className = `toast ${type === 'error' ? 'error' : 'success'}`;
    const icon = type === 'success' ? 
        `<svg style="width:20px;height:20px;color:#10b981;" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>` : 
        `<svg style="width:20px;height:20px;color:#ef4444;" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>`;
        
    toast.innerHTML = `<div style="display:flex;align-items:center;gap:10px;">${icon}<span>${message}</span></div>`;
    
    requestAnimationFrame(() => toast.classList.add('show'));
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => toast.classList.remove('show'), 4000);
}
window.showToast = showToast;

// ────────────────────────────────────────────────────
// DRAG & DROP FILE UPLOAD (Works on both forms)
// ────────────────────────────────────────────────────
function initDragAndDrop() {
    // Target any element with the 'dropzone' class
    const dropzones = document.querySelectorAll('.dropzone');
    
    dropzones.forEach(zone => {
        const fileInput = zone.querySelector('.hidden-file-input');
        const textDisplay = zone.querySelector('.dropzone-text');
        
        if (!fileInput) return;

        // Prevent default browser behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            zone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        // Highlight dropzone when dragging over
        ['dragenter', 'dragover'].forEach(eventName => {
            zone.addEventListener(eventName, () => {
                zone.style.borderColor = 'var(--c-primary)';
                zone.style.backgroundColor = '#eff6ff';
            }, false);
        });

        // Unhighlight when leaving
        ['dragleave', 'drop'].forEach(eventName => {
            zone.addEventListener(eventName, () => {
                zone.style.borderColor = '';
                zone.style.backgroundColor = '';
            }, false);
        });

        // Handle dropped files
        zone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files; // Assign files to hidden input
                updateFileNameDisplay(files[0].name);
            }
        });

        // Handle click to browse
        fileInput.addEventListener('change', function() {
            if (this.files && this.files.length > 0) {
                updateFileNameDisplay(this.files[0].name);
            }
        });

        function updateFileNameDisplay(name) {
            if (textDisplay) {
                textDisplay.innerHTML = `<span style="color:var(--c-primary);font-weight:700;">✅ Selected: ${name}</span>`;
            }
        }
    });

    // Handle form submissions to show loading state
    const forms = document.querySelectorAll('#demoForm');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const btn = this.querySelector('button[type="submit"]');
            if (btn) {
                btn.innerHTML = 'Analyzing Data... ⏳';
                btn.style.opacity = '0.8';
                btn.style.cursor = 'wait';
                // Note: We don't disable the button completely, otherwise Flask won't receive the POST
                btn.style.pointerEvents = 'none'; 
            }
        });
    });
}

// ────────────────────────────────────────────────────
// PROGRESS BARS (Dashboard Emotion Distribution)
// ────────────────────────────────────────────────────
function initProgressBars() {
    const io = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // The width is already set via inline style in the template
                // This just triggers a reflow to make the CSS transition happen
                const el = entry.target;
                const width = el.style.width;
                el.style.width = '0';
                setTimeout(() => { el.style.width = width; }, 100);
                io.unobserve(el);
            }
        });
    }, { threshold: 0.2 });

    document.querySelectorAll('.progress-fill').forEach(el => io.observe(el));
}

// ────────────────────────────────────────────────────
// SMOOTH SCROLL FOR ANCHOR LINKS
// ────────────────────────────────────────────────────
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(a => {
        a.addEventListener('click', e => {
            const targetId = a.getAttribute('href');
            if (targetId === '#') return;
            
            const target = document.querySelector(targetId);
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// ────────────────────────────────────────────────────
// PASSWORD TOGGLE (Login/Register)
// ────────────────────────────────────────────────────
function togglePassword(inputId) {
    const input = document.getElementById(inputId);
    if (input) {
        input.type = input.type === 'password' ? 'text' : 'password';
    }
}
window.togglePassword = togglePassword;

// ────────────────────────────────────────────────────
// MAIN INITIALIZATION
// ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    
    // Global initializations
    initPageTransition();
    initNavbar();
    initScrollReveal();
    initSmoothScroll();
    initDragAndDrop();

    // Page specific initializations based on data-page attribute
    const pageId = document.body.getAttribute('data-page');
    
    if (pageId === 'dashboard') {
        initProgressBars();
    }
    
    if (pageId === 'profile') {
        // Modal logic is handled directly in the profile.html block
    }
});