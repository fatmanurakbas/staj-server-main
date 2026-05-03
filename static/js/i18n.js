/**
 * Internationalization (i18n) System for LiverAId
 * Supports Turkish and English languages
 */

class I18nManager {
    constructor() {
        this.currentLanguage = localStorage.getItem('liveraid-language') || 'tr';
        this.translations = {};
        this.loadedLanguages = new Set();
    }

    // Load language file
    async loadLanguage(lang) {
        if (this.loadedLanguages.has(lang)) {
            return this.translations[lang];
        }

        try {
            const response = await fetch(`/static/js/languages/${lang}.json`);
            if (!response.ok) {
                throw new Error(`Failed to load language ${lang}`);
            }
            
            const translations = await response.json();
            this.translations[lang] = translations;
            this.loadedLanguages.add(lang);
            
            console.log(`✅ Language ${lang} loaded successfully`);
            return translations;
            
        } catch (error) {
            console.error(`❌ Failed to load language ${lang}:`, error);
            // Fallback to Turkish if English fails, or vice versa
            if (lang !== 'tr') {
                return await this.loadLanguage('tr');
            }
            throw error;
        }
    }

    // Get translation for a key path (e.g., "form.age")
    t(keyPath, params = {}) {
        const currentTranslations = this.translations[this.currentLanguage];
        if (!currentTranslations) {
            console.warn(`No translations loaded for language: ${this.currentLanguage}`);
            return keyPath;
        }

        // Navigate through nested object
        const keys = keyPath.split('.');
        let value = currentTranslations;
        
        for (const key of keys) {
            if (value && typeof value === 'object' && key in value) {
                value = value[key];
            } else {
                console.warn(`Translation key not found: ${keyPath} in ${this.currentLanguage}`);
                return keyPath;
            }
        }

        // Handle parameter substitution
        if (typeof value === 'string' && Object.keys(params).length > 0) {
            return value.replace(/\{\{(\w+)\}\}/g, (match, param) => {
                return params[param] || match;
            });
        }

        return value;
    }

    // Change language
    async setLanguage(lang) {
        if (!['tr', 'en'].includes(lang)) {
            console.warn(`Unsupported language: ${lang}. Using Turkish as fallback.`);
            lang = 'tr';
        }

        // Load language if not already loaded
        await this.loadLanguage(lang);
        
        this.currentLanguage = lang;
        localStorage.setItem('liveraid-language', lang);
        
        // Update the page
        this.updatePageLanguage();
        
        // Trigger custom event for other components
        document.dispatchEvent(new CustomEvent('languageChanged', { 
            detail: { language: lang } 
        }));
        
        console.log(`🌐 Language changed to: ${lang}`);
    }

    // Get current language
    getCurrentLanguage() {
        return this.currentLanguage;
    }

    // Update all translatable elements on the page
    updatePageLanguage() {
        // Update elements with data-i18n attribute
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = this.t(key);
            
            if (element.tagName === 'INPUT' && (element.type === 'button' || element.type === 'submit')) {
                element.value = translation;
            } else if (element.tagName === 'INPUT' && element.hasAttribute('placeholder')) {
                element.placeholder = translation;
            } else {
                element.textContent = translation;
            }
        });

        // Update elements with data-i18n-html attribute (for HTML content)
        document.querySelectorAll('[data-i18n-html]').forEach(element => {
            const key = element.getAttribute('data-i18n-html');
            const translation = this.t(key);
            element.innerHTML = translation;
        });

        // Update page title
        const titleElement = document.querySelector('title');
        if (titleElement) {
            titleElement.textContent = this.t('app.title');
        }

        // Update speech recognition language
        this.updateSpeechRecognitionLanguage();
    }

    // Update speech recognition language
    updateSpeechRecognitionLanguage() {
        if (window.recognition && window.recognition.lang) {
            const speechLang = this.currentLanguage === 'tr' ? 'tr-TR' : 'en-US';
            window.recognition.lang = speechLang;
            console.log(`🎤 Speech recognition language updated to: ${speechLang}`);
        }
    }

    // Initialize the i18n system
    async init() {
        try {
            // Load current language
            await this.loadLanguage(this.currentLanguage);
            
            // Update page on load
            this.updatePageLanguage();
            
            // Create language selector if it doesn't exist
            this.createLanguageSelector();
            
            console.log(`🌐 i18n system initialized with language: ${this.currentLanguage}`);
            
        } catch (error) {
            console.error('❌ Failed to initialize i18n system:', error);
        }
    }

    // Create language selector dropdown
    createLanguageSelector() {
        // Check if selector already exists
        if (document.getElementById('languageSelector')) {
            return;
        }

        const selector = document.createElement('div');
        selector.id = 'languageSelector';
        selector.className = 'dropdown';
        selector.innerHTML = `
            <button class="btn btn-outline-light btn-sm dropdown-toggle language-btn" type="button" 
                    data-bs-toggle="dropdown" aria-expanded="false">
                <i class="fas fa-globe me-2"></i>
                <span id="currentLanguageFlag">${this.currentLanguage === 'tr' ? '🇹🇷' : '🇺🇸'}</span>
                <span id="currentLanguage" class="ms-1">${this.currentLanguage.toUpperCase()}</span>
            </button>
            <ul class="dropdown-menu dropdown-menu-end language-dropdown">
                <li><a class="dropdown-item language-option ${this.currentLanguage === 'tr' ? 'active' : ''}" 
                       href="#" onclick="i18n.setLanguage('tr')">
                    <span class="flag-icon">🇹🇷</span>
                    <span class="lang-name">Türkçe</span>
                    <small class="lang-code text-muted">TR</small>
                </a></li>
                <li><a class="dropdown-item language-option ${this.currentLanguage === 'en' ? 'active' : ''}" 
                       href="#" onclick="i18n.setLanguage('en')">
                    <span class="flag-icon">🇺🇸</span>
                    <span class="lang-name">English</span>
                    <small class="lang-code text-muted">EN</small>
                </a></li>
            </ul>
        `;

        // Add to navigation bar
        const navbar = document.querySelector('.navbar-nav');
        if (navbar) {
            const li = document.createElement('li');
            li.className = 'nav-item';
            li.appendChild(selector);
            navbar.appendChild(li);
        }

        // Add custom CSS for better styling
        this.addLanguageSelectorCSS();
    }

    // Update language selector display
    updateLanguageSelector() {
        const currentLangSpan = document.getElementById('currentLanguage');
        const currentFlagSpan = document.getElementById('currentLanguageFlag');
        
        if (currentLangSpan) {
            currentLangSpan.textContent = this.currentLanguage.toUpperCase();
        }
        
        if (currentFlagSpan) {
            currentFlagSpan.textContent = this.currentLanguage === 'tr' ? '🇹🇷' : '🇺🇸';
        }

        // Update active state in dropdown
        document.querySelectorAll('.language-option').forEach(option => {
            option.classList.remove('active');
            if ((this.currentLanguage === 'tr' && option.onclick.toString().includes("'tr'")) ||
                (this.currentLanguage === 'en' && option.onclick.toString().includes("'en'"))) {
                option.classList.add('active');
            }
        });
    }

    // Add custom CSS for language selector
    addLanguageSelectorCSS() {
        const style = document.createElement('style');
        style.id = 'language-selector-styles';
        style.textContent = `
            .language-btn {
                border: 1px solid rgba(255, 255, 255, 0.3) !important;
                background: rgba(255, 255, 255, 0.1) !important;
                color: white !important;
                transition: all 0.3s ease;
                font-size: 0.875rem;
                padding: 0.375rem 0.75rem;
                min-width: 80px;
            }
            
            .language-btn:hover {
                background: rgba(255, 255, 255, 0.2) !important;
                border-color: rgba(255, 255, 255, 0.5) !important;
                color: white !important;
                transform: translateY(-1px);
            }
            
            .language-btn:focus {
                box-shadow: 0 0 0 0.2rem rgba(255, 255, 255, 0.25) !important;
                color: white !important;
            }
            
            .language-dropdown {
                border: none;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
                border-radius: 8px;
                overflow: hidden;
                margin-top: 0.5rem;
                min-width: 160px;
            }
            
            .language-option {
                padding: 0.75rem 1rem !important;
                display: flex !important;
                align-items: center !important;
                gap: 0.75rem;
                transition: all 0.2s ease;
                border: none !important;
            }
            
            .language-option:hover {
                background: linear-gradient(135deg, #007bff, #0056b3) !important;
                color: white !important;
                transform: translateX(3px);
            }
            
            .language-option.active {
                background: linear-gradient(135deg, #28a745, #1e7e34) !important;
                color: white !important;
                font-weight: 500;
            }
            
            .language-option.active:hover {
                background: linear-gradient(135deg, #34ce57, #28a745) !important;
            }
            
            .flag-icon {
                font-size: 1.2rem;
                width: 24px;
                text-align: center;
            }
            
            .lang-name {
                flex: 1;
                font-weight: 500;
            }
            
            .lang-code {
                font-size: 0.75rem;
                opacity: 0.8;
            }
            
            .language-option:hover .lang-code {
                opacity: 1;
                color: rgba(255, 255, 255, 0.8) !important;
            }
            
            .language-option.active .lang-code {
                color: rgba(255, 255, 255, 0.9) !important;
            }
        `;
        
        // Remove existing styles if any
        const existingStyles = document.getElementById('language-selector-styles');
        if (existingStyles) {
            existingStyles.remove();
        }
        
        document.head.appendChild(style);
    }
}

// Create global i18n instance
window.i18n = new I18nManager();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async function() {
    await window.i18n.init();
});

// Listen for language changes to update language selector
document.addEventListener('languageChanged', function(event) {
    window.i18n.updateLanguageSelector();
});
