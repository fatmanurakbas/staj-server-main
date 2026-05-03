"""
Server-side internationalization module for LiverAId
Handles translations for Turkish and English languages
"""

import json
import os
from typing import Dict, Any, Optional
from flask import session


class I18nManager:
    """Server-side internationalization manager"""
    
    def __init__(self, app=None):
        self.translations = {}
        self.default_language = 'tr'
        self.supported_languages = ['tr', 'en']
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize with Flask app"""
        self.app = app
        self.load_all_translations()
        
        # Add template globals
        app.jinja_env.globals['get_translation'] = self.get_translation
        app.jinja_env.globals['get_current_language'] = self.get_current_language
        app.jinja_env.globals['t'] = self.t
        
        # Add template filters
        app.jinja_env.filters['translate'] = self.translate_filter
        
    def load_all_translations(self):
        """Load all translation files"""
        translations_dir = os.path.join(
            os.path.dirname(__file__), 
            'static', 'js', 'languages'
        )
        
        for lang in self.supported_languages:
            file_path = os.path.join(translations_dir, f'{lang}.json')
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.translations[lang] = json.load(f)
                print(f"✅ Loaded translations for {lang}")
            except FileNotFoundError:
                print(f"❌ Translation file not found: {file_path}")
                self.translations[lang] = {}
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in {file_path}: {e}")
                self.translations[lang] = {}
    
    def get_current_language(self) -> str:
        """Get current language from session or use default"""
        return session.get('language', self.default_language)
    
    def set_language(self, language: str) -> bool:
        """Set current language in session"""
        if language in self.supported_languages:
            session['language'] = language
            return True
        return False
    
    def get_translation(self, key_path: str, language: Optional[str] = None, **kwargs) -> str:
        """
        Get translation for a key path (e.g., "form.age")
        
        Args:
            key_path: Dot-separated path to translation key
            language: Language code (defaults to current language)
            **kwargs: Parameters for string formatting
            
        Returns:
            Translated string or original key if not found
        """
        if language is None:
            language = self.get_current_language()
        
        if language not in self.translations:
            language = self.default_language
        
        # Navigate through nested object
        keys = key_path.split('.')
        value = self.translations.get(language, {})
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                # Fallback to default language
                if language != self.default_language:
                    return self.get_translation(key_path, self.default_language, **kwargs)
                return key_path
        
        # Handle string formatting
        if isinstance(value, str) and kwargs:
            try:
                return value.format(**kwargs)
            except (KeyError, ValueError):
                return value
        
        return value
    
    def t(self, key_path: str, **kwargs) -> str:
        """Shorthand for get_translation"""
        return self.get_translation(key_path, **kwargs)
    
    def translate_filter(self, key_path: str, **kwargs) -> str:
        """Jinja2 filter for translations"""
        return self.get_translation(key_path, **kwargs)
    
    def get_all_translations(self, language: Optional[str] = None) -> Dict[str, Any]:
        """Get all translations for a language (for client-side use if needed)"""
        if language is None:
            language = self.get_current_language()
        
        return self.translations.get(language, {})
    
    def get_language_info(self) -> Dict[str, Any]:
        """Get current language information for templates"""
        current_lang = self.get_current_language()
        return {
            'current': current_lang,
            'name': 'Türkçe' if current_lang == 'tr' else 'English',
            'flag': '🇹🇷' if current_lang == 'tr' else '🇺🇸',
            'speech_code': 'tr-TR' if current_lang == 'tr' else 'en-US',
            'supported': self.supported_languages
        }


# Global instance
i18n = I18nManager()
