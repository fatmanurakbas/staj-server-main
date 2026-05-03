"""
Authentication utilities and decorators
"""
from functools import wraps
from flask import session, redirect, url_for, request, flash
import re

def validate_email(email: str) -> bool:
    """Validate email format and allowed domains"""
    # Basic email format validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False
    
    # Check allowed domains
    allowed_domains = [
    ]
    
    domain = email.split('@')[1].lower()
    
    # Allow any .edu domain
    if '.edu.' in domain or domain.endswith('.edu'):
        return True
    
    # Allow any .gov domain
    if '.gov.' in domain or domain.endswith('.gov'):
        return True
        
    return domain in allowed_domains

def validate_password(password: str) -> bool:
    """Validate password strength"""
    if len(password) < 8:
        return False
    
    if not re.search(r'[A-Z]', password):
        return False
    
    if not re.search(r'[a-z]', password):
        return False
    
    if not re.search(r'\d', password):
        return False
    
    return True

def get_medical_fields():
    """Get list of medical fields"""
    return [
        'Gastroenterology',
        'Hepatology',
        'Internal Medicine',
        'Emergency Medicine',
        'Family Medicine',
        'General Surgery',
        'Radiology',
        'Pathology',
        'Medical Student',
        'Resident',
        'Other'
    ]

def get_medical_fields_for_language(language='en'):
    """Get medical fields with translations for the current language"""
    from i18n import i18n
    
    # Get the field translations for the current language
    field_translations = i18n.translations.get(language, {}).get('auth', {}).get('medicalFields', {})
    
    # Base fields (keys)
    base_fields = [
        'Gastroenterology',
        'Hepatology',
        'Internal Medicine',
        'Emergency Medicine',
        'Family Medicine',
        'General Surgery',
        'Radiology',
        'Pathology',
        'Medical Student',
        'Resident',
        'Other'
    ]
    
    # Return translated fields if available, otherwise return base fields
    if field_translations:
        return [(field, field_translations.get(field, field)) for field in base_fields]
    else:
        return [(field, field) for field in base_fields]
