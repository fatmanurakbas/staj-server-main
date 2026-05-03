"""
Comprehensive Liver Disease Risk Assessment System
Supports: Cirrhosis, HCC, and NAFLD risk prediction
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file, make_response
import os
import sys
import re
import json
import tempfile
import traceback
import requests
from typing import Dict, Any
import numpy as np
from dotenv import load_dotenv
import openai
import markdown
from bs4 import BeautifulSoup
from google.genai import Client
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Load environment variables
load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key')

# Configure Google AI Client
genai_client = Client(api_key="AIzaSyCosynuT598IH1I5BPgbOeEdKy0_e5oi-8")

# Import database and authentication utilities
from database import db
from auth_utils import validate_email, validate_password, get_medical_fields, get_medical_fields_for_language

# Import i18n manager
from i18n import i18n
from medical_system_prompt import MEDICAL_SYSTEM_PROMPT
print(MEDICAL_SYSTEM_PROMPT)
print(type(MEDICAL_SYSTEM_PROMPT))
i18n.init_app(app)

# Flask-Login setup
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = None  # Disable automatic login message
login_manager.login_message_category = 'info'

class User(UserMixin):
    def __init__(self, user_id, email, first_name, last_name, doctor_title="Dr."):
        self.id = user_id
        self.email = email
        self.first_name = first_name
        self.last_name = last_name
        self.doctor_title = doctor_title
    
    def get_display_name(self):
        return f"{self.doctor_title} {self.first_name} {self.last_name}"

@login_manager.user_loader
def load_user(user_id):
    user_data = db.get_user_by_id(int(user_id))
    if user_data:
        doctor_title = user_data.get('doctor_title', 'Dr.')
        return User(user_data['id'], user_data['email'], user_data['first_name'], user_data['last_name'], doctor_title)
    return None

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.cirrhosis_model import predict_cirrhosis_risk
from models.hcc_model_final import predict_hcc_risk
from models.nafld_model import predict_nafld_classification

def format_model_name(model_path: str) -> str:
    """Format model names properly for display"""
    model_name = model_path.split('/')[-1]
    
    # Custom formatting for specific models
    if model_name == 'claude-3-haiku':
        return 'Claude 3 Haiku'
    elif model_name == 'gpt-4o':
        return 'GPT-4o'
    elif model_name == 'gemini-flash-1.5':
        return 'Gemini 1.5 Flash'
    else:
        # Generic fallback - replace hyphens with spaces and title case
        return model_name.replace('-', ' ').title()

def calculate_traditional_scores(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate traditional clinical scores once for all diseases
    
    Args:
        patient_data: Dictionary with patient parameters (lowercase keys)
        
    Returns:
        Dictionary with traditional scores
    """
    scores = {}
    
    try:
        # Extract parameters - only calculate scores when sufficient data is available
        age = patient_data.get('age')
        ast = patient_data.get('ast') 
        alt = patient_data.get('alt')
        platelets_raw = patient_data.get('trombosit')  # This should be in x10^3/μL
        albumin = patient_data.get('albumin')
        total_bil = patient_data.get('total_bilirubin')
        inr = patient_data.get('inr')
        creatinine = patient_data.get('creatinine')
        bmi = patient_data.get('bmi')
        
        # FIB-4 Score - only calculate if all required parameters are present
        if all(x is not None and x > 0 for x in [age, ast, alt, platelets_raw]):
            # platelets_raw is already in x10^3/μL, so we use it directly for FIB-4
            scores['FIB-4'] = (age * ast) / (platelets_raw * np.sqrt(alt))
        else:
            scores['FIB-4'] = 'Missing Data'
        
        # APRI Score - only calculate if required parameters are present
        if ast is not None and ast > 0 and platelets_raw is not None and platelets_raw > 0:
            # AST/ULN (assuming ULN = 40 for AST)
            scores['APRI'] = (ast / 40) / platelets_raw * 100
        else:
            scores['APRI'] = 'Missing Data'
        
        # MELD Score - only calculate if all required parameters are present and > 0
        if all(x is not None and x > 0 for x in [total_bil, inr, creatinine]):
            # Ensure minimum values for logarithm (MELD formula requirement)
            safe_bil = max(total_bil, 1.0)
            safe_inr = max(inr, 1.0)
            safe_creat = max(creatinine, 1.0)
            
            scores['MELD'] = (3.78 * np.log(safe_bil) + 
                             11.2 * np.log(safe_inr) + 
                             9.57 * np.log(safe_creat) + 6.43)
        else:
            scores['MELD'] = 'Missing Data'
        
        # BMI Category - only if BMI is provided
        if bmi is not None and bmi > 0:
            if bmi < 18.5:
                scores['BMI Category'] = 'Underweight'
            elif bmi < 25:
                scores['BMI Category'] = 'Normal'
            elif bmi < 30:
                scores['BMI Category'] = 'Overweight'
            else:
                scores['BMI Category'] = 'Obese'
        else:
            scores['BMI Category'] = 'Missing Data'
        
    except Exception as e:
        print(f"Error calculating traditional scores: {e}")
        scores = {
            'FIB-4': 0,
            'APRI': 0,
            'MELD': 0,
            'BMI Category': 'Unknown'
        }
    
    return scores

def get_score_interpretation(score_name: str, score_value: Any) -> Dict[str, str]:
    """
    Get color and interpretation for traditional clinical scores
    
    Args:
        score_name: Name of the score
        score_value: Value of the score
        
    Returns:
        Dictionary with color, interpretation, and level
    """
    # Handle missing or invalid data
    if score_value == 'Missing Data':
        return {'color': 'secondary', 'level': i18n.t('results.missing'), 'interpretation': 'Required parameters not provided'}
    elif score_value == 'Invalid Data':
        return {'color': 'secondary', 'level': i18n.t('results.invalid'), 'interpretation': 'Invalid parameter values'}
    elif score_value is None or (isinstance(score_value, str) and score_value.strip() == ''):
        return {'color': 'secondary', 'level': i18n.t('results.unknown'), 'interpretation': i18n.t('results.unableToCalculate')}
    
    if score_name == 'FIB-4':
        if isinstance(score_value, (int, float)):
            if score_value < 1.30:
                return {'color': 'success', 'level': i18n.t('results.low'), 'interpretation': i18n.t('results.lowProbabilityAdvancedFibrosis')}
            elif score_value <= 2.67:
                return {'color': 'warning', 'level': i18n.t('results.intermediate'), 'interpretation': i18n.t('results.intermediateProbability')}
            else:
                return {'color': 'danger', 'level': i18n.t('results.high'), 'interpretation': i18n.t('results.highProbabilityAdvancedFibrosis')}
        else:
            return {'color': 'secondary', 'level': i18n.t('results.invalid'), 'interpretation': i18n.t('results.unableToCalculate')}
    
    elif score_name == 'APRI':
        if isinstance(score_value, (int, float)):
            if score_value < 0.5:
                return {'color': 'success', 'level': i18n.t('results.low'), 'interpretation': i18n.t('results.lowProbabilitySignificantFibrosis')}
            elif score_value <= 1.5:
                return {'color': 'warning', 'level': i18n.t('results.intermediate'), 'interpretation': i18n.t('results.intermediateProbability')}
            else:
                return {'color': 'danger', 'level': i18n.t('results.high'), 'interpretation': i18n.t('results.highProbabilitySignificantFibrosis')}
        else:
            return {'color': 'secondary', 'level': i18n.t('results.invalid'), 'interpretation': i18n.t('results.unableToCalculate')}
    
    elif score_name == 'MELD':
        if isinstance(score_value, (int, float)):
            if score_value < 10:
                return {'color': 'success', 'level': i18n.t('results.low'), 'interpretation': i18n.t('results.lowMortalityRisk')}
            elif score_value <= 15:
                return {'color': 'warning', 'level': i18n.t('results.moderate'), 'interpretation': i18n.t('results.moderateMortalityRisk')}
            elif score_value <= 20:
                return {'color': 'warning', 'level': i18n.t('results.high'), 'interpretation': i18n.t('results.highMortalityRisk')}
            else:
                return {'color': 'danger', 'level': i18n.t('results.veryHigh'), 'interpretation': i18n.t('results.veryHighMortalityRisk')}
        else:
            return {'color': 'secondary', 'level': i18n.t('results.invalid'), 'interpretation': i18n.t('results.unableToCalculate')}
    
    elif score_name == 'Child-Pugh Score':
        if isinstance(score_value, (int, float)):
            if score_value <= 6:
                return {'color': 'success', 'level': 'Class A', 'interpretation': i18n.t('results.childPughA')}
            elif score_value <= 9:
                return {'color': 'warning', 'level': 'Class B', 'interpretation': i18n.t('results.childPughB')}
            elif score_value <= 15:
                return {'color': 'danger', 'level': 'Class C', 'interpretation': i18n.t('results.childPughC')}
            else:
                return {'color': 'danger', 'level': i18n.t('results.severe'), 'interpretation': i18n.t('results.criticalLiverFunction')}
        else:
            return {'color': 'secondary', 'level': i18n.t('results.invalid'), 'interpretation': i18n.t('results.unableToCalculate')}
    
    elif score_name == 'Child-Pugh Class':
        if score_value == 'A':
            return {'color': 'success', 'level': 'Class A', 'interpretation': i18n.t('results.childPughA')}
        elif score_value == 'B':
            return {'color': 'warning', 'level': 'Class B', 'interpretation': i18n.t('results.childPughB')}
        elif score_value == 'C':
            return {'color': 'danger', 'level': 'Class C', 'interpretation': i18n.t('results.childPughC')}
        else:
            return {'color': 'secondary', 'level': i18n.t('results.unknown'), 'interpretation': i18n.t('results.unableToCalculate')}
    
    elif score_name == 'BMI Category':
        if score_value == 'Underweight':
            return {'color': 'info', 'level': i18n.t('results.underweight'), 'interpretation': i18n.t('results.belowNormalWeight')}
        elif score_value == 'Normal':
            return {'color': 'success', 'level': i18n.t('results.normal'), 'interpretation': i18n.t('results.healthyWeight')}
        elif score_value == 'Overweight':
            return {'color': 'warning', 'level': i18n.t('results.overweight'), 'interpretation': i18n.t('results.aboveNormalWeight')}
        elif score_value == 'Obese':
            return {'color': 'danger', 'level': i18n.t('results.obese'), 'interpretation': i18n.t('results.significantlyAboveNormalWeight')}
        else:
            return {'color': 'secondary', 'level': i18n.t('results.unknown'), 'interpretation': i18n.t('results.unableToDetermineBMI')}
    
    else:
        return {'color': 'secondary', 'level': i18n.t('results.unknown'), 'interpretation': i18n.t('results.noInterpretationAvailable')}

def translate_risk_levels(results):
    """Translate hardcoded risk levels in model results to current language"""
    translation_map = {
        'Low': i18n.t('results.low'),
        'Moderate': i18n.t('results.moderate'), 
        'High': i18n.t('results.high'),
        'Intermediate': i18n.t('results.intermediate'),
        'Very High': i18n.t('results.veryHigh'),
        'Unknown': i18n.t('results.unknown'),
        'Invalid': i18n.t('results.invalid'),
        'Error': i18n.t('results.error')
    }
    
    # Process each disease result
    for disease_key, result in results.items():
        if isinstance(result, dict) and 'risk_level' in result:
            original_level = result['risk_level']
            if original_level in translation_map:
                result['risk_level'] = translation_map[original_level]
    
    return results

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page for medical professionals"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        # Validate inputs
        if not email or not password:
            flash(i18n.t('auth.provideEmailAndPassword'), 'error')
            return render_template('login.html')
        
        # Validate email format and domain
        if not validate_email(email):
            flash(i18n.t('auth.invalidInstitutionalEmail'), 'error')
            return render_template('login.html')
        
        # Check credentials
        user_data = db.verify_user_credentials(email, password)
        if user_data:
            doctor_title = user_data.get('doctor_title', 'Dr.')
            user = User(user_data['id'], user_data['email'], user_data['first_name'], user_data['last_name'], doctor_title)
            login_user(user)
            # flash(f'Welcome back, {user.get_display_name()}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash(i18n.t('auth.invalidCredentials'), 'error')
    
    return render_template('login.html')

def get_doctor_titles_for_language(language='en'):
    """Get doctor titles with translations for the current language"""
    
    # Get the title translations for the current language
    title_translations = i18n.translations.get(language, {}).get('auth', {}).get('doctorTitles', {})
    
    # Base titles (keys)
    base_titles = [
        "Dr.",
        "Prof. Dr.",
        "Doç. Dr.",
        "Öğr. Gör. Dr.",
        "Uzm. Dr.",
        "Op. Dr.",
        "Dt.",
        "Vet.",
        "Ebe",
        "Hemşire"
    ]
    
    # Return translated titles if available, otherwise return base titles
    if title_translations:
        return [(title, title_translations.get(title, title)) for title in base_titles]
    else:
        return [(title, title) for title in base_titles]

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page for medical professionals"""
    if request.method == 'POST':
        # Extract form data
        form_data = {
            'email': request.form.get('email').strip().lower(),
            'password': request.form.get('password'),
            'confirm_password': request.form.get('confirm_password'),
            'first_name': request.form.get('first_name').strip(),
            'last_name': request.form.get('last_name').strip(),
            'doctor_title': request.form.get('doctor_title', 'Dr.').strip(),
            'medical_field': request.form.get('medical_field').strip(),
            'organization': request.form.get('organization').strip(),
            'diploma_number': request.form.get('diploma_number').strip(),
            'years_experience': request.form.get('years_experience', '0'),
            'phone': request.form.get('phone').strip(),
        }
        
        # Validation
        errors = []
        
        # Required fields
        required_fields = ['email', 'password', 'confirm_password', 'first_name', 'last_name', 
                          'medical_field', 'organization', 'diploma_number']
        for field in required_fields:
            if not form_data[field]:
                errors.append(i18n.t(f'auth.{field}Required'))
        
        # Email validation
        if form_data['email'] and not validate_email(form_data['email']):
            errors.append(i18n.t('auth.invalidInstitutionalEmail'))
        
        # Password validation
        if not form_data['password'] or not validate_password(form_data['password']):
            errors.append(i18n.t('auth.weakPassword'))
        
        # Password confirmation
        if form_data['password'] != form_data['confirm_password']:
            errors.append(i18n.t('auth.passwordsNoMatch'))
        
        # Check if user already exists
        if form_data['email'] and db.get_user_by_email(form_data['email']):
            errors.append(i18n.t('auth.accountExists'))
        
        # Years of experience validation
        try:
            years_exp = int(form_data['years_experience'])
            if years_exp < 0 or years_exp > 100:
                errors.append(i18n.t('auth.invalidYearsExperience'))
        except ValueError:
            errors.append(i18n.t('auth.invalidYearsExperienceNumber'))
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('register.html', medical_fields=get_medical_fields_for_language(i18n.get_current_language()), doctor_titles=get_doctor_titles_for_language(i18n.get_current_language()), form_data=form_data)
        
        # Create user
        try:
            user_id = db.create_user(
                email=form_data['email'],
                password=form_data['password'],
                first_name=form_data['first_name'],
                last_name=form_data['last_name'],
                medical_field=form_data['medical_field'],
                organization=form_data['organization'],
                diploma_number=form_data['diploma_number'],
                years_experience=int(form_data['years_experience']),
                phone=form_data['phone'],
                doctor_title=form_data['doctor_title']
            )
            flash(i18n.t('auth.registrationSuccess'), 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(i18n.t('auth.registrationFailed', error=str(e)), 'error')
            return render_template('register.html', medical_fields=get_medical_fields_for_language(i18n.get_current_language()), doctor_titles=get_doctor_titles_for_language(i18n.get_current_language()), form_data=form_data)
    
    return render_template('register.html', medical_fields=get_medical_fields_for_language(i18n.get_current_language()), doctor_titles=get_doctor_titles_for_language(i18n.get_current_language()))

@app.route('/logout')
@login_required
def logout():
    """Logout the current user"""
    # Prevent duplicate logout attempts
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    
    user_name = current_user.get_display_name()
    logout_user()
    
    # Clear existing flash messages to prevent duplicates
    session.pop('_flashes', None)
    flash(i18n.t('auth.goodbyeMessage', user_name=user_name), 'info')
    return redirect(url_for('login'))

@app.route('/set_language/<language>')
def set_language(language):
    """Set language preference"""
    if i18n.set_language(language):
        session.permanent = True
        session['language'] = language
        flash(i18n.get_translation('messages.languageChanged', language=language), 'success')
    else:
        flash(i18n.get_translation('messages.languageNotSupported', language=language), 'error')
    

    # Redirect to the page they came from, or home if none
    return redirect(request.referrer or url_for('index'))

@app.route('/api/translations')
def get_translations():
    """API endpoint to get current translations (for client-side if needed)"""
    return jsonify({
        'translations': i18n.get_all_translations(),
        'language_info': i18n.get_language_info()
    })

@app.route('/')
@login_required
def index():
    """Main page with comprehensive form"""
    return render_template('index.html')

@app.route('/calculate_risks', methods=['POST'])
@login_required
def calculate_risks():
    """Calculate risks for all liver diseases"""
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Prepare patient data for each model
        # Common fields mapping
        patient_data = {}
        
        # Extract and convert form data
        for key, value in form_data.items():
            if value and value.strip():
                try:
                    # Convert to float if it's a numeric field
                    if key in ['age', 'Age', 'ast', 'AST', 'alt', 'ALT', 'trombosit', 'Trombosit', 
                              'albumin', 'Albumin', 'bmi', 'BMI', 'inr', 'INR', 'total_bilirubin', 
                              'Total_Bilirubin', 'creatinine', 'Creatinine', 'direct_bilirubin', 
                              'Direct_Bilirubin', 'alp', 'ALP', 'afp', 'AFP', 'gender', 'Gender', 
                              'obesity', 'Obesity', 'Total_Bil', 'Dir_Bil', 'ggt', 'encephalopathy', 'ascites']:
                        patient_data[key] = float(value)
                    else:
                        patient_data[key] = value
                except ValueError:
                    # If conversion fails, keep as string
                    patient_data[key] = value
        
        # Prepare data for Cirrhosis model
        cirrhosis_data = {}
        for field in ['age', 'gender', 'ast', 'alt', 'trombosit', 'albumin', 'bmi', 'inr', 'total_bilirubin', 'creatinine', 'direct_bilirubin', 'alp']:
            if field in patient_data:
                cirrhosis_data[field] = patient_data[field]
        
        # Prepare data for HCC model
        hcc_data = {}
        for field in ['age', 'gender', 'ast', 'alt', 'albumin', 'creatinine', 'inr', 'trombosit', 'total_bilirubin', 'direct_bilirubin', 'obesity', 'alp']:
            if field in patient_data:
                hcc_data[field] = patient_data[field]
        
        # Add AFP if provided (optional field)
        if 'afp' in patient_data:
            hcc_data['afp'] = patient_data['afp']
        
        # Prepare data for NAFLD model
        nafld_data = {}
        for field in ['age', 'gender', 'ast', 'alt', 'trombosit', 'albumin', 'bmi', 'inr', 'total_bilirubin', 'creatinine', 'direct_bilirubin', 'alp']:
            if field in patient_data:
                nafld_data[field] = patient_data[field]
        
        # Calculate risks for all diseases
        results = {}
        
        # Cirrhosis risk
        try:
            cirrhosis_result = predict_cirrhosis_risk(cirrhosis_data)
            results['cirrhosis'] = cirrhosis_result
        except Exception as e:
            results['cirrhosis'] = {
                'disease': 'Cirrhosis',
                'error': str(e),
                'risk_level': 'Error',
                'risk_color': 'secondary'
            }
        
        # HCC risk
        try:
            hcc_result = predict_hcc_risk(hcc_data)
            results['hcc'] = hcc_result
        except Exception as e:
            results['hcc'] = {
                'disease': 'HCC (Hepatocellular Carcinoma)',
                'error': str(e),
                'risk_level': 'Error',
                'risk_color': 'secondary'
            }
        
        # NAFLD risk
        try:
            nafld_result = predict_nafld_classification(nafld_data)
            results['nafld'] = nafld_result
        except Exception as e:
            results['nafld'] = {
                'disease': 'MAFLD Classification',
                'error': str(e),
                'classification': 'Error',
                'risk_color': 'secondary'
            }
        
        # Calculate traditional scores
        traditional_scores = calculate_traditional_scores(patient_data)
        
        # Translate risk levels in results
        results = translate_risk_levels(results)
        
        # Child-Pugh Score calculation
        def calculate_child_pugh(albumin, bilirubin, inr, ascites, encephalopathy):
            # Albumin (g/dL) - Standard Child-Pugh criteria
            if albumin > 3.5:
                alb_score = 1
            elif 2.8 <= albumin <= 3.5:
                alb_score = 2
            else:  # < 2.8
                alb_score = 3
                
            # Bilirubin (mg/dL) - Standard Child-Pugh criteria
            if bilirubin < 2:
                bil_score = 1
            elif 2 <= bilirubin <= 3:
                bil_score = 2
            else:  # > 3
                bil_score = 3
                
            # INR - Standard Child-Pugh criteria
            if inr < 1.7:
                inr_score = 1
            elif 1.7 <= inr <= 2.3:  # Fixed: was 2.2, should be 2.3
                inr_score = 2
            else:  # > 2.3
                inr_score = 3
                
            # Ascites (0: none, 1: mild, 2: severe/moderate)
            ascites_score = int(ascites) + 1
            
            # Encephalopathy (0: none, 1: grade 1-2, 2: grade 3-4)
            enceph_score = int(encephalopathy) + 1
            
            total = alb_score + bil_score + inr_score + ascites_score + enceph_score
            
            # Child-Pugh Class determination
            if total <= 6:
                c_class = 'A'
            elif 7 <= total <= 9:
                c_class = 'B'
            else:  # >= 10
                c_class = 'C'
                
            return {'score': total, 'class': c_class}
        
        # Extract parameters for Child-Pugh calculation
        def safe_int(val):
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0
        
        try:
            # Validate required parameters - DO NOT default to medically dangerous values
            required_params = ['albumin', 'total_bilirubin', 'inr', 'ascites', 'encephalopathy']
            missing_params = []
            
            for param in required_params:
                if param not in patient_data or patient_data[param] is None or patient_data[param] == '':
                    missing_params.append(param)
            
            if missing_params:
                print(f"Child-Pugh calculation skipped - missing required parameters: {missing_params}")
                # Don't calculate Child-Pugh if critical parameters are missing
                return
            
            # Safe extraction with validation
            albumin = float(patient_data['albumin'])
            bilirubin = float(patient_data['total_bilirubin'])  
            inr = float(patient_data['inr'])
            ascites = safe_int(patient_data['ascites'])
            encephalopathy = safe_int(patient_data['encephalopathy'])
            
            # Validate ranges are medically reasonable
            if albumin <= 0 or bilirubin <= 0 or inr <= 0:
                print(f"Child-Pugh calculation skipped - invalid parameter values: albumin={albumin}, bilirubin={bilirubin}, inr={inr}")
                return
            
            child_pugh = calculate_child_pugh(albumin, bilirubin, inr, ascites, encephalopathy)
            
            # Ensure traditional_scores is a proper dictionary before adding Child-Pugh
            if not isinstance(traditional_scores, dict):
                traditional_scores = {}
            
            # Add Child-Pugh to traditional scores
            traditional_scores['Child-Pugh Score'] = child_pugh['score']
            traditional_scores['Child-Pugh Class'] = child_pugh['class']
        except Exception as e:
            print(f"Error calculating Child-Pugh score: {e}")
            # Ensure traditional_scores is a dictionary even if Child-Pugh fails
            if not isinstance(traditional_scores, dict):
                traditional_scores = {}
            traditional_scores['Child-Pugh Score'] = 0
            traditional_scores['Child-Pugh Class'] = 'Unknown'
            child_pugh = {'score': 0, 'class': 'Unknown'}
        
        # Get score interpretations AFTER adding Child-Pugh scores
        score_interpretations = {}
        for score_name, score_value in traditional_scores.items():
            score_interpretations[score_name] = get_score_interpretation(score_name, score_value)
        
        # Store patient data and results in session for later use (AI doctor assessment)
        # Convert to JSON-serializable format
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        session['patient_data'] = make_serializable(patient_data)
        session['results'] = make_serializable(results)
        session['traditional_scores'] = make_serializable(traditional_scores)
        session['score_interpretations'] = make_serializable(score_interpretations)
        
        return render_template('results.html', 
                             results=results, 
                             patient_data=patient_data,
                             traditional_scores=traditional_scores,
                             score_interpretations=score_interpretations,
                             has_afp='afp' in patient_data and patient_data['afp'],
                             child_pugh=child_pugh)
    except Exception as e:
        error_msg = f"An error occurred during risk calculation: {str(e)}"
        # Calculate traditional scores even if there's an error
        try:
            traditional_scores = calculate_traditional_scores(patient_data)
            
            # Also calculate Child-Pugh in error case
            def safe_int(val):
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return 0
            
            def calculate_child_pugh(albumin, bilirubin, inr, ascites, encephalopathy):
                # Albumin (g/dL) - Standard Child-Pugh criteria
                if albumin > 3.5:
                    alb_score = 1
                elif 2.8 <= albumin <= 3.5:
                    alb_score = 2
                else:  # < 2.8
                    alb_score = 3
                    
                # Bilirubin (mg/dL) - Standard Child-Pugh criteria
                if bilirubin < 2:
                    bil_score = 1
                elif 2 <= bilirubin <= 3:
                    bil_score = 2
                else:  # > 3
                    bil_score = 3
                    
                # INR - Standard Child-Pugh criteria
                if inr < 1.7:
                    inr_score = 1
                elif 1.7 <= inr <= 2.3:  # Fixed: was 2.2, should be 2.3
                    inr_score = 2
                else:  # > 2.3
                    inr_score = 3
                    
                # Ascites (0: none, 1: mild, 2: severe/moderate)
                ascites_score = int(ascites) + 1
                
                # Encephalopathy (0: none, 1: grade 1-2, 2: grade 3-4)
                enceph_score = int(encephalopathy) + 1
                
                total = alb_score + bil_score + inr_score + ascites_score + enceph_score
                
                # Child-Pugh Class determination
                if total <= 6:
                    c_class = 'A'
                elif 7 <= total <= 9:
                    c_class = 'B'
                else:  # >= 10
                    c_class = 'C'
                    
                return {'score': total, 'class': c_class}
            
            try:
                # Validate required parameters - DO NOT default to medically dangerous values
                required_params = ['albumin', 'total_bilirubin', 'inr', 'ascites', 'encephalopathy']
                missing_params = []
                
                for param in required_params:
                    if param not in patient_data or patient_data[param] is None or patient_data[param] == '':
                        missing_params.append(param)
                
                if missing_params:
                    print(f"Child-Pugh calculation skipped in error handler - missing required parameters: {missing_params}")
                    # Don't calculate Child-Pugh if critical parameters are missing
                    if not isinstance(traditional_scores, dict):
                        traditional_scores = {}
                    traditional_scores['Child-Pugh Score'] = 'Missing Data'
                    traditional_scores['Child-Pugh Class'] = 'Missing Data'
                else:
                    # Safe extraction with validation
                    albumin = float(patient_data['albumin'])
                    bilirubin = float(patient_data['total_bilirubin'])  
                    inr = float(patient_data['inr'])
                    ascites = safe_int(patient_data['ascites'])
                    encephalopathy = safe_int(patient_data['encephalopathy'])
                    
                    # Validate ranges are medically reasonable
                    if albumin <= 0 or bilirubin <= 0 or inr <= 0:
                        print(f"Child-Pugh calculation skipped - invalid parameter values: albumin={albumin}, bilirubin={bilirubin}, inr={inr}")
                        if not isinstance(traditional_scores, dict):
                            traditional_scores = {}
                        traditional_scores['Child-Pugh Score'] = 'Invalid Data'
                        traditional_scores['Child-Pugh Class'] = 'Invalid Data'
                    else:
                        child_pugh = calculate_child_pugh(albumin, bilirubin, inr, ascites, encephalopathy)
                        
                        # Ensure traditional_scores is a proper dictionary before adding Child-Pugh
                        if not isinstance(traditional_scores, dict):
                            traditional_scores = {}
                        
                        # Add Child-Pugh to traditional scores
                        traditional_scores['Child-Pugh Score'] = child_pugh['score']
                        traditional_scores['Child-Pugh Class'] = child_pugh['class']
            except Exception as child_pugh_error:
                print(f"Error calculating Child-Pugh in error handler: {child_pugh_error}")
                # Ensure traditional_scores is a dictionary even if Child-Pugh fails
                if not isinstance(traditional_scores, dict):
                    traditional_scores = {}
                traditional_scores['Child-Pugh Score'] = 0
                traditional_scores['Child-Pugh Class'] = 'Unknown'
            score_interpretations = {}
            for score_name, score_value in traditional_scores.items():
                score_interpretations[score_name] = get_score_interpretation(score_name, score_value)
        except:
            traditional_scores = {}
            score_interpretations = {}
        
        return render_template('results.html', 
                             error=error_msg,
                             results={},
                             patient_data=patient_data,
                             traditional_scores=traditional_scores,
                             score_interpretations=score_interpretations)

@app.route('/api/calculate_risks', methods=['POST'])
@login_required
def api_calculate_risks():
    """API endpoint for risk calculation"""
    try:
        data = request.get_json()
        
        # Process similar to above but return JSON
        results = {}
        
        # Cirrhosis risk
        cirrhosis_data = {
            'Age': data.get('age', 0),
            'Gender': data.get('gender', 0),
            'AST': data.get('ast', 0),
            'ALT': data.get('alt', 0),
            'Trombosit': data.get('trombosit', 0),
            'Albumin': data.get('albumin', 0),
            'BMI': data.get('bmi', 0),
            'INR': data.get('inr', 0),
            'Total_Bilirubin': data.get('total_bilirubin', 0),
            'Creatinine': data.get('creatinine', 0),
            'Direct_Bilirubin': data.get('direct_bilirubin', 0),
            'ALP': data.get('alp', 0)
        }
        results['cirrhosis'] = predict_cirrhosis_risk(cirrhosis_data)
        
        # HCC risk
        hcc_data = {
            'Age': data.get('age', 0),
            'Gender': data.get('gender', 0) - 1,
            'AST': data.get('ast', 0),
            'ALT': data.get('alt', 0),
            'Albumin': data.get('albumin', 0),
            'Creatinine': data.get('creatinine', 0),
            'INR': data.get('inr', 0),
            'Trombosit': data.get('trombosit', 0),
            'Total_Bil': data.get('total_bilirubin', 0),
            'Dir_Bil': data.get('direct_bilirubin', 0),
            'Obesity': data.get('obesity', 0),
            'ALP': data.get('alp', 0)
        }
        if 'afp' in data and data['afp']:
            hcc_data['AFP'] = data['afp']
        results['hcc'] = predict_hcc_risk(hcc_data)
        
        # NAFLD risk
        nafld_data = {
            'age': data.get('age', 0),
            'gender': data.get('gender', 0),
            'AST': data.get('ast', 0),
            'ALT': data.get('alt', 0),
            'trombosit': data.get('trombosit', 0),
            'albumin': data.get('albumin', 0),
            'bmi': data.get('bmi', 0),
            'inr': data.get('inr', 0),
            'total_bilirubin': data.get('total_bilirubin', 0),
            'creatinine': data.get('creatinine', 0),
            'direct_bilirubin': data.get('direct_bilirubin', 0),
            'ALP': data.get('alp', 0)
        }
        results['nafld'] = predict_nafld_classification(nafld_data)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """
    Chatbot API endpoint for medical assistance
    Supports different user roles: Uzman Doktor, Asistan, Öğrenci
    """
    try:
        data = request.get_json()
        role = data.get('role', 'Öğrenci')
        message = data.get('message', '')
        
        if not message.strip():
            return jsonify({'reply': 'Lütfen bir soru yazın.'}), 400
        
        # Role-based prompts
        role_prompts = {
            'Uzman Doktor': "Sen bir tıp uzmanısın. Cevapların kısa ve teknik olmalı.",
            'Asistan': "Sen bir asistan doktorsun. Cevapların orta düzey açıklayıcı olsun.", 
            'Öğrenci': "Sen bir öğretmensin. Cevapların detaylı ve öğretici olsun."
        }
        
        system_prompt = MEDICAL_SYSTEM_PROMPT.format(
            role=role,
            role_prompt=role_prompts.get(role, role_prompts['Öğrenci'])
        )
        
        # Use OpenRouter API (since your OPENAI_API_KEY is actually an OpenRouter key)
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5002",
                "X-Title": "Liver Disease Assessment Chatbot"
            },
            json={
                "model": "openai/gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
        )
        
        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            return jsonify({"reply": reply})
        else:
            error_detail = response.text if response.text else f"Status code: {response.status_code}"
            print(f"OpenRouter API error: {error_detail}")
            return jsonify({"reply": "API bağlantısında bir hata oluştu. Lütfen tekrar deneyin."}), 500
            
    except Exception as e:
        print(f"Chat API error: {e}")
        print(f"Error type: {type(e)}")
        traceback.print_exc()
        return jsonify({"reply": "Bir hata oluştu. Lütfen tekrar deneyin."}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'service': 'Liver Disease Risk Assessment',
        'ai_model_loaded': True
    })

@app.route('/sample/<patient_id>')
def get_sample_patient(patient_id):
    """Get sample patient data for testing"""
    sample_patients = {
        'low': {
            'name': 'Low Risk Patient - Healthy Young Adult',
            'age': 28,
            'gender': 1,
            'bmi': 21.5,
            'obesity': 0,
            'ast': 18,
            'alt': 22,
            'alp': 65,
            'trombosit': 320,
            'albumin': 4.7,
            'inr': 0.9,
            'total_bilirubin': 0.5,
            'direct_bilirubin': 0.12,
            'creatinine': 0.7,
            'afp': 1.8
        },
        'moderate': {
            'name': 'Moderate Risk Patient - MAFLD with Fibrosis',
            'age': 52,
            'gender': 2,
            'bmi': 29.2,
            'obesity': 0,
            'ast': 78,
            'alt': 92,
            'alp': 145,
            'trombosit': 135,
            'albumin': 3.4,
            'inr': 1.4,
            'total_bilirubin': 2.1,
            'direct_bilirubin': 0.8,
            'creatinine': 1.3,
            'afp': 18.5
        },
        'high': {
            'name': 'High Risk Patient - Advanced Liver Disease',
            'age': 58,
            'gender': 2,
            'bmi': 33.8,
            'obesity': 1,
            'ast': 210,
            'alt': 185,
            'alp': 285,
            'trombosit': 72,
            'albumin': 2.4,
            'inr': 2.8,
            'total_bilirubin': 6.2,
            'direct_bilirubin': 3.8,
            'creatinine': 2.1,
            'afp': 280.0
        }
    }
    
    if patient_id in sample_patients:
        return jsonify(sample_patients[patient_id])
    else:
        return jsonify({'error': 'Patient not found'}), 404

@app.route('/doctor-assessment', methods=['POST'])
@login_required
def doctor_assessment():
    try:
        data = request.get_json()
        doctor = data.get('doctor', 'smith')
        doctor_map = {
            'smith': {'name': 'Smith', 'model': 'anthropic/claude-3-haiku'},
            'johnson': {'name': 'Johnson', 'model': 'openai/gpt-4o'},
            'brown': {'name': 'Brown', 'model': 'google/gemini-flash-2.5'}
        }
        if doctor not in doctor_map:
            return jsonify({'success': False, 'error': 'Invalid doctor selected.'})
        doctor_info = doctor_map[doctor]
        # Retrieve patient data and results from session
        patient_data = session.get('patient_data')
        results = session.get('results')
        traditional_scores = session.get('traditional_scores')
        if not patient_data or not results or not traditional_scores:
            return jsonify({'success': False, 'error': 'Session expired or missing data. Please recalculate risks.'})
        # Prepare prompt
        with open('PROMPT', 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        # Format patient data for prompt
        patient_data_str = '\n'.join([f"{k}: {v}" for k, v in patient_data.items()])
        traditional_scores_str = '\n'.join([f"{k}: {v}" for k, v in traditional_scores.items()])
        prompt = prompt_template.replace('{{doctor_name}}', doctor_info['name']) \
            .replace('{{patient_inputted_data_like_gender_age_alt_ast_all_of_them}}', patient_data_str) \
            .replace('{{predicted_cirrhosis_risk}}', str(results['cirrhosis'].get('risk_percentage', 'N/A'))) \
            .replace('{{predicted_hcc_risk}}', str(results['hcc'].get('risk_percentage', 'N/A'))) \
            .replace('{{predicted_nafld_risk}}', str(results['nafld'].get('risk_percentage', 'N/A'))) \
            .replace('{{traditional_scores}}', traditional_scores_str)
        
        # Log the prompt to console for debugging
        print(f"\n{'='*60}")
        print(f"AI DOCTOR ASSESSMENT REQUEST - Doctor {doctor_info['name']}")
        print(f"{'='*60}")
        print(f"Model: {doctor_info['model']}")
        print(f"Prompt:\n{prompt}")
        print(f"{'='*60}\n")
        
        # Call OpenRouter API using OpenAI SDK
        client = openai.OpenAI(
            api_key=os.environ.get('OPENROUTER_API_KEY'),
            base_url='https://openrouter.ai/api/v1'
        )
        model = doctor_info['model']
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7
        )
        assessment = response.choices[0].message.content
        
        # Log the AI response for debugging
        print(f"AI RESPONSE from {doctor_info['name']} ({doctor_info['model']}):")
        print(f"Response: {assessment}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'title': f"{doctor_info['name']}'s ({model.split('/')[-1].replace('-', ' ').title()})",
            'assessment': assessment
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/process_document', methods=['POST'])
@login_required
def process_document():
    """Process uploaded medical document using Google Gemini OCR"""
    try:
        if 'document' not in request.files:
            return jsonify({'success': False, 'error': i18n.t('errors.noDocumentUploaded')})
        
        file = request.files['document']
        if file.filename == '':
            return jsonify({'success': False, 'error': i18n.t('errors.noFileSelected')})
        
        # Read the OCR prompt
        with open('DOCUMENT_OCR_PROMPT', 'r', encoding='utf-8') as f:
            ocr_prompt = f.read()
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Upload file to Google AI
            uploaded_file = genai_client.files.upload(file=temp_file_path)
            
            response = genai_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[
                    ocr_prompt,
                    uploaded_file
                ]
            )
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
        
        # Debug: Print the raw response from Gemini
        print(f"GEMINI RAW RESPONSE:")
        print(f"Response type: {type(response)}")
        print(f"Response attributes: {dir(response)}")
        
        # Get response text with better error handling
        response_text = ""
        if hasattr(response, 'text'):
            response_text = response.text
            print(f"Response.text: '{response_text}'")
        elif hasattr(response, 'content'):
            response_text = response.content
            print(f"Response.content: '{response_text}'")
        elif hasattr(response, 'candidates') and response.candidates:
            if hasattr(response.candidates[0], 'content'):
                response_text = response.candidates[0].content.parts[0].text if response.candidates[0].content.parts else ""
                print(f"Response.candidates[0].content.parts[0].text: '{response_text}'")
        else:
            response_text = str(response)
            print(f"str(response): '{response_text}'")
        
        print(f"Final response_text: '{response_text}'")
        print(f"Response length: {len(response_text)}")
        print(f"{'='*60}\n")
        
        # Parse the JSON response with better error handling
        try:
            if not response_text or response_text.strip() == "":
                raise ValueError("Empty response from Gemini")
            
            # Clean the response text - remove markdown code blocks if present
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '').strip()
            elif cleaned_text.startswith('```'):
                cleaned_text = cleaned_text.replace('```', '').strip()
            
            print(f"Cleaned response text: '{cleaned_text}'")
            extracted_data = json.loads(cleaned_text)
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}")
            print(f"Attempting to extract JSON from response...")
            
            # Try to find JSON within the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    extracted_data = json.loads(json_match.group(0))
                    print(f"Successfully extracted JSON from response")
                except json.JSONDecodeError:
                    # Return error with the actual response for debugging
                    return jsonify({
                        'success': False, 
                        'error': i18n.t('errors.invalidJsonResponse'),
                        'debug_info': {
                            'response_type': str(type(response)),
                            'raw_response': response_text[:1000]
                        }
                    })
            else:
                return jsonify({
                    'success': False, 
                    'error': i18n.t('errors.noValidJsonFound'),
                    'debug_info': {
                        'response_type': str(type(response)),
                        'raw_response': response_text[:1000]
                    }
                })
        
        # Log the extraction for debugging
        print(f"DOCUMENT OCR EXTRACTION SUCCESS:")
        print(f"Extracted data: {extracted_data}")
        print(f"Data type: {type(extracted_data)}")
        print(f"Data keys: {list(extracted_data.keys()) if isinstance(extracted_data, dict) else 'Not a dict'}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'data': extracted_data
        })
        
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"{'='*60}\n")
        return jsonify({'success': False, 'error': i18n.t('errors.processingError', error=str(e))})

@app.route('/get_ai_assessment', methods=['POST'])
@login_required
def get_ai_assessment():
    """Get AI doctor assessment enriched with XAI (Explainable AI) insights"""
    try:
        # İstekten seçilen doktoru al
        data = request.get_json()
        selected_doctor = data.get('doctor', '')
        lang = session.get('language', 'tr')
        
        # Oturum verilerini geri al
        patient_data = session.get('patient_data', {})
        results = session.get('results', {})
        traditional_scores = session.get('traditional_scores', {})
        
        if not patient_data or not results:
            return jsonify({
                'success': False, 
                'error': 'Oturum verisi bulunamadı. Lütfen önce riskleri hesaplayın.'
            })
    
        doctor_map = {
            'smith': {'name': i18n.t('drclaude'), 'model': 'anthropic/claude-3-haiku'},
            'johnson': {'name': i18n.t('drgpt'), 'model': 'openai/gpt-4o'},
            'brown': {'name': i18n.t('drgemini'), 'model': 'google/gemini-2.5-flash'}
        }
        
        if selected_doctor not in doctor_map:
            return jsonify({
                'success': False, 
                'error': 'Geçersiz doktor seçimi.'
            })
        
        doctor_info = doctor_map[selected_doctor]
        
        # --- XAI (AÇIKLANABİLİRLİK) VERİLERİNİ HAZIRLA ---
        xai_data_for_prompt = ""
        cirrhosis_res = results.get('cirrhosis', {})
        
        if 'xai' in cirrhosis_res and cirrhosis_res['xai']:
            insights = cirrhosis_res['xai'].get('actionable_insights', [])
            if insights:
                xai_data_for_prompt = "\n\n### YAPAY ZEKA MODEL ANALİZİ (XAI):\n"
                xai_data_for_prompt += "Matematiksel modelimiz (SHAP), bu hasta için şu iyileştirmelerin riski doğrudan düşüreceğini saptadı:\n"
                for i in insights:
                    xai_data_for_prompt += f"- {i['feature'].upper()}: Şu an {i['current']} birim. {i['target']} birimin altına düşerse risk %{i['impact']} oranında azalacaktır.\n"
                xai_data_for_prompt += "\nLütfen bu verileri kullanarak hastaya somut ve eyleme geçirilebilir tıbbi önerilerde bulun."

        # Prompt şablonunu yükle
        with open('PROMPT', 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # Verileri prompt için formatla
        patient_data_str = '\n'.join([f"{k}: {v}" for k, v in patient_data.items()])
        # XAI bilgisini hasta verilerinin hemen sonuna ekliyoruz ki AI doktoru bunu görsün
        patient_data_str += xai_data_for_prompt 
        
        traditional_scores_str = '\n'.join([f"{k}: {v}" for k, v in traditional_scores.items()])
        
        # Değişkenleri prompt içine yerleştir
        prompt = prompt_template.replace('{{doctor_name}}', doctor_info['name']) \
            .replace('{{patient_inputted_data_like_gender_age_alt_ast_all_of_them}}', patient_data_str) \
            .replace('{{predicted_cirrhosis_risk}}', str(results['cirrhosis'].get('risk_percentage', 'N/A'))) \
            .replace('{{predicted_hcc_risk}}', str(results['hcc'].get('risk_percentage', 'N/A'))) \
            .replace('{{predicted_nafld_risk}}', str(results['nafld'].get('classification', 'N/A')) + f" (confidence: {results['nafld'].get('confidence', 'N/A')}%)" if results['nafld'].get('classification') else 'N/A') \
            .replace('{{traditional_scores}}', traditional_scores_str) \
            .replace('{{lang}}', lang)
        
        # Loglama (Hata ayıklama için terminale basar)
        print(f"\n{'='*60}")
        print(f"XAI DESTEKLİ AI DOKTOR TAVSİYESİ - {doctor_info['name']}")
        print(f"{'='*60}")
        
        # OpenRouter API Çağrısı
        client = openai.OpenAI(
            api_key=os.environ.get('OPENROUTER_API_KEY'),
            base_url='https://openrouter.ai/api/v1'
        )
        model = doctor_info['model']
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000, # XAI verisi geldiği için token limitini biraz artırdık
            temperature=0.7
        )
        
        ai_assessment = response.choices[0].message.content
        ai_assessment_title = f"{doctor_info['name']}'s ({format_model_name(doctor_info['model'])}) XAI-Powered Assessment"
        
        # Markdown'ı HTML'e çevir
        ai_assessment_html = markdown.markdown(ai_assessment)
        
        return jsonify({
            'success': True,
            'assessment': ai_assessment_html,
            'title': ai_assessment_title
        })
        
    except Exception as e:
        print(f"Error getting AI assessment with XAI: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'AI doktor değerlendirmesi alınırken hata oluştu: {str(e)}'
        })
    
@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    from reportlab.platypus import Table, TableStyle, Paragraph, SimpleDocTemplate, Spacer
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    import math
    import re

    # Always use backend session data for PDF
    patient = session.get('patient_data', {})
    results = session.get('results', {})
    traditional_scores = session.get('traditional_scores', {})
    score_interpretations = session.get('score_interpretations', {})
    lang = session.get('language', 'tr')
    i18n.set_language(lang)
    
    # Get AI assessment from request
    ai_assessment = ''
    if request.is_json:
        ai_assessment = request.get_json().get('ai_assessment', '')
    else:
        ai_assessment = request.form.get('ai_assessment', '')

    # Helper for language
    def _(key, default=None):
        try:
            return i18n.t(key)
        except Exception:
            return default or key

    # Gender/Obesity mapping
    gender_map = {'tr': {1: 'Erkek', 2: 'Kadın'}, 'en': {1: 'Male', 2: 'Female'}}
    obesity_map = {'tr': {0: 'Hayır', 1: 'Evet'}, 'en': {0: 'No', 1: 'Yes'}}
    lang_short = lang if lang in ['tr', 'en'] else 'tr'

    # Format patient info
    def fmt(val, digits=1):
        if isinstance(val, float):
            if val.is_integer():
                return str(int(val))
            return f"{val:.{digits}f}"
        return str(val)

    # PDF setup
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    content_width = width - 2 * margin
    y = height - margin

    # Register Inter font (with fallback)
    try:
        INTER_FONT_PATH = os.path.join(os.path.dirname(__file__), 'static', 'js', 'fonts', 'Inter-Regular.ttf')
        INTER_BOLD_PATH = os.path.join(os.path.dirname(__file__), 'static', 'js', 'fonts', 'Inter-Bold.ttf')
        
        if os.path.exists(INTER_FONT_PATH):
            pdfmetrics.registerFont(TTFont('Inter', INTER_FONT_PATH))
        if os.path.exists(INTER_BOLD_PATH):
            pdfmetrics.registerFont(TTFont('Inter-Bold', INTER_BOLD_PATH))
        else:
            # Use regular Inter for bold if bold variant not available
            pdfmetrics.registerFont(TTFont('Inter-Bold', INTER_FONT_PATH))
        
        font_family = 'Inter'
        font_bold = 'Inter-Bold'
    except:
        # Fallback to Helvetica
        font_family = 'Helvetica'
        font_bold = 'Helvetica-Bold'

    # Helper function to draw rounded rectangle
    def draw_rounded_rect(x, y, width, height, radius, fill_color, stroke_color=None):
        p.setFillColorRGB(*fill_color)
        if stroke_color:
            p.setStrokeColorRGB(*stroke_color)
        p.roundRect(x, y, width, height, radius, fill=1, stroke=1 if stroke_color else 0)

    # Helper function to check if new page is needed
    def check_new_page(current_y, needed_space):
        if current_y - needed_space < 60:
            p.showPage()
            return height - margin
        return current_y

    # Helper function for word wrapping
    def wrap_text(text, font_name, font_size, max_width):
        """Wrap text to fit within max_width"""
        p.setFont(font_name, font_size)
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            text_width = p.stringWidth(test_line, font_name, font_size)
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is too long, break it
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

    # Header with gradient-like effect
    header_height = 80
    draw_rounded_rect(margin, y - header_height, content_width, header_height, 10, 
                     (0.16, 0.5, 0.73), (0.12, 0.4, 0.6))
    
    # Title
    p.setFont(font_bold, 20)
    p.setFillColorRGB(1, 1, 1)  # White text
    title = _("results.report_title", 'LiverAId')
    p.drawCentredString(width/2, y - 25, title)
    
    # Subtitle
    p.setFont(font_family, 12)
    p.setFillColorRGB(0.9, 0.9, 0.9)  # Light gray
    subtitle = _("results.report_subtitle", 'Siroz, HCC ve MAFLD için Kapsamlı Analiz')
    p.drawCentredString(width/2, y - 45, subtitle)
    
    # Date
    p.setFont(font_family, 10)
    from datetime import datetime
    date_str = _("results.generated_at", 'Oluşturulma Tarihi:') + f" {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
    p.drawCentredString(width/2, y - 65, date_str)
    
    y -= header_height + 20

    # Helper function to draw section header
    def draw_section_header(title, y_pos, icon="●"):
        y_pos = check_new_page(y_pos, 30)
        # Background for section header
        draw_rounded_rect(margin, y_pos - 25, content_width, 25, 5, (0.95, 0.95, 0.95))
        
        p.setFont(font_bold, 14)
        p.setFillColorRGB(0.2, 0.2, 0.2)
        p.drawString(margin + 10, y_pos - 18, f"{title}")
        return y_pos - 35

    # Helper function to draw table with better styling
    def draw_styled_table(data, y_pos, col_widths, header_bg=(0.2, 0.4, 0.6), row_bg=(0.98, 0.98, 0.98)):
        y_pos = check_new_page(y_pos, len(data) * 20 + 10)
        
        row_height = 18
        table_width = sum(col_widths)
        
        for i, row in enumerate(data):
            # Row background
            if i == 0:  # Header
                draw_rounded_rect(margin, y_pos - row_height, table_width, row_height, 3, header_bg)
                p.setFont(font_bold, 10)
                p.setFillColorRGB(1, 1, 1)  # White text for header
            else:  # Data rows
                if i % 2 == 0:  # Alternate row colors
                    draw_rounded_rect(margin, y_pos - row_height, table_width, row_height, 3, row_bg)
                p.setFont(font_family, 9)
                p.setFillColorRGB(0.2, 0.2, 0.2)
            
            # Draw cells
            x = margin + 5
            for j, cell in enumerate(row):
                cell_text = str(cell)
                # Truncate long text
                if len(cell_text) > 25:
                    cell_text = cell_text[:22] + "..."
                p.drawString(x, y_pos - 13, cell_text)
                x += col_widths[j]
            
            y_pos -= row_height
        
        return y_pos - 10

    # Helper functions for ascites and encephalopathy descriptions
    def get_ascites_description(value, lang):
        """Get localized description for ascites value"""
        ascites_map = {
            'tr': {0: 'Yok', 1: 'Hafif', 2: 'Ağır/Refrakter'},
            'en': {0: 'None', 1: 'Mild', 2: 'Severe/Refractory'}
        }
        try:
            return ascites_map[lang].get(int(value), _("results.unspecified", "Belirtilmemiş"))
        except (ValueError, TypeError):
            return _("results.unspecified", "Belirtilmemiş")
    
    def get_encephalopathy_description(value, lang):
        """Get localized description for encephalopathy value"""
        encephalopathy_map = {
            'tr': {0: 'Yok', 1: 'Hafif/Grade I-II', 2: 'Ağır/Grade III-IV'},
            'en': {0: 'None', 1: 'Mild/Grade I-II', 2: 'Severe/Grade III-IV'}
        }
        try:
            return encephalopathy_map[lang].get(int(value), _("results.unspecified", "Belirtilmemiş"))
        except (ValueError, TypeError):
            return _("results.unspecified", "Belirtilmemiş")

    # Patient Information Section
    y = draw_section_header(_("results.patient_info", "Hasta Bilgileri"), y)
    
    patient_info = [
        [_("results.parameter", "Parametre"), _("results.value", "Değer")],
        [_("results.age", "Yaş"), fmt(patient.get('age', 'Belirtilmemiş')) + f" {_('results.years', 'yıl')}" if patient.get('age') else _("results.unspecified", "Belirtilmemiş")],
        [_("results.gender", "Cinsiyet"), gender_map[lang_short].get(int(patient.get('gender', 0)), _("results.unspecified", "Belirtilmemiş"))],
        [_("results.bmi", "BMI"), fmt(patient.get('bmi', 'Belirtilmemiş')) if patient.get('bmi') else _("results.unspecified", "Belirtilmemiş")],
        [_("results.obesity", "Obezite"), obesity_map[lang_short].get(int(patient.get('obesity', 0)), _("results.unspecified", "Belirtilmemiş"))],
    ]
    
    y = draw_styled_table(patient_info, y, [150, 150])

    # Laboratory Values Section
    y = draw_section_header(_("results.lab_values", "Laboratuvar Değerleri"), y)
    
    lab_fields = [
        ("AST", 'ast', 'IU/L', '5-40'),
        ("ALT", 'alt', 'IU/L', '7-56'),
        ("ALP", 'alp', 'IU/L', '44-147'),
        ("Total Bilirubin", 'total_bilirubin', 'mg/dL', '0.3-1.2'),
        ("Direct Bilirubin", 'direct_bilirubin', 'mg/dL', '0.0-0.3'),
        ("Albumin", 'albumin', 'g/dL', '3.5-5.0'),
        ("Platelets", 'trombosit', '×10³/μL', '150-450'),
        ("INR", 'inr', '% (0-1)', '0.8-1.1'),
        ("Creatinine", 'creatinine', 'mg/dL', '0.7-1.3'),
        ("AFP", 'afp', 'ng/mL', '<10'),
        ("GGT", 'ggt', 'IU/L', '9-48')
    ]
    
    lab_data = [[_("results.lab_name", "Parametre"), _("results.lab_value", "Değer"), _("results.lab_unit", "Birim"), _("results.lab_normal", "Normal")]]
    for name, key, unit, normal in lab_fields:
        value = patient.get(key, '')
        if value != '':
            lab_data.append([name, fmt(value), unit, normal])
    
    # Add clinical assessment parameters (ascites and encephalopathy) to lab data
    if patient.get('ascites') is not None:
        lab_data.append([_("form.ascites", "Asit"), get_ascites_description(patient.get('ascites', 0), lang_short), "Grade", "0-2"])
    if patient.get('encephalopathy') is not None:
        lab_data.append([_("form.encephalopathy", "Ensefalopati"), get_encephalopathy_description(patient.get('encephalopathy', 0), lang_short), "Grade", "0-2"])
    
    y = draw_styled_table(lab_data, y, [120, 80, 60, 80], header_bg=(0.2, 0.6, 0.4))

    # Risk Assessment Section
    y = draw_section_header(_("results.risk_assessment", "Risk Değerlendirme Sonuçları"), y)
    
    risk_data = [[_("results.disease", "Hastalık"), _("results.risk_value", "Risk (%)"), _("results.risk_level", "Seviye"), _("results.model", "Model")]]
    for disease, res in results.items():
        if isinstance(res, dict):
            risk_level = res.get('risk_level', res.get('classification', _("results.unspecified", "Belirtilmemiş")))
            # NAFLD special: show classification + confidence
            risk_value = res.get('risk_percentage', res.get('risk', ''))
            if isinstance(risk_value, float):
                risk_value = fmt(risk_value)
            if disease == 'nafld':
                risk_value = fmt(res.get('confidence'))
                classification = res.get('classification', _("results.unspecified", "Belirtilmemiş"))
                confidence = res.get('confidence', None)
                if confidence is not None:
                    risk_level = f"{classification} ({fmt(confidence)}%)"
                else:
                    risk_level = classification
            model = res.get('model', 'AI Model')
            risk_data.append([
                res.get('disease', disease.title()),
                risk_value,
                risk_level,
                model
            ])
    
    y = draw_styled_table(risk_data, y, [120, 80, 100, 80], header_bg=(0.6, 0.2, 0.2))

    # Traditional Scores Section - ensure title and content stay together
    # Check if we have enough space for both the header and at least 3 rows of the table
    min_space_needed = 35 + (len(traditional_scores) * 18) + 30  # header + table rows + padding
    y = check_new_page(y, min_space_needed)
    
    y = draw_section_header(_("results.traditional_scores", "Geleneksel Klinik Skorlar"), y)
    
    scores_data = [[_("results.score_name", "Skor"), _("results.score_value", "Değer")]]
    for score_name, score_value in traditional_scores.items():
        interpretation = ''
        if score_interpretations and score_name in score_interpretations:
            interp = score_interpretations[score_name]
            interpretation = interp.get('level', '')
        scores_data.append([score_name, fmt(score_value)])
    
    y = draw_styled_table(scores_data, y, [200, 100], header_bg=(0.4, 0.2, 0.6))

    # AI Assessment Section with proper HTML rendering and background
    if ai_assessment and ai_assessment.strip():
        # Ensure AI Assessment section has enough space (minimum 100px for header + some content)
        y = check_new_page(y, 100)
        
        y = draw_section_header(_("results.ai_assessment", "AI Doktor Değerlendirmesi"), y)
        
        def render_html_to_pdf_with_background(html_content, start_y):
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # First pass: calculate total height needed
            temp_y = start_y
            line_height = 14
            text_margin = margin + 15
            text_width = content_width - 30
            
            # Calculate content height
            def calculate_height(element, y_pos):
                if element.name is None:  # Text node
                    text = element.strip()
                    if text:
                        lines = wrap_text(text, font_family, 10, text_width)
                        return y_pos - (len(lines) * line_height)
                    return y_pos
                
                elif element.name == 'p':
                    text = element.get_text().strip()
                    if text:
                        lines = wrap_text(text, font_family, 10, text_width)
                        return y_pos - (len(lines) * line_height) - 5
                    return y_pos
                
                elif element.name in ['strong', 'b']:
                    text = element.get_text().strip()
                    if text:
                        lines = wrap_text(text, font_bold, 10, text_width)
                        return y_pos - (len(lines) * line_height)
                    return y_pos
                
                elif element.name in ['em', 'i']:
                    text = element.get_text().strip()
                    if text:
                        lines = wrap_text(text, font_family, 10, text_width)
                        return y_pos - (len(lines) * line_height)
                    return y_pos
                
                elif element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    text = element.get_text().strip()
                    if text:
                        lines = wrap_text(text, font_bold, 12, text_width)
                        return y_pos - (len(lines) * line_height) - 8
                    return y_pos
                
                elif element.name in ['ul', 'ol']:
                    for li in element.find_all('li', recursive=False):
                        li_text = li.get_text().strip()
                        if li_text:
                            prefix = "• " if element.name == 'ul' else "1. "
                            lines = wrap_text(f"{prefix}{li_text}", font_family, 10, text_width)
                            y_pos -= len(lines) * line_height
                    return y_pos
                
                elif element.name == 'br':
                    return y_pos - line_height
                
                elif element.name == 'div':
                    for child in element.children:
                        y_pos = calculate_height(child, y_pos)
                    return y_pos
                
                else:
                    text = element.get_text().strip()
                    if text:
                        lines = wrap_text(text, font_family, 10, text_width)
                        return y_pos - (len(lines) * line_height)
                    return y_pos
            
            # Calculate total content height
            content_end_y = start_y
            for element in soup.children:
                content_end_y = calculate_height(element, content_end_y)
            
            total_height = start_y - content_end_y + 30  # Add padding
            
            # Draw background box with gradient-like effect
            bg_y = start_y - 10
            draw_rounded_rect(margin + 5, bg_y - total_height, content_width - 10, total_height, 8, 
                            (0.96, 0.98, 1.0), (0.85, 0.90, 0.98))
            
            # Add subtle inner shadow effect
            draw_rounded_rect(margin + 7, bg_y - total_height + 2, content_width - 14, total_height - 4, 6, 
                            (0.94, 0.96, 0.99))
            
            # Now render the actual content
            # FIX: Increased top padding by changing start_y - 15 to start_y - 25
            current_y = start_y - 25  # Start inside the background box
            
            def draw_wrapped_text_with_font(text, x, y, font_name, font_size, text_color=(0.1, 0.1, 0.1)):
                """Draw text with proper word wrapping and font setting"""
                if not text.strip():
                    return y
                
                # Set font and color
                p.setFont(font_name, font_size)
                p.setFillColorRGB(*text_color)
                
                # Wrap text
                wrapped_lines = wrap_text(text, font_name, font_size, text_width)
                
                for line in wrapped_lines:
                    current_y = check_new_page(y, line_height)
                    p.drawString(x, current_y, line)
                    y -= line_height
                
                return y
            
            def process_element_with_styling(element, y_pos):
                if element.name is None:  # Text node
                    text = element.strip()
                    if text:
                        return draw_wrapped_text_with_font(text, text_margin, y_pos, font_family, 10)
                    return y_pos
                
                elif element.name == 'p':
                    text = element.get_text().strip()
                    if text:
                        y_pos = draw_wrapped_text_with_font(text, text_margin, y_pos, font_family, 10)
                        y_pos -= 5  # Extra spacing after paragraph
                    return y_pos
                
                elif element.name in ['strong', 'b']:
                    text = element.get_text().strip()
                    if text:
                        return draw_wrapped_text_with_font(text, text_margin, y_pos, font_bold, 10, (0.0, 0.0, 0.0))
                    return y_pos
                
                elif element.name in ['em', 'i']:
                    text = element.get_text().strip()
                    if text:
                        return draw_wrapped_text_with_font(text, text_margin, y_pos, font_family, 10, (0.2, 0.2, 0.2))
                    return y_pos
                
                elif element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    text = element.get_text().strip()
                    if text:
                        font_size = 14 if element.name == 'h1' else 12 if element.name == 'h2' else 11
                        y_pos = draw_wrapped_text_with_font(text, text_margin, y_pos, font_bold, font_size, (0.0, 0.0, 0.0))
                        y_pos -= 8  # Extra spacing after headers
                    return y_pos
                
                elif element.name == 'ul':
                    for li in element.find_all('li', recursive=False):
                        y_pos = check_new_page(y_pos, line_height)
                        li_text = li.get_text().strip()
                        if li_text:
                            y_pos = draw_wrapped_text_with_font(f"• {li_text}", text_margin, y_pos, font_family, 10)
                    return y_pos
                
                elif element.name == 'ol':
                    for i, li in enumerate(element.find_all('li', recursive=False), 1):
                        y_pos = check_new_page(y_pos, line_height)
                        li_text = li.get_text().strip()
                        if li_text:
                            y_pos = draw_wrapped_text_with_font(f"{i}. {li_text}", text_margin, y_pos, font_family, 10)
                    return y_pos
                
                elif element.name == 'br':
                    return y_pos - line_height
                
                elif element.name == 'div':
                    for child in element.children:
                        y_pos = process_element_with_styling(child, y_pos)
                    return y_pos
                
                else:
                    text = element.get_text().strip()
                    if text:
                        return draw_wrapped_text_with_font(text, text_margin, y_pos, font_family, 10)
                    return y_pos
            
            # Process all elements in the soup
            for element in soup.children:
                current_y = process_element_with_styling(element, current_y)
            
            return current_y - 15  # Add some bottom padding

        y = render_html_to_pdf_with_background(ai_assessment, y)
        y -= 20  # Extra spacing after AI assessment

    # Footer with disclaimer
    y = check_new_page(y, 100)
    
    # Disclaimer box
    disclaimer_height = 80
    draw_rounded_rect(margin, y - disclaimer_height, content_width, disclaimer_height, 5, (1.0, 0.95, 0.95), (0.8, 0.6, 0.6))
    
    p.setFont(font_bold, 10)
    p.setFillColorRGB(0.8, 0.2, 0.2)
    p.drawString(margin + 10, y - 15, _("results.disclaimer_title", 'TIBBİ SORUMLULUK REDDİ:'))
    
    # Properly wrap disclaimer text
    p.setFont(font_family, 8)
    p.setFillColorRGB(0.4, 0.4, 0.4)
    disclaimer = _("results.disclaimer_text", 'Bu araç laboratuvar değerleri ve klinik parametrelere dayalı risk değerlendirmesi sağlar. Sonuçlar kalifiye sağlık profesyonelleri tarafından yorumlanmalı ve klinik yargıyı veya tanı prosedürlerini değiştirmez. Yalnızca klinik araştırma ve eğitim amaçlıdır.')
    
    # Use proper word wrapping for disclaimer
    disclaimer_lines = wrap_text(disclaimer, font_family, 8, content_width - 20)
    
    text_y = y - 30
    for line in disclaimer_lines[:5]:  # Limit to 5 lines to fit in box
        p.drawString(margin + 10, text_y, line)
        text_y -= 10
    
    # Copyright
    p.setFont(font_family, 8)
    p.setFillColorRGB(0.6, 0.6, 0.6)
    p.drawString(margin, 30, '© 2025 LiverAId Risk Prediction System')
    p.drawRightString(width - margin, 30, f"Sayfa 1")

    p.save()
    buffer.seek(0)
    
    response = make_response(send_file(buffer, as_attachment=True, download_name='LiverAId_Risk_Assessment.pdf', mimetype='application/pdf'))
    response.headers['Content-Disposition'] = 'attachment; filename=LiverAId_Risk_Assessment.pdf'
    return response

# --- PDF export language keys are now managed in static/js/languages/tr.json and en.json ---
# Please update those files for any translation changes. The PDF_EXPORT_LANGUAGE_KEYS dict is no longer needed.

if __name__ == '__main__':
    print("Starting Comprehensive Liver Disease Risk Assessment System...")
    print("Supporting: Cirrhosis, HCC, and NAFLD risk predictions")
    
    # Database is initialized automatically when imported
    print("Database ready!")
    
    app.run(debug=False, host='0.0.0.0', port=5002)
