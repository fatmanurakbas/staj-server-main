# Comprehensive Liver Disease Risk Assessment System

A Flask-based web application for assessing the risk of multiple liver diseases including Cirrhosis, HCC (Hepatocellular Carcinoma), and NAFLD (Non-Alcoholic Fatty Liver Disease).

## Features

- **Multi-Disease Assessment**: Calculates risk for three major liver diseases simultaneously
- **Comprehensive Input**: Single form with all necessary clinical parameters
- **AI-Powered Predictions**: Uses machine learning models for risk assessment
- **Traditional Scores**: Includes clinical scores like FIB-4, APRI, MELD, etc.
- **AFP Integration**: Optional AFP input for enhanced HCC risk assessment
- **User-Friendly Interface**: Clean, responsive Bootstrap-based UI
- **Sample Data**: Quick-fill buttons for testing different risk scenarios

## Supported Diseases

### 1. Cirrhosis
**Input fields:**
- Age, Gender (Female=1, Male=2)
- AST, ALT, ALP, Platelets (Trombosit)
- Albumin, Body Mass Index, INR
- Total Bilirubin, Creatinine, Direct Bilirubin

### 2. HCC (Hepatocellular Carcinoma)
**Input fields:**
- Age, Gender, AST, ALT, Albumin, Creatinine
- INR, Platelets (Trombosit), Total Bilirubin, Direct Bilirubin
- Obesity, ALP, AFP (optional)

### 3. NAFLD (Non-Alcoholic Fatty Liver Disease)
**Input fields:**
- Age, Gender, AST, ALT, Platelets (Trombosit), Albumin
- BMI, INR, Total Bilirubin, Creatinine, Direct Bilirubin, ALP

## Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and go to `http://localhost:5000`

## Project Structure

```
final-final/
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── models/                   # Disease prediction models
│   ├── __init__.py
│   ├── cirrhosis_model.py    # Cirrhosis risk assessment
│   ├── hcc_model.py          # HCC risk assessment
│   └── nafld_model.py        # NAFLD risk assessment
├── templates/                # HTML templates
│   ├── base.html            # Base template
│   ├── index.html           # Main form
│   └── results.html         # Results display
├── static/                  # Static files
│   ├── css/
│   │   └── style.css        # Custom styles
│   └── js/
│       └── main.js          # JavaScript functionality
└── data/                    # Data files
    └── hcc_selected_parameters.csv
```

## Usage

1. **Access the Application**: Open the web interface
2. **Enter Patient Data**: Fill in the comprehensive form with patient clinical data
3. **Optional AFP**: Include AFP value for enhanced HCC assessment
4. **Calculate Risks**: Submit the form to get risk assessments for all three diseases
5. **Review Results**: View risk percentages, traditional scores, and patient data summary

## Model Integration

The system is designed for seamless model integration:

- **Mock Calculations**: Currently uses clinical rule-based mock calculations
- **Model Loading**: Automatically loads trained models if available (`.joblib` files)
- **Extensible**: Easy to replace mock calculations with actual trained models

### Adding Trained Models

Place your trained models in the `/models/` directory:
- `cirrhosis_model.joblib` - Cirrhosis risk model
- `hcc_model.joblib` - HCC risk model (without AFP)
- `hcc_model_with_afp.joblib` - HCC risk model (with AFP)
- `nafld_model.joblib` - NAFLD risk model

## API Endpoints

- `GET /` - Main form page
- `POST /calculate_risks` - Calculate risks and display results
- `POST /api/calculate_risks` - JSON API for risk calculation
- `GET /health` - Health check endpoint

## Traditional Scores

The system calculates various traditional clinical scores:

- **FIB-4**: Fibrosis-4 score for liver fibrosis assessment
- **APRI**: AST to Platelet Ratio Index
- **MELD**: Model for End-Stage Liver Disease
- **NAFLD Fibrosis Score**: Specific to NAFLD assessment
- **AST/ALT Ratio**: Liver enzyme ratio

## Sample Data

The interface includes quick-fill buttons for testing:
- **Low Risk Patient**: Minimal risk factors
- **Moderate Risk Patient**: Some elevated parameters
- **High Risk Patient**: Multiple risk factors present

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: Bootstrap 5, HTML5, JavaScript
- **Models**: Scikit-learn compatible
- **Data Format**: CSV, JSON
- **Deployment**: Heroku-ready with Procfile support

## Future Enhancements

- Integration with actual trained models
- Additional liver disease types
- Enhanced visualization
- Patient history tracking
- Export functionality (PDF reports)
- Multi-language support

## License

This project is for educational and research purposes.
