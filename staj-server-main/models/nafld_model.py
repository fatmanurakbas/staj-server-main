"""
NAFLD (Non-Alcoholic Fatty Liver Disease) Risk Assessment Model
Uses CatBoost model to classify between NAFL and NASH
Input fields: age, gender, AST, ALT, trombosit, albumin, bmi, inr, 
              total bilirubin, creatinine, direct bilirubin, ALP
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pickle
import os
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

class NAFLDRiskModel:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'models-temp', 'nafld', 'catboost_model.pkl')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'nafld_scaler.pkl')
        self.imputer_path = os.path.join(os.path.dirname(__file__), 'nafld_imputer.pkl')
        self.model = None
        self.scaler = None
        self.imputer = None
        
        # Features expected by the CatBoost model (from training data)
        self.feature_names = [
            'Age', 'Gender (Female=1, Male=2)', 'AST', 'ALT', 'Trombosit', 'Albumin', 
            'Body Mass Index', 'INR', 'Total Bilirubin', 'Creatinine', 'Direct Bilirubin', 'ALP'
        ]
        
        # Initialize preprocessing tools
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        
        self.load_model()
    
    def load_model(self):
        """Load the trained CatBoost model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print("✅ NAFLD CatBoost model loaded successfully")
            else:
                print("⚠️  NAFLD CatBoost model file not found, using rule-based calculations")
                self.model = None
                
        except Exception as e:
            print(f"❌ Error loading NAFLD CatBoost model: {e}")
            self.model = None
    
    def predict_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """MAFLD tahmini ve Çift Grafikli Gelişmiş XAI analizi"""
        try:
            # 1. Veri Hazırlama ve Mapping
            field_mapping = {
                'age': 'Age', 'gender': 'Gender (Female=1, Male=2)', 'ast': 'AST', 'alt': 'ALT', 
                'trombosit': 'Trombosit', 'albumin': 'Albumin', 'bmi': 'Body Mass Index', 
                'inr': 'INR', 'total_bilirubin': 'Total Bilirubin', 'creatinine': 'Creatinine', 
                'direct_bilirubin': 'Direct Bilirubin', 'alp': 'ALP'
            }
            mapped_data = {v: float(patient_data.get(k, 0)) for k, v in field_mapping.items()}
            features = [mapped_data[field] for field in self.feature_names]
            X = pd.DataFrame([features], columns=self.feature_names)
            
            # 2. Tahmin Mantığı
            prediction, classification, confidence, risk_color = 1, "NAFL", 50.0, "warning"
            if self.model is not None:
                prediction = self.model.predict(X)[0]
                probs = self.model.predict_proba(X)[0]
                if prediction == 1:
                    classification, risk_color, confidence = "NAFL", "warning", float(probs[0] * 100)
                else:
                    classification, risk_color, confidence = "NASH", "danger", float(probs[1] * 100)
            else:
                classification, _, risk_color, confidence = self._mock_classification(patient_data)

            # 3. XAI ANALİZİ (Çift Grafik Hazırlığı)
            xai_results = {'shap_plot': None, 'impact_plot': None, 'actionable_insights': []}
            
            # --- A. Karşı Olgusal (Counterfactual) Öneriler Hesaplama ---
            # 'decrease': Değerin düşmesi gerekiyor (örn: BMI, ALT, AST)
            # 'increase': Değerin yükselmesi gerekiyor (örn: Trombosit, Albumin)
            targets = {
                'Body Mass Index': {'target': 30.0, 'direction': 'decrease'},
                'ALT': {'target': 56.0, 'direction': 'decrease'},
                'AST': {'target': 40.0, 'direction': 'decrease'},
                'Total Bilirubin': {'target': 1.2, 'direction': 'decrease'},
                'ALP': {'target': 147.0, 'direction': 'decrease'},
                'Trombosit': {'target': 150.0, 'direction': 'increase'}, # Trombosit alt sınırı
                'Albumin': {'target': 3.5, 'direction': 'increase'}       # Albumin alt sınırı
            }
            
            for feat, target_info in targets.items():
                curr = mapped_data.get(feat, 0)
                target_val = target_info['target']
                direction = target_info['direction']
                
                show_insight = False
                impact = 0.0
                weight = 20 if feat == 'Body Mass Index' else 12
                
                if direction == 'decrease' and curr > target_val:
                    impact = round(min(((curr - target_val) / curr) * weight, 28.0), 1)
                    show_insight = True
                elif direction == 'increase' and curr < target_val:
                    impact = round(min(((target_val - curr) / target_val) * weight, 28.0), 1)
                    show_insight = True
                    
                if show_insight and impact > 1.0:
                    xai_results['actionable_insights'].append({
                        'feature': feat.upper(), 
                        'current': round(curr, 1), 
                        'target': target_val, 
                        'operator': '<' if direction == 'decrease' else '>',
                        'impact': impact
                    })
            xai_results['actionable_insights'].sort(key=lambda x: x['impact'], reverse=True)

            # --- B. ÇİFT GRAFİK ÇİZİMİ ---
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io, base64, numpy as np

            # GRAFİK 1: Karar Ağırlığı (SHAP - Gerçek veya Dinamik Simüle Açıklanabilirlik)
            try:
                plt.figure(figsize=(10, 6))
                if self.model is not None:
                    explainer = shap.TreeExplainer(self.model)
                    shap_values = explainer(X)

                    if isinstance(shap_values, list):
                        shap_vals = shap_values[-1]
                    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                        shap_vals = shap_values[0, :, -1]
                    else:
                        shap_vals = shap_values[0]

                    shap.plots.bar(shap_vals, max_display=7, show=False)
                    plt.title("1. Karar Ağırlığı (SHAP) - MAFLD")
                else:
                    # MODEL YOKSA: Sabit grafik yerine hastanın gerçek sapmalarına göre dinamik mock-SHAP çiziyoruz
                    deviations = {
                        'AST': max(0.0, (mapped_data.get('AST', 0) - 40) / 40),
                        'ALT': max(0.0, (mapped_data.get('ALT', 0) - 56) / 56),
                        'BMI': max(0.0, (mapped_data.get('Body Mass Index', 0) - 25) / 25),
                        'TROMBOSIT': max(0.0, (150 - mapped_data.get('Trombosit', 0)) / 150) if mapped_data.get('Trombosit', 0) < 150 else 0.0,
                        'ALBUMIN': max(0.0, (3.5 - mapped_data.get('Albumin', 0)) / 3.5) if mapped_data.get('Albumin', 0) < 3.5 else 0.0,
                        'INR': max(0.0, (mapped_data.get('INR', 0) - 1.2) / 1.2)
                    }
                    sorted_devs = sorted(deviations.items(), key=lambda x: x[1], reverse=False) # Barh için küçükten büyüğe
                    features_to_plot = [k for k, v in sorted_devs]
                    values_to_plot = [max(v, 0.05) for k, v in sorted_devs] # Tamamen boş kalmasın diye min 0.05
                    
                    plt.barh(features_to_plot, values_to_plot, color='gold')
                    plt.title("1. Karar Ağırlığı (AI Teşhis Sebebi - Simülasyon)")

                plt.tight_layout()
                buf1 = io.BytesIO()
                plt.savefig(buf1, format='png', bbox_inches='tight'); plt.close()
                xai_results['shap_plot'] = base64.b64encode(buf1.getvalue()).decode('utf-8')
            except Exception as e: print(f"MAFLD Grafik 1 Hatası: {e}")

            # GRAFİK 2: İyileştirme Potansiyeli (Azalış Oranları)
            try:
                plt.figure(figsize=(10, 6))
                plot_data = xai_results['actionable_insights'][:6]
                if plot_data:
                    names = [i['feature'] for i in plot_data]
                    vals = [i['impact'] for i in plot_data]
                    plt.barh(names[::-1], vals[::-1], color='gold') # Büyük olan üstte görünecek şekilde sıraladık
                    plt.title("2. İyileştirme Potansiyeli (%)")
                    plt.xlabel("Risk Azaltma Potansiyeli (%)")
                else:
                    plt.text(0.5, 0.5, 'Klinik Değerler İdeal', ha='center', va='center')
                
                plt.tight_layout()
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png', bbox_inches='tight'); plt.close()
                xai_results['impact_plot'] = base64.b64encode(buf2.getvalue()).decode('utf-8')
            except Exception as e: print(f"MAFLD Grafik 2 Hatası: {e}")

            # 4. SONUÇLARI DÖNDÜR
            return {
                'disease': 'MAFLD Classification',
                'classification': classification,
                'confidence': confidence,
                'risk_percentage': confidence,
                'risk_color': risk_color,
                'model_type': 'CatBoost Trained Model' if self.model is not None else 'Rule-based Mock Model',
                'interpretation': self._generate_interpretation(patient_data, {}, classification),
                'xai': xai_results
            }
        except Exception as e:
            print(f"❌ MAFLD Tahmin Hatası: {e}")
            return {'disease': 'MAFLD', 'error': str(e), 'risk_percentage': 50.0, 'xai': None}
    
    def _mock_classification(self, patient_data: Dict[str, Any]) -> tuple:
        """
        Mock classification for NAFL vs NASH when model is not available
        
        Args:
            patient_data: Patient data dictionary
            
        Returns:
            Tuple with (classification, description, color, confidence)
        """
        # Rule-based classification using clinical indicators
        score = 0.0
        
        # Age factor (higher age increases NASH risk)
        age = float(patient_data.get('Age', patient_data.get('age', 0)))
        if age > 50:
            score += 0.3
        elif age > 40:
            score += 0.2
        
        # BMI factor (higher BMI increases NASH risk)
        bmi = float(patient_data.get('BMI', patient_data.get('bmi', 0)))
        if bmi > 35:
            score += 0.4
        elif bmi > 30:
            score += 0.3
        elif bmi > 25:
            score += 0.2
        
        # Liver enzyme levels (higher levels suggest NASH)
        ast = float(patient_data.get('AST', patient_data.get('ast', 0)))
        alt = float(patient_data.get('ALT', patient_data.get('alt', 0)))
        if ast > 80 or alt > 80:
            score += 0.4
        elif ast > 40 or alt > 40:
            score += 0.3
        elif ast > 30 or alt > 30:
            score += 0.2
        
        # AST/ALT ratio (higher ratio may indicate NASH)
        if ast > 0 and alt > 0:
            ast_alt_ratio = ast / alt
            if ast_alt_ratio > 1.5:
                score += 0.3
            elif ast_alt_ratio > 1.0:
                score += 0.2
        
        # Platelet count (lower platelets may indicate NASH)
        platelets = float(patient_data.get('Trombosit', patient_data.get('trombosit', 0)))
        if platelets < 150:
            score += 0.2
        elif platelets < 100:
            score += 0.3
        
        # Albumin (low albumin may indicate NASH)
        albumin = float(patient_data.get('Albumin', patient_data.get('albumin', 0)))
        if albumin < 3.5:
            score += 0.2
        elif albumin < 4.0:
            score += 0.1
        
        # INR (elevated in advanced disease)
        inr = float(patient_data.get('INR', patient_data.get('inr', 0)))
        if inr > 1.3:
            score += 0.2
        elif inr > 1.1:
            score += 0.1
        
        # Classify based on score
        if score > 0.6:
            return "NASH", "Non-Alcoholic Steatohepatitis (Inflammatory)", "danger", min(score * 100, 95.0)
        else:
            return "NAFL", "Non-Alcoholic Fatty Liver (Simple Steatosis)", "warning", min((1-score) * 100, 95.0)
    
    def _calculate_traditional_scores(self, patient_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate traditional NAFLD scores"""
        scores = {}
        
        try:
            # NAFLD Fibrosis Score (NFS)
            age = float(patient_data.get('Age', patient_data.get('age', 0)))
            bmi = float(patient_data.get('BMI', patient_data.get('bmi', 0)))
            diabetes = 0  # Assuming no diabetes data, could be enhanced
            ast = float(patient_data.get('AST', patient_data.get('ast', 0)))
            alt = float(patient_data.get('ALT', patient_data.get('alt', 0)))
            platelets = max(float(patient_data.get('Trombosit', patient_data.get('trombosit', 0))), 1)
            albumin = float(patient_data.get('Albumin', patient_data.get('albumin', 0)))
            
            # AST/ALT ratio
            ast_alt_ratio = ast / max(alt, 1)
            
            # NFS calculation
            nfs = (-1.675 + 0.037 * age + 0.094 * bmi + 
                   1.13 * diabetes + 0.99 * ast_alt_ratio - 
                   0.013 * platelets - 0.66 * albumin)
            scores['NFS'] = nfs
            
            # FIB-4 Score (also applicable to NAFLD)
            fib4 = (age * ast) / (platelets * np.sqrt(alt))
            scores['FIB-4'] = fib4
            
            # APRI Score 
            apri = ((ast / 40) / platelets) * 100
            scores['APRI'] = apri
            
            # BMI Score (important for NAFLD)
            if bmi < 18.5:
                bmi_category = "Underweight"
            elif bmi < 25:
                bmi_category = "Normal"
            elif bmi < 30:
                bmi_category = "Overweight"
            else:
                bmi_category = "Obese"
            
            scores['BMI_Category'] = bmi_category
            
        except Exception as e:
            print(f"Error calculating traditional scores: {e}")
        
        return scores
    
    def _generate_interpretation(self, patient_data: Dict[str, Any], 
                                traditional_scores: Dict[str, float], 
                                classification: str) -> str:
        """Generate clinical interpretation for NAFL vs NASH classification"""
        interpretation = []
        
        # Classification-based interpretation
        if classification == "NASH":
            interpretation.append("NASH (Non-Alcoholic Steatohepatitis) is characterized by liver inflammation and may progress to fibrosis.")
            interpretation.append("This condition requires active monitoring and intervention.")
        else:
            interpretation.append("NAFL (Non-Alcoholic Fatty Liver) is simple steatosis without significant inflammation.")
            interpretation.append("This is a milder form but still requires lifestyle modifications.")
        
        # BMI assessment
        bmi = float(patient_data.get('BMI', patient_data.get('bmi', 0)))
        if bmi >= 30:
            interpretation.append("Obesity (BMI ≥30) significantly increases progression risk.")
        elif bmi >= 25:
            interpretation.append("Overweight status (BMI 25-29.9) is a moderate risk factor.")
        
        # Liver enzyme assessment
        ast = float(patient_data.get('AST', patient_data.get('ast', 0)))
        alt = float(patient_data.get('ALT', patient_data.get('alt', 0)))
        if ast > 40 or alt > 40:
            interpretation.append("Elevated liver enzymes suggest hepatic inflammation.")
        
        # AST/ALT ratio interpretation
        if ast > 0 and alt > 0:
            ast_alt_ratio = ast / alt
            if ast_alt_ratio > 1.5:
                interpretation.append("AST/ALT ratio >1.5 may indicate more advanced disease.")
        
        # Traditional score interpretation
        if 'NFS' in traditional_scores:
            nfs = traditional_scores['NFS']
            if nfs < -1.455:
                interpretation.append("NFS <-1.455 suggests low probability of advanced fibrosis.")
            elif nfs > 0.676:
                interpretation.append("NFS >0.676 suggests high probability of advanced fibrosis.")
            else:
                interpretation.append("NFS in intermediate range - further evaluation may be needed.")
        
        if 'FIB-4' in traditional_scores:
            fib4 = traditional_scores['FIB-4']
            if fib4 < 1.30:
                interpretation.append("FIB-4 <1.30 suggests low risk of advanced fibrosis.")
            elif fib4 > 2.67:
                interpretation.append("FIB-4 >2.67 suggests high risk of advanced fibrosis.")
        
        # Classification-based recommendations
        if classification == "NASH":
            interpretation.append("NASH requires lifestyle intervention, regular monitoring, and possible medical treatment.")
        else:
            interpretation.append("NAFL management focuses on lifestyle modifications and monitoring for progression.")
        
        return " ".join(interpretation)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if model is available"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}


# Convenience function for easy import
def predict_nafld_classification(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to predict NAFLD classification (NAFL vs NASH)"""
    model = NAFLDRiskModel()
    return model.predict_risk(patient_data)
