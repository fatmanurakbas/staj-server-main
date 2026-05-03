"""
Cirrhosis Risk Assessment Model
Enhanced with real trained model and clinical scoring systems
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import shap
import matplotlib
matplotlib.use('Agg') # Sunucu tarafında hata almamak için gerekli
import matplotlib.pyplot as plt
import io
import base64

class CirrhosisRiskModel:
    def __init__(self):
        # Try to load the new XGBoost model first, fallback to old model
        self.model_path_xgb = os.path.join(os.path.dirname(__file__), 'cirrhosis_model_xgb.pkl')
        self.scaler_path_xgb = os.path.join(os.path.dirname(__file__), 'cirrhosis_scaler_xgb.pkl')
        self.imputer_path_xgb = os.path.join(os.path.dirname(__file__), 'cirrhosis_imputer_xgb.pkl')
        
        # Fallback to old model paths
        self.model_path = os.path.join(os.path.dirname(__file__), 'cirrhosis_model.pkl')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'cirrhosis_scaler.pkl')
        self.imputer_path = os.path.join(os.path.dirname(__file__), 'cirrhosis_imputer.pkl')
        
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')
        
        self.model = None
        self.scaler = None
        self.imputer = None
        self.model_type = None
        
        # XGBoost model uses these feature names (from actual dataset)
        self.xgb_feature_names = [
            'age', 'gender', 'ast', 'alt', 'platelet', 'albumin', 'bmi', 
            'inr', 'total_bilirubin', 'creatin', 'direct_bilirubin', 'alp'
        ]
        
        # Legacy model feature names (for backward compatibility)
        self.legacy_feature_names = [
            'Age', 'Gender (Female=1, Male=2)', 'AST', 'ALT', 'Trombosit', 'Albumin', 
            'Body Mass Index', 'INR', 'Total Bilirubin', 'Creatinine', 'Direct Bilirubin', 'ALP'
        ]
        
        # Map form field names to model feature names
        self.field_mapping = {
            'age': 'age',
            'gender': 'gender',  # XGBoost model: 0=Female, 1=Male (matches notebook)
            'ast': 'ast',
            'alt': 'alt',
            'trombosit': 'platelet',  # Map trombosit -> platelet
            'albumin': 'albumin',
            'bmi': 'bmi',
            'inr': 'inr',
            'total_bilirubin': 'total_bilirubin',
            'creatinine': 'creatin',  # Map creatinine -> creatin
            'direct_bilirubin': 'direct_bilirubin',
            'alp': 'alp'
        }
        
        # Legacy field mapping for backward compatibility
        self.legacy_field_mapping = {
            'age': 'Age',
            'gender': 'Gender (Female=1, Male=2)',
            'ast': 'AST',
            'alt': 'ALT',
            'trombosit': 'Trombosit',  
            'albumin': 'Albumin',
            'bmi': 'Body Mass Index',
            'inr': 'INR',
            'total_bilirubin': 'Total Bilirubin',
            'creatinine': 'Creatinine',  
            'direct_bilirubin': 'Direct Bilirubin',
            'alp': 'ALP'
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing tools"""
        try:
            # Try to load XGBoost model first
            if (os.path.exists(self.model_path_xgb) and 
                os.path.exists(self.scaler_path_xgb) and 
                os.path.exists(self.imputer_path_xgb)):
                
                self.model = joblib.load(self.model_path_xgb)
                self.scaler = joblib.load(self.scaler_path_xgb)
                self.imputer = joblib.load(self.imputer_path_xgb)
                self.model_type = 'XGBoost'
                self.feature_names = self.xgb_feature_names
                print("✅ XGBoost cirrhosis model loaded successfully")
                
            # Fallback to legacy model
            elif (os.path.exists(self.model_path) and 
                  os.path.exists(self.scaler_path) and 
                  os.path.exists(self.imputer_path)):
                
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.imputer = joblib.load(self.imputer_path)
                self.model_type = 'Legacy'
                self.feature_names = self.legacy_feature_names
                print("✅ Legacy cirrhosis model loaded successfully")
                
            else:
                print("⚠️  Cirrhosis model files not found, using rule-based calculations")
                self.model = None
                self.model_type = 'Rule-based'
                
        except Exception as e:
            print(f"❌ Error loading cirrhosis model: {e}")
            self.model = None
            self.model_type = 'Rule-based'
    
    def predict_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gelişmiş Çift Grafikli XAI (SHAP & Counterfactuals) Risk Tahmini
        """
        try:
            # 1. VERİ HAZIRLAMA VE MAPPING
            mapped_data = {}
            missing_fields = []
            field_mapping = self.field_mapping if self.model_type == 'XGBoost' else self.legacy_field_mapping
            
            for form_field, model_field in field_mapping.items():
                if form_field in patient_data:
                    mapped_data[model_field] = patient_data[form_field]
                else:
                    missing_fields.append(form_field)
            
            if missing_fields:
                raise ValueError(f"Eksik alanlar: {', '.join(set(missing_fields))}")
            
            features = [float(mapped_data[field]) for field in self.feature_names]
            X = pd.DataFrame([features], columns=self.feature_names)
            
            # 2. TAHMİN (PREDICTION)
            risk_probability = 0.5
            risk_class = 0
            if self.model is not None:
                if self.model_type == 'XGBoost':
                    risk_probability = float(self.model.predict_proba(X)[0][1])
                    risk_class = int(self.model.predict(X)[0])
                else:
                    X_scaled = self.scaler.transform(self.imputer.transform(X))
                    risk_probability = float(self.model.predict_proba(X_scaled)[0][1])
                    risk_class = int(self.model.predict(X_scaled)[0])
            else:
                risk_probability, risk_class = self._enhanced_rule_based_prediction(mapped_data)

            # 3. XAI (AÇIKLANABİLİRLİK) ANALİZİ
            # Artık iki farklı grafik alanı tanımlıyoruz
            xai_results = {'shap_plot': None, 'impact_plot': None, 'actionable_insights': []}
            
            # --- A. Karşı Olgusal (Counterfactual) Öneriler Hesaplama ---
            targets = {'alt': 56.0, 'ast': 40.0, 'bmi': 30.0, 'trombosit': 250.0, 
                       'total_bilirubin': 1.2, 'creatin': 1.2, 'inr': 1.2}

            for feature in self.feature_names:
                f_key = feature.lower().replace('body mass index', 'bmi').replace('trombosit', 'platelet').replace('creatinine', 'creatin')
                if f_key in targets:
                    curr = float(mapped_data[feature])
                    if curr > targets[f_key]:
                        # Simülasyon: Değer düşerse risk ne kadar azalır?
                        impact = round(min(((curr - targets[f_key]) / curr) * 15, 25.0), 1)
                        if impact > 1.0:
                            xai_results['actionable_insights'].append({
                                'feature': feature.upper(), 'current': round(curr, 1), 
                                'target': targets[f_key], 'impact': impact
                            })
            xai_results['actionable_insights'].sort(key=lambda x: x['impact'], reverse=True)

            # --- B. ÇİFT GRAFİK ÇİZİMİ ---
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io, base64

            if self.model is not None:
                # GRAFİK 1: Karar Ağırlığı (AI Neye Baktı?)
                try:
                    plt.figure(figsize=(10, 6))
                    try:
                        import shap
                        explainer = shap.TreeExplainer(self.model)
                        shap_v = explainer.shap_values(X.values)
                        if isinstance(shap_v, list): shap_v = shap_v[1]
                        vals = np.abs(shap_v).mean(0) if len(shap_v.shape) > 1 else np.abs(shap_v[0])
                        plt.barh([f.upper() for f in self.feature_names], vals, color='teal')
                    except:
                        plt.barh([f.upper() for f in self.feature_names], self.model.feature_importances_, color='cadetblue')
                    
                    plt.title("1. Karar Ağırlığı (Neden?)")
                    plt.tight_layout()
                    buf1 = io.BytesIO()
                    plt.savefig(buf1, format='png', bbox_inches='tight'); plt.close()
                    xai_results['shap_plot'] = base64.b64encode(buf1.getvalue()).decode('utf-8')
                except Exception as e: print(f"Grafik 1 Hatası: {e}")

                # GRAFİK 2: İyileştirme Potansiyeli (% Azalış Oranları)
                try:
                    plt.figure(figsize=(10, 6))
                    plot_data = xai_results['actionable_insights'][:6]
                    if plot_data:
                        names = [i['feature'] for i in plot_data]
                        impacts = [i['impact'] for i in plot_data]
                        plt.barh(names, impacts, color='forestgreen')
                        plt.title("2. İyileştirme Potansiyeli (% Etki)")
                        plt.xlabel("Riski Azaltma Oranı (%)")
                    else:
                        plt.text(0.5, 0.5, 'Tüm Değerler İdeal', ha='center', va='center')
                    
                    plt.tight_layout()
                    buf2 = io.BytesIO()
                    plt.savefig(buf2, format='png', bbox_inches='tight'); plt.close()
                    xai_results['impact_plot'] = base64.b64encode(buf2.getvalue()).decode('utf-8')
                except Exception as e: print(f"Grafik 2 Hatası: {e}")

            # 4. SONUÇLARI DÖNDÜR
            traditional_scores = self._calculate_traditional_scores(mapped_data)
            interpretation = self._generate_interpretation(mapped_data, traditional_scores, risk_probability)
            
            risk_level = "High" if risk_probability >= 0.7 else "Moderate" if risk_probability >= 0.3 else "Low"
            risk_color = "danger" if risk_level == "High" else "warning" if risk_level == "Moderate" else "success"

            return {
                'disease': 'Cirrhosis',
                'risk_probability': risk_probability,
                'risk_percentage': round(risk_probability * 100, 2),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'traditional_scores': traditional_scores,
                'model_type': 'Raw XGBoost Model',
                'interpretation': interpretation,
                'confidence': round(abs(risk_probability - 0.5) * 2, 2),
                'xai': xai_results # shap_plot ve impact_plot içeriyor
            }
  
        except Exception as e:
            print(f"❌ Tahmin Hatası: {e}")
            return {
                'disease': 'Cirrhosis',
                'error': str(e),
                'risk_probability': 0.5,
                'risk_percentage': 50.0,
                'risk_level': "Error",
                'risk_color': "secondary",
                'xai': None
            }
    
    def _enhanced_rule_based_prediction(self, data: Dict[str, Any]) -> Tuple[float, int]:
        """
        Enhanced rule-based prediction when trained model is not available
        Based on clinical knowledge and the patterns from the training data
        """
        try:
            # Check for required fields - use field names based on model type
            if self.model_type == 'XGBoost':
                required_fields = ['age', 'ast', 'alt', 'platelet', 'albumin', 'inr', 'total_bilirubin']
                age_field = 'age'
                ast_field = 'ast'
                alt_field = 'alt'
                platelet_field = 'platelet'
                albumin_field = 'albumin'
                inr_field = 'inr'
                total_bil_field = 'total_bilirubin'
                bmi_field = 'bmi'
                alp_field = 'alp'
            else:
                # Legacy field names
                required_fields = ['Age', 'AST', 'ALT', 'Trombosit', 'Albumin', 'INR', 'Total Bilirubin']
                age_field = 'Age'
                ast_field = 'AST'
                alt_field = 'ALT'
                platelet_field = 'Trombosit'
                albumin_field = 'Albumin'
                inr_field = 'INR'
                total_bil_field = 'Total Bilirubin'
                bmi_field = 'Body Mass Index'
                alp_field = 'ALP'
            
            missing_fields = []
            for field in required_fields:
                if field not in data:
                    missing_fields.append(field)
            
            if missing_fields:
                raise ValueError(f"Missing required fields for rule-based prediction: {', '.join(missing_fields)}")
            
            # Key indicators for cirrhosis risk
            risk_score = 0.0
            
            # Age factor (higher risk with age, especially > 50)
            age = data[age_field]
            if age > 60:
                risk_score += 0.15
            elif age > 50:
                risk_score += 0.10
            elif age > 40:
                risk_score += 0.05
            
            # AST/ALT ratio (>1 suggests cirrhosis)
            ast = data[ast_field]
            alt = data[alt_field]
            if alt > 0:
                ast_alt_ratio = ast / alt
                if ast_alt_ratio > 2:
                    risk_score += 0.20
                elif ast_alt_ratio > 1.5:
                    risk_score += 0.15
                elif ast_alt_ratio > 1:
                    risk_score += 0.10
            
            # Platelet count (thrombocytopenia indicates portal hypertension)
            platelet = data[platelet_field]
            if platelet < 100:
                risk_score += 0.25
            elif platelet < 150:
                risk_score += 0.15
            elif platelet < 200:
                risk_score += 0.10
            
            # Albumin (hypoalbuminemia indicates liver dysfunction)
            albumin = data[albumin_field]
            if albumin < 3.0:
                risk_score += 0.20
            elif albumin < 3.5:
                risk_score += 0.15
            elif albumin < 4.0:
                risk_score += 0.10
            
            # INR (coagulopathy indicates liver dysfunction)
            inr = data[inr_field]
            if inr > 1.5:
                risk_score += 0.20
            elif inr > 1.3:
                risk_score += 0.15
            elif inr > 1.1:
                risk_score += 0.10
            
            # Total bilirubin (hyperbilirubinemia)
            total_bil = data[total_bil_field]
            if total_bil > 2.0:
                risk_score += 0.15
            elif total_bil > 1.5:
                risk_score += 0.10
            elif total_bil > 1.2:
                risk_score += 0.05
            
            # BMI (obesity can worsen liver disease) - optional field
            bmi = data.get(bmi_field, 25)  # BMI can have default as it's often calculated
            if bmi > 35:
                risk_score += 0.10
            elif bmi > 30:
                risk_score += 0.05
            
            # ALP elevation - optional field
            alp = data.get(alp_field, 100)  # ALP can have default as it's supplementary
            if alp > 200:
                risk_score += 0.10
            elif alp > 150:
                risk_score += 0.05
            
            # Normalize to probability (0-1)
            risk_probability = min(risk_score, 0.95)  # Cap at 95%
            risk_probability = max(risk_probability, 0.05)  # Floor at 5%
            
            # Determine class
            risk_class = 1 if risk_probability > 0.5 else 0
            
            print(f"🧮 Enhanced rule-based cirrhosis risk calculation: {risk_probability:.3f}")
            
            return risk_probability, risk_class
            
        except Exception as e:
            print(f"❌ Error in enhanced rule-based prediction: {e}")
            return 0.5, 0
    
    def _calculate_traditional_scores(self, mapped_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate traditional clinical scores"""
        scores = {}
        
        try:
            # Check required fields for score calculations
            required_for_fib4 = ['Age', 'AST', 'ALT', 'Trombosit']
            required_for_apri = ['AST', 'Trombosit']
            required_for_meld = ['INR', 'Total Bilirubin', 'Creatinine']
            
            # FIB-4 Score
            if all(field in mapped_data for field in required_for_fib4):
                age = mapped_data['Age']
                ast = mapped_data['AST']
                alt = mapped_data['ALT']
                platelet = mapped_data['Trombosit']
                
                if platelet > 0 and alt > 0:
                    fib4 = (age * ast) / (platelet * (alt ** 0.5))
                    scores['FIB-4'] = round(fib4, 2)
                    
                    if fib4 < 1.45:
                        scores['FIB-4 Interpretation'] = "Low probability of advanced fibrosis"
                    elif fib4 < 3.25:
                        scores['FIB-4 Interpretation'] = "Intermediate probability - further evaluation needed"
                    else:
                        scores['FIB-4 Interpretation'] = "High probability of advanced fibrosis"
            else:
                missing_fib4 = [field for field in required_for_fib4 if field not in mapped_data]
                scores['FIB-4 Error'] = f"Missing fields: {', '.join(missing_fib4)}"
            
            # APRI Score
            if all(field in mapped_data for field in required_for_apri):
                ast = mapped_data['AST']
                platelet = mapped_data['Trombosit']
                
                if platelet > 0:
                    apri = (ast / 40) / (platelet / 1000) * 100
                    scores['APRI'] = round(apri, 2)
                    
                    if apri < 0.5:
                        scores['APRI Interpretation'] = "Low probability of significant fibrosis"
                    elif apri < 1.5:
                        scores['APRI Interpretation'] = "Intermediate probability"
                    else:
                        scores['APRI Interpretation'] = "High probability of significant fibrosis"
            else:
                missing_apri = [field for field in required_for_apri if field not in mapped_data]
                scores['APRI Error'] = f"Missing fields: {', '.join(missing_apri)}"
            
            # MELD Score (for liver disease severity)
            if all(field in mapped_data for field in required_for_meld):
                inr = mapped_data['INR']
                total_bil = mapped_data['Total Bilirubin']
                creatinine = mapped_data['Creatinine']
                
                meld = 3.78 * np.log(total_bil) + 11.2 * np.log(inr) + 9.57 * np.log(creatinine) + 6.43
                meld = max(6, min(40, meld))  # MELD score range 6-40
                scores['MELD'] = round(meld, 1)
                
                if meld < 10:
                    scores['MELD Interpretation'] = "Low mortality risk"
                elif meld < 15:
                    scores['MELD Interpretation'] = "Moderate mortality risk"
                elif meld < 20:
                    scores['MELD Interpretation'] = "High mortality risk"
                else:
                    scores['MELD Interpretation'] = "Very high mortality risk"
            else:
                missing_meld = [field for field in required_for_meld if field not in mapped_data]
                scores['MELD Error'] = f"Missing fields: {', '.join(missing_meld)}"
                
        except Exception as e:
            print(f"⚠️ Error calculating traditional scores: {e}")
            scores['Error'] = str(e)
        
        return scores
    
    def _generate_interpretation(self, mapped_data: Dict[str, Any], 
                                traditional_scores: Dict[str, Any], 
                                risk_probability: float) -> str:
        """Generate clinical interpretation"""
        try:
            interpretation = []
            
            # Risk level interpretation
            if risk_probability < 0.3:
                interpretation.append("Low cirrhosis risk based on current laboratory values.")
            elif risk_probability < 0.7:
                interpretation.append("Moderate cirrhosis risk detected. Enhanced monitoring recommended.")
            else:
                interpretation.append("High cirrhosis risk indicated. Immediate clinical evaluation advised.")
            
            # Key findings - only analyze if fields are present
            findings = []
            
            # Check AST/ALT ratio
            if 'AST' in mapped_data and 'ALT' in mapped_data:
                ast = mapped_data['AST']
                alt = mapped_data['ALT']
                if alt > 0:
                    ratio = ast / alt
                    if ratio > 1:
                        findings.append(f"AST/ALT ratio of {ratio:.2f} suggests possible liver damage")
            
            # Check platelet count
            if 'Trombosit' in mapped_data:
                platelet = mapped_data['Trombosit']
                if platelet < 150:
                    findings.append(f"Low platelet count ({platelet}) may indicate portal hypertension")
            
            # Check albumin
            if 'Albumin' in mapped_data:
                albumin = mapped_data['Albumin']
                if albumin < 3.5:
                    findings.append(f"Low albumin ({albumin}) suggests impaired liver synthesis")
            
            # Check INR
            if 'INR' in mapped_data:
                inr = mapped_data['INR']
                if inr > 1.3:
                    findings.append(f"Elevated INR ({inr}) indicates coagulopathy")
            
            if findings:
                interpretation.append("Key findings: " + "; ".join(findings))
            
            # Traditional scores summary
            if 'FIB-4' in traditional_scores:
                interpretation.append(f"FIB-4 score: {traditional_scores['FIB-4']} - {traditional_scores.get('FIB-4 Interpretation', '')}")
            
            return " ".join(interpretation)
            
        except Exception as e:
            return f"Error generating interpretation: {str(e)}"
        
        # Platelet assessment
        platelets = float(mapped_data['Trombosit'])
        if platelets < 100:
            interpretation.append("Thrombocytopenia may indicate portal hypertension.")
        elif platelets < 150:
            interpretation.append("Mild thrombocytopenia noted.")
        
        # Albumin assessment
        albumin = float(mapped_data['Albumin'])
        if albumin < 3.0:
            interpretation.append("Low albumin suggests impaired hepatic synthetic function.")
        elif albumin < 3.5:
            interpretation.append("Mild hypoalbuminemia noted.")
        
        # INR assessment
        inr = float(mapped_data['INR'])
        if inr > 1.5:
            interpretation.append("Elevated INR indicates coagulopathy.")
        elif inr > 1.2:
            interpretation.append("Mild coagulopathy noted.")
        
        # Traditional score interpretation
        if 'FIB-4' in traditional_scores:
            fib4 = traditional_scores['FIB-4']
            if fib4 < 1.45:
                interpretation.append("FIB-4 <1.45 suggests low probability of advanced fibrosis.")
            elif fib4 > 3.25:
                interpretation.append("FIB-4 >3.25 suggests high probability of advanced fibrosis.")
        
        if 'APRI' in traditional_scores:
            apri = traditional_scores['APRI']
            if apri < 0.5:
                interpretation.append("APRI <0.5 suggests low probability of significant fibrosis.")
            elif apri > 1.5:
                interpretation.append("APRI >1.5 suggests high probability of significant fibrosis.")
        
        if 'MELD' in traditional_scores:
            meld = traditional_scores['MELD']
            if meld < 10:
                interpretation.append("MELD <10 indicates low short-term mortality risk.")
            elif meld > 15:
                interpretation.append("MELD >15 indicates increased short-term mortality risk.")
        
        # Risk-based recommendations
        if risk_probability > 0.7:
            interpretation.append("High risk warrants urgent hepatology consultation.")
        elif risk_probability > 0.3:
            interpretation.append("Moderate risk - consider hepatology referral and regular monitoring.")
        else:
            interpretation.append("Low risk - routine monitoring and lifestyle modifications advised.")
        
        return " ".join(interpretation)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if model is available"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}


# Convenience function for easy import
def predict_cirrhosis_risk(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to predict cirrhosis risk"""
    model = CirrhosisRiskModel()
    return model.predict_risk(patient_data)
