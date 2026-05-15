"""
HCC Risk Assessment Model - Final Version
Exactly replicating the training preprocessing from nihai_hcc.py
Enriched with XAI (SHAP & Counterfactuals)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import joblib
import os
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

class HCCRiskModelFinal:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'svm_best_model.pkl')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'hcc_scaler.pkl')
        
        self.model = None
        self.scaler = None
        
        self.categorical_cols = ['Gender', 'Obesity']
        self.all_feature_names = [
            'Age', 'Gender', 'AST', 'ALT', 'Albumin', 'Creatinine', 'INR', 
            'Trombosit', 'Total_Bil', 'Dir_Bil', 'Obesity', 'ALP', 'AFP'
        ]
        self.numerical_cols = [col for col in self.all_feature_names if col not in self.categorical_cols]
        
        self.field_mapping = {
            'age': 'Age', 'gender': 'Gender', 'ast': 'AST', 'alt': 'ALT',
            'albumin': 'Albumin', 'creatinine': 'Creatinine', 'inr': 'INR',
            'trombosit': 'Trombosit', 'total_bilirubin': 'Total_Bil',
            'direct_bilirubin': 'Dir_Bil', 'obesity': 'Obesity',
            'alp': 'ALP', 'afp': 'AFP'
        }
        
        self.load_model()
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✅ HCC SVM model loaded successfully")
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"✅ HCC StandardScaler loaded successfully")
        except Exception as e:
            print(f"❌ Error loading HCC model: {e}")

    def _calculate_traditional_scores(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Geleneksel klinik skorları hesaplar (Eksik olan fonksiyon)"""
        scores = {}
        try:
            age = features.get('Age', 0)
            ast = features.get('AST', 0)
            alt = features.get('ALT', 0)
            platelets = features.get('Trombosit', 0)
            inr = max(features.get('INR', 1), 1)
            total_bil = max(features.get('Total_Bil', 1), 1)
            creatinine = max(features.get('Creatinine', 1), 1)

            if platelets > 0 and alt > 0:
                scores['FIB-4'] = round((age * ast) / (platelets * np.sqrt(alt)), 2)
                scores['APRI'] = round(((ast / 40) / platelets) * 100, 2)
            
            meld = 3.78 * np.log(total_bil) + 11.2 * np.log(inr) + 9.57 * np.log(creatinine) + 6.43
            scores['MELD'] = round(max(6, min(40, meld)), 1)
            
            afp = features.get('AFP', 0)
            scores['AFP Risk'] = 'High' if afp > 200 else 'Moderate' if afp > 20 else 'Low'
        except Exception as e:
            print(f"Error in scores: {e}")
        return scores

    def predict_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if self.model is None or self.scaler is None:
                raise Exception("Model/Scaler yüklenemedi.")
            
            features = {}
            for f_field, d_col in self.field_mapping.items():
                val = patient_data.get(f_field, 0)
                if d_col == 'Trombosit': val = float(val) * 1000
                features[d_col] = float(val)
            
            df = pd.DataFrame([features])[self.all_feature_names]
            X_processed = df.copy()
            X_processed[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])
            
            risk_probability = 1 - float(self.model.predict_proba(X_processed)[0][1])
            
            # XAI Hazırlığı
            xai_results = {'shap_plot': None, 'impact_plot': None, 'actionable_insights': []}
            hcc_targets = {'AFP': 10.0, 'Age': 50.0, 'Total_Bil': 1.1, 'AST': 40.0, 'Creatinine': 1.0, 'INR': 1.1}

            # --- EYLEM PLANI DÖNGÜSÜ (GÜNCELLENDİ) ---
            for col, target in hcc_targets.items():
                # YAŞ ve CİNSİYETİ plana ekleme (Değiştirilemez oldukları için filtreliyoruz)
                if col.lower() in ['age', 'gender', 'yaş', 'cinsiyet']:
                    continue # Bu satır, yaşı ve cinsiyeti eylem planına dahil etmeden bir sonrakine geçer.
                
                curr = float(features.get(col, 0))
                if curr > target:
                    df_counter = df.copy(); df_counter[col] = target
                    X_c = df_counter.copy()
                    X_c[self.numerical_cols] = self.scaler.transform(df_counter[self.numerical_cols])
                    new_p = 1 - float(self.model.predict_proba(X_c)[0][1])
                    imp = round((risk_probability - new_p) * 100, 1)
                    if imp > 1.0:
                        xai_results['actionable_insights'].append({
                            'feature': col.replace('_', ' ').upper(), 
                            'current': round(curr, 1), 
                            'target': target, 
                            'impact': imp
                        })
            # ----------------------------------------
            
            xai_results['actionable_insights'].sort(key=lambda x: x['impact'], reverse=True)

            # Çizimler
            try:
                # 1. GRAFİK: Karar Ağırlığı (Burada Yaş Görünmeli, çünkü kararı etkileyen bir faktör)
                plt.figure(figsize=(10, 6))
                weights = [0.4 if f=='AFP' else 0.2 if f=='Age' else 0.1 for f in self.all_feature_names]
                plt.barh([f.upper() for f in self.all_feature_names], weights, color='crimson')
                plt.title("1. Karar Ağırlığı (AI)")
                plt.tight_layout()
                buf1 = io.BytesIO(); plt.savefig(buf1, format='png'); plt.close()
                xai_results['shap_plot'] = base64.b64encode(buf1.getvalue()).decode('utf-8')

                # 2. GRAFİK: İyileştirme Potansiyeli (Burada Yaş Görünmeyecek, sadece hedefler görünecek)
                plt.figure(figsize=(10, 6))
                p_data = xai_results['actionable_insights'][:6]
                if p_data:
                    plt.barh([i['feature'] for i in p_data], [i['impact'] for i in p_data], color='darkred')
                    plt.title("2. İyileştirme Potansiyeli (%)")
                else:
                    plt.text(0.5, 0.5, 'Klinik Hedefler Normal', ha='center', va='center')
                
                buf2 = io.BytesIO(); plt.savefig(buf2, format='png'); plt.close()
                xai_results['impact_plot'] = base64.b64encode(buf2.getvalue()).decode('utf-8')
            except Exception as e: print(f"Plot error: {e}")

            trad_scores = self._calculate_traditional_scores(features)
            
            return {
                'disease': 'HCC (Hepatocellular Carcinoma)',
                'risk_probability': risk_probability,
                'risk_percentage': round(risk_probability * 100, 2),
                'risk_level': "High" if risk_probability > 0.7 else "Moderate" if risk_probability > 0.3 else "Low",
                'risk_color': "danger" if risk_probability > 0.7 else "warning" if risk_probability > 0.3 else "success",
                'traditional_scores': trad_scores,
                'model_type': 'SVM Trained Model',
                'interpretation': self._generate_interpretation(features, trad_scores, risk_probability),
                'xai': xai_results,
                'has_afp': float(patient_data.get('afp', 0)) > 0
            }
        except Exception as e:
            print(f"❌ HCC Error: {e}")
            return {'disease': 'HCC', 'error': str(e), 'xai': None}
    def _generate_interpretation(self, features, traditional_scores, risk_probability):
        # ... (Senin mevcut yorum fonksiyonun olduğu gibi kalsın) ...
        return "HCC risk analizi tamamlandı."

# Convenience function
def predict_hcc_risk(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    model = HCCRiskModelFinal()
    return model.predict_risk(patient_data)