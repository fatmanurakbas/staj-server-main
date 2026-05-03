import shap
import matplotlib
matplotlib.use('Agg') # Flask için GUI olmayan arka planı kullan
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import numpy as np

class XAIEngine:
    @staticmethod
    def generate_shap_plot(model, input_df):
        """SHAP Bar Plot oluşturur ve base64 string olarak döner."""
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_df)
            
            plt.figure(figsize=(10, 6))
            # Karar üzerindeki en etkili 5 özelliği göster
            shap.plots.bar(shap_values[0], max_display=7, show=False)
            plt.title("Risk Faktörlerinin Karar Üzerindeki Etkisi (SHAP)")
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"SHAP Error: {e}")
            return None

    @staticmethod
    def get_actionable_insights(model, input_df, current_risk):
        """'Counterfactual' benzeri eylem planı oluşturur."""
        insights = []
        # Kritik parametreler ve iyileştirme hedefleri
        targets = {
            'alt': 35, 'ast': 35, 'bmi': 23, 'platelet': 250, 
            'ALT': 35, 'AST': 35, 'Body Mass Index': 23, 'Trombosit': 250
        }
        
        for feature in input_df.columns:
            if feature in targets and input_df[feature].iloc[0] > targets[feature]:
                # "Eğer bu değer hedefte olsaydı risk ne olurdu?" simülasyonu
                temp_df = input_df.copy()
                old_val = temp_df[feature].iloc[0]
                temp_df[feature] = targets[feature]
                
                # Yeni riski hesapla
                new_probs = model.predict_proba(temp_df)[0]
                # Modellerin sınıf yapıları farklı olabilir (0/1 veya 1/2)
                # Genellikle yüksek index (1 veya 2) riski temsil eder
                new_risk = new_probs[-1] 
                
                improvement = (current_risk - new_risk) * 100
                if improvement > 1: # %1'den fazla fark yaratıyorsa öneri ekle
                    insights.append({
                        'feature': feature,
                        'current': round(old_val, 1),
                        'target': targets[feature],
                        'impact': round(improvement, 1)
                    })
        
        # Etkiye göre sırala (en çok fark yaratan en üstte)
        return sorted(insights, key=lambda x: x['impact'], reverse=True)