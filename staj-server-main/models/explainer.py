import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import numpy as np

class XAIEngine:
    @staticmethod
    def generate_shap_plot(model, input_df, model_type='tree'):
        """SHAP Bar Plot oluşturur ve base64 string olarak döner.
        
        Args:
            model: Eğitilmiş model
            input_df: Tek satırlı DataFrame (hasta verisi)
            model_type: 'tree', 'linear', veya 'svm'
        """
        try:
            if model_type == 'tree':
                explainer = shap.TreeExplainer(model)
            elif model_type == 'linear':
                explainer = shap.LinearExplainer(model, input_df)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, input_df)

            shap_values = explainer(input_df)

            # --- SHAP yapısını kontrol et ve risk sınıfını seç ---
            if isinstance(shap_values, list):
                # Çok sınıflı model: shap_values = [class0, class1, ...]
                shap_plot_values = shap_values[-1]
            elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                # Explanation objesi, shape=(1, n_features, n_classes)
                shap_plot_values = shap_values[0, :, -1]
            else:
                # Binary log-odds çıktısı, shape=(1, n_features)
                shap_plot_values = shap_values[0]

            plt.figure(figsize=(10, 6))
            shap.plots.bar(shap_plot_values, max_display=7, show=False)
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
    def generate_shap_waterfall(model, input_df, model_type='tree'):
        """SHAP Waterfall plot - tek hasta için detaylı açıklama."""
        try:
            if model_type == 'tree':
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, input_df)

            shap_values = explainer(input_df)

            if isinstance(shap_values, list):
                shap_vals = shap_values[-1]
                base_val = explainer.expected_value[-1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                shap_vals = shap_values[0, :, -1]
                base_val = explainer.expected_value[-1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                shap_vals = shap_values[0]
                base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value

            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap.Explanation(values=shap_vals, 
                                                   base_values=base_val,
                                                   data=input_df.iloc[0],
                                                   feature_names=input_df.columns.tolist()),
                                 max_display=7, show=False)
            plt.title("Kararın Detaylı Açıklaması (SHAP Waterfall)")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"SHAP Waterfall Error: {e}")
            return None

    @staticmethod
    def get_actionable_insights(model, input_df, current_risk):
        """Counterfactual benzeri eylem planı oluşturur."""
        insights = []
        targets = {
            'alt': 35, 'ast': 35, 'bmi': 23, 'platelet': 250,
            'ALT': 35, 'AST': 35, 'Body Mass Index': 23, 'Trombosit': 250
        }

        for feature in input_df.columns:
            if feature in targets and input_df[feature].iloc[0] > targets[feature]:
                temp_df = input_df.copy()
                old_val = temp_df[feature].iloc[0]
                temp_df[feature] = targets[feature]

                new_probs = model.predict_proba(temp_df)[0]
                new_risk = new_probs[-1]

                improvement = (current_risk - new_risk) * 100
                if improvement > 1:
                    insights.append({
                        'feature': feature,
                        'current': round(old_val, 1),
                        'target': targets[feature],
                        'impact': round(improvement, 1)
                    })

        return sorted(insights, key=lambda x: x['impact'], reverse=True)

    @staticmethod
    def analyze_model(model, input_df, feature_names, current_risk=None):
        """Tek çağrıda hem SHAP plot hem de içgörüleri üretir."""
        result = {}

        # SHAP plot
        result['shap_plot'] = XAIEngine.generate_shap_plot(model, input_df)

        # Actionable insights
        if current_risk is not None and hasattr(model, 'predict_proba'):
            result['actionable_insights'] = XAIEngine.get_actionable_insights(model, input_df, current_risk)

        return result
