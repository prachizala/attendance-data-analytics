# 📋 Model Card: Workforce Late-Arrival Predictive Classifier

## 1. Model Details
- **Architecture:** Gradient Boosted Decision Trees (`XGBClassifier`)
- **Version:** 1.0.0 (Production Flagship)
- **Frameworks:** `xgboost>=2.0.0`, `scikit-learn>=1.3.0`, `pandas>=2.0.0`
- **Trained By:** Prachi Zala (Academic Project & Workforce Intelligence Case Study)
- **License:** MIT

---

## 2. Intended Use & Target
- **Primary Objective:** Forecast probability of severe morning late arrivals ($>15\text{ minutes}$) across enterprise workforce operations.
- **Decision Context:** Powers the interactive Streamlit HR Policy Simulator (`streamlit_app.py`) to demonstrate transit friction absorption under 15-minute flexible arrival bands.

---

## 3. Training & Evaluation Protocol
- **Dataset Size:** 3,550 total shift records across 50 employees over 90 calendar days.
- **Evaluation Protocol:** 5-Fold Stratified Cross-Validation on training split (2,840 shifts) + independent holdout test split (710 shifts).
- **Benchmark Performance:**
  - **5-Fold CV ROC-AUC:** $0.8823 \pm 0.0167$
  - **Holdout Test ROC-AUC:** $0.9085$
  - **Test Accuracy:** $90.8\%$
  - **Test Precision:** $73.9\%$

---

## 4. Feature Gain Hierarchy
1. **Commute Distance ($40.8\%$):** Urban transit distance across Ahmedabad metro (SG Highway & SP Ring Road).
2. **Commute Distance × Monday ($24.6\%$):** Monday peak morning transit choke point interaction.
3. **Department Structural Factor ($14.2\%$):** Departmental transit exposure (Operations, Sales, Engineering, HR, Design).
4. **Employee Baseline Reliability ($12.1\%$):** Historical punctuality distribution.
