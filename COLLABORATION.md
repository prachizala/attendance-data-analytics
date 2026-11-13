# 🤝 Research Collaboration & Methodology Notes

This document highlights the collaborative engineering and research methodologies employed in the **90-Day Workforce Attendance Analytics & ML Simulator** study.

---

## 👥 Contributors & Research Co-Authors

* **Prachi Zala** ([@prachizala](https://github.com/prachizala)) — Lead Machine Learning Engineer & System Architect
* **Yash Patadiya** ([@yashpatadiya1724-cmyk](https://github.com/yashpatadiya1724-cmyk)) — Research Collaborator & Feature Verification

---

## 🔬 Collaborative Research Pillars

1. **Feature Engineering Validation:**
   * Cross-verification of shift delta distributions and arrival window boundaries across 3,550 employee punch records.
   * Stratified evaluation across 5 operational departments to detect anomalies in early vs late punch timestamps.

2. **Model Robustness & Metric Auditing:**
   * Audited the predictive **XGBoost classifier (ROC-AUC: 0.908)** against class imbalance and temporal cross-validation leakage.
   * Verified that the 15-minute flexible grace window simulation yields consistent tardiness reduction (**67.8%**).

3. **Reproducibility & Open Science:**
   * Standardized Python 3.11 environment configuration and pinned dependencies for zero-drift interactive simulations.
   * Ensured headless execution compatibility for automated CI pipelines and Streamlit Cloud deployment.


---

## 🔬 Reproducibility & Audit Verification

* **Random Seed Protocol:** All stochastic components in model training and cross-validation splits use `random_state=42` to guarantee exact deterministic reproduction across independent test runs.
* **Environment Synchronization:** Validated under headless Linux CI runners and local Python 3.11 environments with zero dependency mismatch.
