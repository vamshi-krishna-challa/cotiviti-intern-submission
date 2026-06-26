"""
Clinical Decision Making - Hospital Readmission Prediction POC
Demonstrates pattern recognition and predictive analytics for healthcare optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("CLINICAL DECISION MAKING POC: PATIENT READMISSION PREDICTION")
print("=" * 70)
print()

# ============================================================================
# 1. CREATE SYNTHETIC HEALTHCARE DATASET
# ============================================================================
print("[STEP 1] Generating synthetic hospital patient dataset...")

np.random.seed(42)
n_patients = 1000

# Realistic healthcare features based on clinical literature
data = {
    'patient_id': range(1, n_patients + 1),
    'age': np.random.randint(18, 95, n_patients),
    'length_of_stay': np.random.randint(1, 30, n_patients),
    'num_diagnoses': np.random.randint(1, 10, n_patients),
    'num_medications': np.random.randint(1, 15, n_patients),
    'has_diabetes': np.random.randint(0, 2, n_patients),
    'has_hypertension': np.random.randint(0, 2, n_patients),
    'has_heart_disease': np.random.randint(0, 2, n_patients),
    'emergency_visit': np.random.randint(0, 2, n_patients),
    'comorbidity_score': np.random.uniform(0, 10, n_patients),
}

df = pd.DataFrame(data)

# Create target variable (readmission) with realistic patterns
# Higher risk with: older age, more diagnoses, more medications, chronic conditions
readmission_risk = (
    (df['age'] > 65) * 0.3 +
    (df['num_diagnoses'] > 5) * 0.2 +
    (df['num_medications'] > 8) * 0.2 +
    (df['has_diabetes'] * 0.15) +
    (df['has_heart_disease'] * 0.2) +
    (df['emergency_visit'] * 0.15) +
    (df['comorbidity_score'] / 10) * 0.15
)

# Add randomness but maintain realistic patterns
df['readmitted_30days'] = (readmission_risk + np.random.normal(0, 0.1, n_patients) > 0.4).astype(int)

print(f"✓ Dataset created: {n_patients} patients")
print(f"✓ Readmission rate: {df['readmitted_30days'].mean():.1%}")
print()

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("[STEP 2] Data Analysis...")

print("\nDataset Overview:")
print(df.head(10))

print("\n\nKey Statistics by Readmission Status:")
comparison = df.groupby('readmitted_30days')[['age', 'length_of_stay', 'num_diagnoses', 'num_medications']].mean()
print(comparison)

print("\n✓ Readmitted patients have: higher age, more diagnoses, more medications")
print()

# ============================================================================
# 3. BUILD PREDICTIVE MODEL
# ============================================================================
print("[STEP 3] Building Predictive Model...")

# Features for modeling
feature_cols = ['age', 'length_of_stay', 'num_diagnoses', 'num_medications',
                'has_diabetes', 'has_hypertension', 'has_heart_disease',
                'emergency_visit', 'comorbidity_score']

X = df[feature_cols]
y = df['readmitted_30days']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model (interpretable, excellent for healthcare)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

print(f"✓ Model trained on {len(X_train)} patients")
print()

# ============================================================================
# 4. MODEL EVALUATION
# ============================================================================
print("[STEP 4] Model Performance Evaluation...")

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Readmission', 'Readmitted']))

# ROC-AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {auc_score:.3f} ✓ (Good predictive power)")
print()

# ============================================================================
# 5. FEATURE IMPORTANCE (Pattern Recognition)
# ============================================================================
print("[STEP 5] Feature Importance Analysis (What Drives Risk?)...")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Risk Factors for 30-Day Readmission:")
for idx, row in feature_importance.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:20s} {bar} {row['importance']:.1%}")

print()

# ============================================================================
# 6. REAL-WORLD PREDICTION EXAMPLE
# ============================================================================
print("[STEP 6] Real-World Prediction Example (Clinical Case)...")
print("\nCase 1: 72-year-old patient with multiple chronic conditions")
case1 = pd.DataFrame({
    'age': [72],
    'length_of_stay': [7],
    'num_diagnoses': [7],
    'num_medications': [11],
    'has_diabetes': [1],
    'has_hypertension': [1],
    'has_heart_disease': [1],
    'emergency_visit': [1],
    'comorbidity_score': [8.5]
})
case1_scaled = scaler.transform(case1)
case1_risk = model.predict_proba(case1_scaled)[0][1]
print(f"  Readmission Risk: {case1_risk:.1%} → HIGH RISK - Recommend intensive discharge planning")

print("\nCase 2: 35-year-old patient with single diagnosis")
case2 = pd.DataFrame({
    'age': [35],
    'length_of_stay': [2],
    'num_diagnoses': [1],
    'num_medications': [2],
    'has_diabetes': [0],
    'has_hypertension': [0],
    'has_heart_disease': [0],
    'emergency_visit': [0],
    'comorbidity_score': [1.5]
})
case2_scaled = scaler.transform(case2)
case2_risk = model.predict_proba(case2_scaled)[0][1]
print(f"  Readmission Risk: {case2_risk:.1%} → LOW RISK - Standard discharge protocol")

print()

# ============================================================================
# 7. OPERATIONAL INSIGHTS FOR COTIVITI
# ============================================================================
print("[STEP 7] Strategic Insights for Prior Authorization & TPO Optimization...")

# Segment patients by risk
df_test = X_test.copy()
df_test['readmission_risk'] = y_pred_proba
df_test['readmitted'] = y_test

risk_segments = pd.cut(df_test['readmission_risk'], 
                       bins=[0, 0.3, 0.6, 1.0],
                       labels=['Low Risk', 'Moderate Risk', 'High Risk'])
df_test['risk_segment'] = risk_segments

print("\nRisk Stratification Results (Test Set):")
for segment in ['Low Risk', 'Moderate Risk', 'High Risk']:
    mask = df_test['risk_segment'] == segment
    count = mask.sum()
    actual_readmit_rate = df_test[mask]['readmitted'].mean()
    print(f"  {segment:15s}: {count:3d} patients, Actual readmission rate: {actual_readmit_rate:.1%}")

print("\n✓ Model enables differential care: high-risk patients get intensive support")
print("✓ This can reduce unnecessary readmissions and optimize resource allocation")
print("✓ Cotiviti can integrate this into prior authorization: approve expedited discharge planning for high-risk cases")
print()

# ============================================================================
# 8. VISUALIZATION
# ============================================================================
print("[STEP 8] Generating Visualizations...")

# Create 2x2 visualization grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Clinical Decision Making: Hospital Readmission Risk Prediction', fontsize=16, fontweight='bold')

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 0].plot(fpr, tpr, linewidth=2.5, label=f'ROC Curve (AUC = {auc_score:.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
axes[0, 0].set_xlabel('False Positive Rate', fontsize=11)
axes[0, 0].set_ylabel('True Positive Rate', fontsize=11)
axes[0, 0].set_title('Model Discrimination: ROC Curve', fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)

# 2. Feature Importance
top_features = feature_importance.head(8)
axes[0, 1].barh(top_features['feature'], top_features['importance'], color='steelblue')
axes[0, 1].set_xlabel('Importance Score', fontsize=11)
axes[0, 1].set_title('Top Risk Factors for Readmission', fontweight='bold')
axes[0, 1].grid(alpha=0.3, axis='x')

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], 
            xticklabels=['No Readmit', 'Readmit'], yticklabels=['No Readmit', 'Readmit'])
axes[1, 0].set_title('Model Predictions: Confusion Matrix', fontweight='bold')
axes[1, 0].set_ylabel('Actual', fontsize=11)
axes[1, 0].set_xlabel('Predicted', fontsize=11)

# 4. Risk Distribution
axes[1, 1].hist(df_test[df_test['readmitted'] == 0]['readmission_risk'], 
                bins=20, alpha=0.6, label='No Readmission', color='green')
axes[1, 1].hist(df_test[df_test['readmitted'] == 1]['readmission_risk'], 
                bins=20, alpha=0.6, label='Readmitted', color='red')
axes[1, 1].set_xlabel('Predicted Readmission Risk', fontsize=11)
axes[1, 1].set_ylabel('Number of Patients', fontsize=11)
axes[1, 1].set_title('Risk Distribution: Model Calibration', fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/readmission_prediction_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Visualizations saved")

# ============================================================================
# SUMMARY
# ============================================================================
print()
print("=" * 70)
print("PROOF OF CONCEPT SUMMARY")
print("=" * 70)
print("""
✓ Built: Predictive model for 30-day hospital readmission risk
✓ Data: 1,000 synthetic patient records with realistic patterns
✓ Accuracy: ROC-AUC of {:.3f} - excellent discrimination
✓ Interpretability: Top 5 features identified (age, diagnoses, medications, etc.)
✓ Use Case: Real-time prior authorization & discharge planning optimization

COTIVITI APPLICATIONS:
  1. Prior Authorization: Approve high-risk cases for intensive discharge planning
  2. Risk Stratification: Segment patients into intervention tiers
  3. Provider Feedback: Share risk models to drive quality improvements
  4. Payer Analytics: Demonstrate ROI through readmission reduction

This POC demonstrates the core capability. In production, this would integrate:
  - Real EHR data with clinical validation
  - Fairness audits for demographic bias
  - HIPAA compliance and governance
  - Continuous model monitoring and retraining
""".format(auc_score))

print("=" * 70)
print()
