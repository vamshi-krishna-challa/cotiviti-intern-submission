# Clinical Decision Making in Healthcare
## Hospital Readmission Risk Prediction POC

**Applicant:** Vamshi Challa  
**University:** Louisiana Tech University  
**Position:** Temp Intern - GenAI Sci  

---

## 📋 Deliverables

### 1. Strategic Report
- **File:** `Clinical_Decision_Making_Report.docx`
- **Content:** 2-page analysis with 7 peer-reviewed citations (APA format)
- **Focus:** Market opportunity, strategic recommendations, competitive positioning

### 2. Proof of Concept
- **File:** `readmission_prediction_poc.py`
- **Model:** Random Forest classifier
- **Performance:** 94.7% ROC-AUC on 1,000 patient dataset
- **Data:** 80/20 train/test split (800 training, 200 test patients)

### 3. Results Visualization
- **File:** `readmission_prediction_analysis.png`
- **Content:** 4-panel analysis
  - ROC Curve (AUC: 94.7%)
  - Feature Importance (Age, Medications, Diagnoses)
  - Confusion Matrix (95% sensitivity, 67% specificity)
  - Risk Distribution (Clear separation)

### 4. Professional Presentation
- **File:** `Clinical_Decision_Making_PNNL_Format.pptx`
- **Format:** 10 professional slides
- **Content:** Problem → Research → Model → Applications → Implementation Roadmap

### 5. Video Demonstration
- **File:** `cotiviti_compressed.mp4`
- **Duration:** 6-7 minutes
- **Size:** 23 MB
- **Content:**
  - Introduction to clinical decision making challenge
  - Research methodology and approach
  - **Live Python POC execution** (showing 94.7% ROC-AUC)
  - Real-world clinical case examples (High Risk vs Low Risk patients)
  - Implementation roadmap (4 months to production)
  - Strategic vision and competitive positioning

---

## 🎯 Key Results

✓ **Model Accuracy:** 94.7% ROC-AUC (excellent discrimination)  
✓ **Sensitivity:** 95% (catches high-risk patients)  
✓ **Specificity:** 67% (avoids over-treatment)  
✓ **Risk Stratification:** Low 5.3%, Moderate 40.7%, High 92.9%  
✓ **Top Risk Factors:** Age (21.6%), Medications (16.3%), Diagnoses (15.2%)  

---

## 💡 Strategic Applications for Cotiviti

1. **Prior Authorization Intelligence**
   - Embed readmission risk into PA workflows
   - Approve high-risk cases for intensive discharge planning
   - Low-risk cases: standard discharge protocol

2. **Real-Time Claim Anomaly Detection**
   - Flag coding deviations at submission
   - Reduce denials and accelerate clean claims
   - Prevent inappropriate care at source

3. **Risk Stratification Pipeline**
   - Segment patients into intervention tiers
   - Optimize resource allocation for outcomes
   - Enable differential care strategies

---

## 📈 Business Impact

- **Cost Reduction:** 33% reduction in unnecessary discharge planning costs
- **Sensitivity:** 95% (defensible from liability perspective)
- **Implementation Timeline:** 4 months from validation to national deployment
- **ROI:** Potential $1.76B savings at 10% scale
- **Market Opportunity:** $17.6B annual hospital readmission penalties

---

## 🔧 How to Use the POC

### Run Locally
```bash
pip install scikit-learn pandas matplotlib seaborn numpy
python readmission_prediction_poc.py
```

### View Results
Opens a 4-panel visualization showing:
- ROC curve with 94.7% AUC
- Feature importance rankings
- Confusion matrix analysis
- Risk distribution calibration

### Understand Output
