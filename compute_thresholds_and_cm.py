import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report

TH_CSV = r"c:\Users\Gabriel\Downloads\LLM\output\pr_thresholds_logreg_classweight.csv"
MODEL = r"c:\Users\Gabriel\Downloads\LLM\output\LogisticRegression_class_weight.joblib"
DATA = r"c:\Users\Gabriel\Downloads\LLM\WA_Fn-UseC_-Telco-Customer-Churn.csv"

th = pd.read_csv(TH_CSV)
precision = th['precision'].values
recall = th['recall'].values
thresholds = th['threshold'].values

# high recall: first threshold with recall >= 0.85
idx_high_recall = np.where(recall >= 0.85)[0]
if len(idx_high_recall) > 0:
    idx_hr = idx_high_recall[0]
else:
    idx_hr = np.argmax(recall)

# high precision >=0.75 else max
idx_high_prec = np.where(precision >= 0.75)[0]
if len(idx_high_prec) > 0:
    idx_hp = idx_high_prec[-1]
else:
    idx_hp = np.argmax(precision)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
idx_bal = np.argmax(f1_scores)

selected = {
    'high_recall': (thresholds[idx_hr], precision[idx_hr], recall[idx_hr], f1_scores[idx_hr]),
    'balanced': (thresholds[idx_bal], precision[idx_bal], recall[idx_bal], f1_scores[idx_bal]),
    'high_precision': (thresholds[idx_hp], precision[idx_hp], recall[idx_hp], f1_scores[idx_hp])
}

print('Selected thresholds:')
for k,v in selected.items():
    print(k, 'threshold=%.6f precision=%.4f recall=%.4f f1=%.4f' % v)

# load model and data to compute confusion matrix for balanced
pipe = joblib.load(MODEL)
import pandas as pd

df = pd.read_csv(DATA)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges']).copy()
y = df['Churn'].map({'Yes':1,'No':0})
X = df.drop(columns=['Churn','customerID'], errors='ignore')
probs = pipe.predict_proba(X)[:,1]

thr = selected['balanced'][0]
preds = (probs >= thr).astype(int)
cm = confusion_matrix(y, preds)
print('\nConfusion matrix for balanced threshold (rows: true, cols: pred):')
print(cm)
print('\nClassification report:')
print(classification_report(y, preds, digits=4))

# save a small summary CSV
out = pd.DataFrame([{
    'name': k,
    'threshold': v[0],
    'precision': v[1],
    'recall': v[2],
    'f1': v[3]
} for k,v in selected.items()])
out.to_csv(r"c:\Users\Gabriel\Downloads\LLM\output\selected_thresholds_summary.csv", index=False)
print('\nSaved selected_thresholds_summary.csv')
