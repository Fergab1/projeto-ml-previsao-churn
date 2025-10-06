"""
plot_pr_thresholds.py
Carrega pipeline LogisticRegression_class_weight.joblib, calcula curva precision-recall
Gera: output/pr_curve_logreg_classweight.png e output/pr_thresholds_logreg_classweight.csv
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

MODEL_PATH = r"c:\Users\Gabriel\Downloads\LLM\output\LogisticRegression_class_weight.joblib"
DATA_PATH = r"c:\Users\Gabriel\Downloads\LLM\WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_DIR = r"c:\Users\Gabriel\Downloads\LLM\output"
os.makedirs(OUT_DIR, exist_ok=True)

print('Carregando modelo...')
pipe = joblib.load(MODEL_PATH)
print('Carregando dados...')
df = pd.read_csv(DATA_PATH)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges']).copy()
df = df.drop(columns=['customerID'])
y = df['Churn'].map({'Yes':1,'No':0})
X = df.drop(columns=['Churn'])

# obter probabilidades
probs = pipe.predict_proba(X)[:,1]

precision, recall, thresholds = precision_recall_curve(y, probs)
avg_prec = average_precision_score(y, probs)

# salvar CSV com thresholds (note: thresholds has len = len(precision)-1)
th_df = pd.DataFrame({
    'threshold': np.append(thresholds, 1.0),
    'precision': precision,
    'recall': recall
})
th_df.to_csv(os.path.join(OUT_DIR, 'pr_thresholds_logreg_classweight.csv'), index=False)
print('Saved thresholds CSV')

# plot
plt.figure(figsize=(8,6))
plt.plot(recall, precision, marker='.', label=f'AP={avg_prec:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve - LogisticRegression (class_weight)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, 'pr_curve_logreg_classweight.png'))
print('Saved PR curve PNG')
