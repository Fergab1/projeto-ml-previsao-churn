"""
analise_churn_compare.py
Treina LogisticRegression, RandomForest e XGBoost (se disponível), compara métricas no conjunto de teste
Salva o melhor modelo em output/best_model.joblib
"""
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib

try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception:
    xgb_available = False

import sklearn
v = sklearn.__version__.split('.')
major, minor = int(v[0]), int(v[1]) if len(v) > 1 else 0

DATA_PATH = r"c:\Users\Gabriel\Downloads\LLM\WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_DIR = r"c:\Users\Gabriel\Downloads\LLM\output"
os.makedirs(OUT_DIR, exist_ok=True)

print('Carregando dados...')
df = pd.read_csv(DATA_PATH)
# limpeza mínima
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges']).copy()
df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')
df = df.drop(columns=['customerID'])
cols_replace_no = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','MultipleLines']
for c in cols_replace_no:
    if c in df.columns:
        df[c] = df[c].replace({'No internet service':'No', 'No phone service':'No'})

X = df.drop(columns=['Churn'])
y = df['Churn'].map({'Yes':1,'No':0})

num_cols = ['tenure','MonthlyCharges','TotalCharges']
cat_cols = [c for c in X.columns if c not in num_cols]

# compat OneHotEncoder
if major > 1 or (major == 1 and minor >= 2):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', ohe, cat_cols)
])

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42)
}
if xgb_available:
    models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
else:
    print('XGBoost não disponível; será ignorado. Para instalar: pip install xgboost')

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

results = {}

for name, clf in models.items():
    print(f'\nTreinando {name}...')
    start = time.time()
    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', clf)
    ])
    pipe.fit(X_train, y_train)
    elapsed = time.time() - start

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc,
        'time_s': elapsed,
        'pipeline': pipe
    }

    print('Time (s):', round(elapsed, 2))
    print('Accuracy:', round(acc, 4))
    print('Precision:', round(prec, 4))
    print('Recall:', round(rec, 4))
    print('F1:', round(f1, 4))
    print('ROC AUC:', round(auc, 4))

# escolher melhor pelo AUC
best_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
best = results[best_name]
print('\nMelhor modelo por ROC AUC:', best_name)
print('Métricas:', {k: round(v,4) for k,v in best.items() if k!='pipeline'})

best_path = os.path.join(OUT_DIR, 'best_model.joblib')
joblib.dump(best['pipeline'], best_path)
print('Melhor pipeline salvo em', best_path)

# salvar resumo em csv
summary = []
for k, v in results.items():
    summary.append({
        'model': k,
        'accuracy': v['accuracy'],
        'precision': v['precision'],
        'recall': v['recall'],
        'f1': v['f1'],
        'roc_auc': v['roc_auc'],
        'time_s': v['time_s']
    })

pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, 'model_comparison.csv'), index=False)
print('Resumo salvo em', os.path.join(OUT_DIR, 'model_comparison.csv'))
