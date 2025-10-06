"""
analise_churn_balance.py
Testa estratégias de balanceamento (nenhuma, class_weight, SMOTE) para LogisticRegression e RandomForest,
comparando métricas focadas em recall. Salva resultados em output/model_comparison_balance.csv
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

try:
    from imblearn.over_sampling import SMOTE
    imblearn_available = True
except Exception:
    imblearn_available = False

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

strategies = ['baseline', 'class_weight', 'smote']
models = ['LogisticRegression', 'RandomForest']
results = []

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

for strat in strategies:
    if strat == 'smote' and not imblearn_available:
        print('SMOTE não disponível, pulando SMOTE. Para instalar: pip install imbalanced-learn')
        continue

    # prepare training data depending on strategy
    if strat == 'baseline' or strat == 'class_weight':
        X_train = X_train_full.copy()
        y_train = y_train_full.copy()
    elif strat == 'smote':
        # aplicar pré-processamento numérico + ohe antes do SMOTE? SMOTE works on numeric arrays after encoding
        # Então primeiro aplicar transformer fit_transform no X_train_full
        transformer = preprocessor.fit(X_train_full)
        X_train_encoded = transformer.transform(X_train_full)
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train_encoded, y_train_full)

    for m in models:
        print(f'\nStrategy={strat} Model={m}')
        if strat == 'class_weight':
            if m == 'LogisticRegression':
                clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            else:
                clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

            pipe = Pipeline([
                ('pre', preprocessor),
                ('clf', clf)
            ])

            start = time.time()
            pipe.fit(X_train, y_train)
            elapsed = time.time() - start

        elif strat == 'baseline':
            if m == 'LogisticRegression':
                clf = LogisticRegression(max_iter=1000)
            else:
                clf = RandomForestClassifier(n_estimators=200, random_state=42)
            pipe = Pipeline([
                ('pre', preprocessor),
                ('clf', clf)
            ])
            start = time.time()
            pipe.fit(X_train, y_train)
            elapsed = time.time() - start

        elif strat == 'smote':
            # model trained on resampled numeric array; need to wrap classifier without preprocessing
            if m == 'LogisticRegression':
                clf = LogisticRegression(max_iter=1000)
            else:
                clf = RandomForestClassifier(n_estimators=200, random_state=42)
            start = time.time()
            clf.fit(X_res, y_res)
            elapsed = time.time() - start

            # for testing, we need probabilities/predictions: transform X_test
            X_test_enc = preprocessor.transform(X_test)
            preds = clf.predict(X_test_enc)
            probs = clf.predict_proba(X_test_enc)[:,1]

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)

            results.append({
                'strategy': strat,
                'model': m,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'roc_auc': auc,
                'time_s': elapsed
            })

            # salvar modelo
            fname = f'{m}_{strat}.joblib'
            joblib.dump(clf, os.path.join(OUT_DIR, fname))
            print('Saved', fname)
            continue

        # para baseline e class_weight: prever no X_test
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:,1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        results.append({
            'strategy': strat,
            'model': m,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': auc,
            'time_s': elapsed
        })

        # salvar pipeline
        fname = f'{m}_{strat}.joblib'
        joblib.dump(pipe, os.path.join(OUT_DIR, fname))
        print('Saved', fname)

# salvar resumo
pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, 'model_comparison_balance.csv'), index=False)
print('\nResumo salvo em', os.path.join(OUT_DIR, 'model_comparison_balance.csv'))
