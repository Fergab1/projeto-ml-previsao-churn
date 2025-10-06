"""
analise_churn.py
Script reproduzindo o fluxo do notebook:
- Carrega o CSV
- Limpeza e pré-processamento
- EDA básica (salva alguns gráficos em PNG)
- Treina um baseline LogisticRegression e salva o pipeline

Uso: python analise_churn.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import sklearn

DATA_PATH = r"c:\Users\Gabriel\Downloads\LLM\WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_DIR = r"c:\Users\Gabriel\Downloads\LLM\output"
os.makedirs(OUT_DIR, exist_ok=True)

print('Carregando dados...')
df = pd.read_csv(DATA_PATH)
print('Shape:', df.shape)

# Conversões e limpeza
print('Convertendo TotalCharges para numérico...')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print('TotalCharges NaN:', df['TotalCharges'].isnull().sum())
df = df.dropna(subset=['TotalCharges']).copy()

df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')
df = df.drop(columns=['customerID'])

# Corrigir colunas com 'No internet service' e 'No phone service'
# Muitas colunas usam essas strings; para features booleanas, convertê-las para 'No'
cols_replace_no = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','MultipleLines']
for c in cols_replace_no:
    if c in df.columns:
        df[c] = df[c].replace({'No internet service':'No', 'No phone service':'No'})

# EDA simples
print('\nContagem de Churn:')
print(df['Churn'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title('Distribuição de Churn')
plt.savefig(os.path.join(OUT_DIR, 'churn_count.png'))
plt.close()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(df['tenure'], bins=30)
plt.title('Tenure')
plt.subplot(1,2,2)
sns.histplot(df['MonthlyCharges'], bins=30)
plt.title('MonthlyCharges')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tenure_monthly.png'))
plt.close()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Churn por tipo de contrato')
plt.savefig(os.path.join(OUT_DIR, 'churn_by_contract.png'))
plt.close()

# Modelagem baseline
X = df.drop(columns=['Churn'])
y = df['Churn'].map({'Yes':1,'No':0})

num_cols = ['tenure','MonthlyCharges','TotalCharges']
cat_cols = [c for c in X.columns if c not in num_cols]

# Build a OneHotEncoder compatible with the installed scikit-learn version
v = sklearn.__version__.split('.')
major, minor = int(v[0]), int(v[1]) if len(v) > 1 else 0
if major > 1 or (major == 1 and minor >= 2):
    # sklearn >= 1.2 uses `sparse_output`
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    # older sklearn versions use `sparse`
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', ohe, cat_cols)
])

pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print('\nTreinando modelo...')
pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:,1]

print('\nClassification report:')
print(classification_report(y_test, preds))
print('ROC AUC:', roc_auc_score(y_test, probs))

# salvar pipeline
model_path = os.path.join(OUT_DIR, 'churn_baseline_pipeline.joblib')
joblib.dump(pipeline, model_path)
print('Pipeline salvo em', model_path)

print('\nAlguns artefatos foram salvos em', OUT_DIR)
