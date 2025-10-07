import joblib
import pandas as pd
from pathlib import Path


def predict_csv(model_path: str, csv_path: str):
    pipe = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges']).copy()
    X = df.drop(columns=['Churn','customerID'], errors='ignore')
    probs = pipe.predict_proba(X)[:,1]
    df['prob_churn'] = probs
    return df[['prob_churn'] + [c for c in df.columns if c not in ['prob_churn']][:5]]

if __name__ == '__main__':
    import sys
    model = sys.argv[1]
    csv = sys.argv[2]
    print(predict_csv(model, csv).head())
