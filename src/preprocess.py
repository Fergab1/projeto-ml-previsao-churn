import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges']).copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    cols_replace_no = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','MultipleLines']
    for c in cols_replace_no:
        if c in df.columns:
            df[c] = df[c].replace({'No internet service':'No', 'No phone service':'No'})
    return df


def split_X_y(df: pd.DataFrame, target: str = 'Churn', test_size: float = 0.2, random_state: int = 42):
    y = df[target].map({'Yes':1, 'No':0})
    X = df.drop(columns=[target])
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
