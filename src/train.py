import argparse
import joblib
from pathlib import Path
import yaml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
import pandas as pd
import numpy as np
from src.preprocess import load_and_clean, split_X_y


def build_preprocessor(X):
    num_cols = ['tenure','MonthlyCharges','TotalCharges']
    cat_cols = [c for c in X.columns if c not in num_cols]
    # Compat for sklearn versions: use sparse_output if available, otherwise sparse
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', ohe, cat_cols)
    ])
    return preprocessor


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ds = Path(cfg.get('dataset_path'))
    out = Path(cfg.get('output_dir', 'output'))
    out.mkdir(parents=True, exist_ok=True)

    if not ds.exists():
        print('Dataset not found at', ds)
        print('Please ensure the CSV is in the project folder or update config.yaml to point to it.')
        return

    df = load_and_clean(str(ds))
    X_train, X_test, y_train, y_test = split_X_y(df, test_size=cfg.get('test_size', 0.2), random_state=cfg.get('random_state', 42))

    pre = build_preprocessor(X_train)
    # simple logistic baseline
    pipe = Pipeline([
        ('pre', pre),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:,1]
    report = classification_report(y_test, preds, output_dict=True)
    print('Logistic report:\n', classification_report(y_test, preds))

    # save model and probs and PR curve data
    joblib.dump(pipe, out / 'LogisticRegression_class_weight.joblib')
    pr = precision_recall_curve(y_test, probs)
    pr_df = pd.DataFrame({'precision': pr[0], 'recall': pr[1], 'threshold': np.append(pr[2], np.nan)})
    pr_df.to_csv(out / 'pr_thresholds_logreg_classweight.csv', index=False)
    print('Artifacts saved to', out)

    # Log experiment: try MLflow, otherwise append to runs.csv
    run_info = {
        'model': 'LogisticRegression_class_weight',
        'n_train': len(X_train),
        'n_test': len(X_test),
        'accuracy': float(report.get('accuracy', 0.0)) if isinstance(report, dict) else 0.0,
        'precision_class1': float(report.get('1', {}).get('precision', 0.0)) if isinstance(report, dict) else 0.0,
        'recall_class1': float(report.get('1', {}).get('recall', 0.0)) if isinstance(report, dict) else 0.0,
        'f1_class1': float(report.get('1', {}).get('f1-score', 0.0)) if isinstance(report, dict) else 0.0,
        'model_path': str(out / 'LogisticRegression_class_weight.joblib'),
        'pr_csv': str(out / 'pr_thresholds_logreg_classweight.csv')
    }
    try:
        import mlflow
        mlflow.set_experiment(cfg.get('mlflow_experiment', 'churn_experiment'))
        with mlflow.start_run():
            mlflow.log_params({'model': 'LogisticRegression', 'class_weight': 'balanced'})
            mlflow.log_metric('accuracy', run_info['accuracy'])
            mlflow.log_metric('precision_class1', run_info['precision_class1'])
            mlflow.log_metric('recall_class1', run_info['recall_class1'])
            mlflow.log_artifact(out / 'LogisticRegression_class_weight.joblib')
            mlflow.log_artifact(out / 'pr_thresholds_logreg_classweight.csv')
        print('Logged run to MLflow')
    except Exception:
        # fallback: append to runs.csv
        runs_csv = out / 'runs.csv'
        import csv
        headers = list(run_info.keys())
        write_header = not runs_csv.exists()
        with open(runs_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()
            writer.writerow(run_info)
        print('Appended run metadata to', runs_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    main(args.config)
