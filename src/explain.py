"""
Compute SHAP (or permutation importance fallback) for the trained model and save artifacts to output/
Saves:
 - output/shap_summary.png (or permutation_importance.png)
 - output/shap_topk.csv (top-3 features per test row)
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from src.preprocess import load_and_clean, split_X_y

NUM_COLS = ['tenure','MonthlyCharges','TotalCharges']


def get_feature_names(pre, X_columns, num_cols=NUM_COLS):
    # pre is ColumnTransformer; we'll build feature names consistent with the preprocessor used in train.py
    cat_cols = [c for c in X_columns if c not in num_cols]
    feature_names = []
    # numeric first
    feature_names.extend(num_cols)
    # attempt to extract categories from the fitted OneHotEncoder
    try:
        # locate transformer named 'cat' in pre.transformers_
        for name, trans, cols in pre.transformers_:
            if name == 'cat':
                ohe = trans
                break
        else:
            ohe = None
        if ohe is not None and hasattr(ohe, 'categories_'):
            for col, cats in zip(cat_cols, ohe.categories_):
                for cat in cats:
                    feature_names.append(f"{col}={cat}")
        else:
            # fallback: use raw cat_cols
            feature_names.extend(cat_cols)
    except Exception:
        feature_names.extend(cat_cols)
    return feature_names


def main(cfg_path='config.yaml'):
    cfg = yaml.safe_load(open(cfg_path)) if Path(cfg_path).exists() else {}
    ds_path = Path(cfg.get('dataset_path', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
    out = Path(cfg.get('output_dir', 'output'))
    out.mkdir(parents=True, exist_ok=True)

    if not ds_path.exists():
        print('Dataset not found at', ds_path)
        return
    if not (out / 'LogisticRegression_class_weight.joblib').exists():
        print('Model artifact not found in', out)
        return

    print('Loading data and model...')
    df = load_and_clean(str(ds_path))
    X_train, X_test, y_train, y_test = split_X_y(df, test_size=0.2, random_state=42)
    pipe = joblib.load(out / 'LogisticRegression_class_weight.joblib')
    pre = pipe.named_steps['pre']
    clf = pipe.named_steps['clf']

    print('Building transformed datasets...')
    X_train_t = pre.transform(X_train)
    X_test_t = pre.transform(X_test)

    feature_names = get_feature_names(pre, X_train.columns)

    # Try SHAP LinearExplainer for linear models
    try:
        import shap
        print('Using SHAP LinearExplainer')
        explainer = shap.LinearExplainer(clf, X_train_t, feature_perturbation='interventional')
        shap_values = explainer.shap_values(X_test_t)
        # shap_values shape may be (n_features,) or (n_samples, n_features)
        sv = np.array(shap_values)
        # summary plot
        plt.figure(figsize=(8,6))
        shap.summary_plot(sv, features=X_test_t, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out / 'shap_summary.png')
        plt.close()
        # top-3 per row
        abs_sv = np.abs(sv)
        topk_idx = np.argsort(-abs_sv, axis=1)[:, :3]
        rows = []
        for i, idxs in enumerate(topk_idx):
            names = [feature_names[j] if j < len(feature_names) else f'F{j}' for j in idxs]
            rows.append({'index': i, 'top1': names[0], 'top2': names[1], 'top3': names[2]})
        pd.DataFrame(rows).to_csv(out / 'shap_topk.csv', index=False)
        print('SHAP artifacts saved to', out)
        return
    except Exception as e:
        print('SHAP failed:', e)

    # Fallback: permutation importance
    try:
        from sklearn.inspection import permutation_importance
        print('Computing permutation importance (fallback)')
        r = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
        importances = r.importances_mean
        # bar plot
        names = list(X_test.columns)
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(8,6))
        plt.bar([names[i] for i in order], importances[order])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(out / 'permutation_importance.png')
        # Save top-3 per sample as the same top-3 global (since permutation is global)
        top3 = [names[i] for i in order[:3]]
        rows = []
        for i in range(len(X_test)):
            rows.append({'index': i, 'top1': top3[0], 'top2': top3[1], 'top3': top3[2]})
        pd.DataFrame(rows).to_csv(out / 'shap_topk.csv', index=False)
        print('Permutation importance artifacts saved to', out)
    except Exception as e:
        print('Permutation importance also failed:', e)


if __name__ == '__main__':
    main()
