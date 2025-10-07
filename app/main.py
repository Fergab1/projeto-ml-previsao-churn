from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd
import csv
import traceback

app = FastAPI()
MODEL_PATH = Path('output') / 'LogisticRegression_class_weight.joblib'
PIPE = None
EXPLAINER = None

class InputRecord(BaseModel):
    customer: dict

@app.on_event('startup')
def load_model():
    global PIPE
    PIPE = None
    if MODEL_PATH.exists():
        PIPE = joblib.load(MODEL_PATH)
    # try to build a SHAP explainer for the pipeline if shap is installed
    global EXPLAINER
    EXPLAINER = None
    try:
        import shap
        if PIPE is not None and hasattr(PIPE.named_steps['pre'], 'transform'):
            # fit explainer on a small subset if possible
            EXPLAINER = None
            try:
                # attempt to build explainer using transformed train data if available
                EXPLAINER = shap.explainers.Linear(PIPE.named_steps['clf'], masker=PIPE.named_steps['pre'])
            except Exception:
                try:
                    EXPLAINER = shap.LinearExplainer(PIPE.named_steps['clf'], PIPE.named_steps['pre'].transform)
                except Exception:
                    EXPLAINER = None
    except Exception:
        EXPLAINER = None

@app.post('/predict')
def predict(rec: InputRecord):
    if PIPE is None:
        return {'error': 'model not loaded'}
    df = pd.DataFrame([rec.customer])
    try:
        probs = PIPE.predict_proba(df)[:,1]
        return {'probability': float(probs[0])}
    except Exception as e:
        return {'error': str(e)}


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': PIPE is not None}


@app.post('/predict_with_explain')
def predict_with_explain(rec: InputRecord):
    if PIPE is None:
        return {'error': 'model not loaded'}
    df = pd.DataFrame([rec.customer])
    try:
        probs = PIPE.predict_proba(df)[:,1]
    except Exception as e:
        return {'error': str(e)}
    out = Path('output')
    topk = None
    explanation = None
    # Prefer live SHAP explanation if available
    try:
        if EXPLAINER is not None:
            import shap
            # need to transform df through preprocessor
            pre = PIPE.named_steps['pre']
            X_t = pre.transform(df)
            # get shap values
            sv = EXPLAINER.shap_values(X_t)
            # convert to list
            explanation = {'shap_values': sv.tolist() if hasattr(sv, 'tolist') else list(sv)}
        else:
            # fallback to previously computed topk CSV if present
            if (out / 'shap_topk.csv').exists():
                tk = pd.read_csv(out / 'shap_topk.csv')
                if not tk.empty:
                    explanation = tk.iloc[0].to_dict()
    except Exception as e:
        explanation = {'error': str(e), 'trace': traceback.format_exc()}
    return {'probability': float(probs[0]), 'explanation': explanation}
