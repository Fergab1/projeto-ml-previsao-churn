from src.preprocess import load_and_clean
import os

def test_load_and_clean_exists():
    path = os.path.join(os.path.dirname(__file__), '..', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    path = os.path.abspath(path)
    # If dataset is not present in tests location, skip
    if not os.path.exists(path):
        return
    df = load_and_clean(path)
    assert 'TotalCharges' in df.columns
    assert df.shape[0] > 0
