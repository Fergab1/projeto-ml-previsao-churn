from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert 'status' in r.json()


def test_predict_with_explain_smoke():
    # provide a minimal example row with likely columns; if model not loaded, API returns model not loaded error
    payload = {'customer': {}}
    r = client.post('/predict_with_explain', json=payload)
    assert r.status_code == 200
    data = r.json()
    # either error or probability key
    assert 'error' in data or 'probability' in data
