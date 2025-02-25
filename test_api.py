import pytest
from fastapi.testclient import TestClient
from app_model import app
import pandas as pd
import os

client = TestClient(app)

# Configuración de rutas de prueba
TEST_MODEL_PATH = "test_model.pkl"
TEST_DATA_PATH = "test_data.csv"

@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Setup: Crear archivos temporales
    pd.DataFrame(columns=["tv", "radio", "newspaper", "sales"]).to_csv(TEST_DATA_PATH, index=False)
    
    # Ejecutar el test
    yield
    
    # Teardown: Eliminar archivos después de los tests
    if os.path.exists(TEST_MODEL_PATH):
        os.remove(TEST_MODEL_PATH)
    if os.path.exists(TEST_DATA_PATH):
        os.remove(TEST_DATA_PATH)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "API operativa" in response.json()["message"]

def test_predict_endpoint():
    test_data = {"tv": 230.1, "radio": 37.8, "newspaper": 69.2}
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)

def test_ingest_endpoint():
    test_data = [{"tv": 150.5, "radio": 25.3, "newspaper": 45.1, "sales": 18.7}]
    response = client.post("/ingest", json=test_data)
    assert response.status_code == 200
    assert "1 registros añadidos" in response.json()["message"]

def test_retrain_endpoint():
    # Ingestar datos primero
    test_ingest_endpoint()
    
    response = client.post("/retrain")
    assert response.status_code == 200
    assert response.json()["r2_score"] >= 0  # R² debe ser un valor válido

def test_invalid_data():
    response = client.post("/predict", json={"tv": "invalid", "radio": 25, "newspaper": 30})
    assert response.status_code == 422  # Error de validaciónpython