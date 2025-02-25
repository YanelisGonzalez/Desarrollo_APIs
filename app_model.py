from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import os
from typing import List
import csv

app = FastAPI(title='Marketing Sales API')

@app.get("/")
async def home():
    return {
        "message": "API de predicción de ventas operativa",
        "endpoints": {
            "documentación": "/docs",
            "predicción": "/predict (POST)",
            "ingesta": "/ingest (POST)",
            "reentrenamiento": "/retrain (POST)"
        }
    }

#Configuración inicial
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,'advertising_model.pkl')
DATA_PATH = os.path.join(BASE_DIR,'Advertising.csv')


#Cargar modelo inicial 
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    model = LinearRegression()
    X_dummy = pd.DataFrame([[100, 50, 30]], columns=['tv', 'radio', 'newspaper'])
    y_dummy = pd.Series([20])
    model.fit(X_dummy, y_dummy)
    
#Esquemas
class MarketingData(BaseModel):
    tv: float
    radio: float
    newspaper: float
    sales: float

class PredictionInput(BaseModel):
    tv: float
    radio: float
    newspaper: float

class RetrainRequest(BaseModel):
    test_size: float = 0.2
    random_state: int = 42


# 1. Endpoint de predicción
@app.post('/predict')
async def predict(input_data: PredictionInput):
    try:
        features = [[input_data.tv, input_data.radio, input_data.newspaper]]
        prediction = model.predict(features)
        return{'preediction': round(float(prediction[0]), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. Endpoint de ingesta de datos
@app.post('/ingest')
async def ingest_data(new_data: List[MarketingData]):
    try:
        file_exists = os.path.isfile(DATA_PATH)
        with open(DATA_PATH, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['tv', 'radio', 'newspaper', 'sales'])
            if not file_exists:
                writer.writeheader()

            for data in new_data:
                writer.writerow(data.dict())

        return {'message': f'{len(new_data)} registros añadidos correctamente'}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# 3. Endpoint de reentramiento del modelo
@app.post('/retrain')
async def retrain_model(request: RetrainRequest):
    try:
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=400, detail='No hay datos para entrenamiento')
        
        df = pd.read_csv(DATA_PATH)
        if len(df) < 10:
            raise HTTPException(status_code=400, detail='Insuficientes datos para entrenamiento')
        
        X = df[['tv', 'radio', 'newspaper']]
        y = df['sales']

        new_model = LinearRegression()
        new_model.fit(X,y)

        #Guardar con pickle
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(new_model, f)

        global model
        model = new_model

        return { 
            'message': 'Modelo reentrenado exitosamente',
            'r2_score': round(new_model.score(X,y), 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)


#Pasos para pobar la API, desde el navegador http://localhost:8000/ y 
# http://127.0.0.1:8000/