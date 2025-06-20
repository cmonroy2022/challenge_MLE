import fastapi
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from .model import DelayModel

app = fastapi.FastAPI()

# Inicializar el modelo globalmente
model = DelayModel()

# Cargar y entrenar el modelo al iniciar la aplicación
def load_model():
    """Carga y entrena el modelo con los datos disponibles"""
    try:
        # Cargar datos de entrenamiento
        data = pd.read_csv("data/data.csv")
        
        # Preprocesar datos
        features, target = model.preprocess(data, target_column="delay")
        
        # Entrenar modelo
        model.fit(features, target)
        
        print("Modelo cargado y entrenado exitosamente")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        raise

# Cargar modelo al iniciar
load_model()

class FlightData(BaseModel):
    """Modelo para validar datos de vuelo individual"""
    OPERA: str
    TIPOVUELO: str
    MES: int

class PredictRequest(BaseModel):
    """Modelo para validar request de predicción"""
    flights: List[FlightData]

class PredictResponse(BaseModel):
    """Modelo para response de predicción"""
    predict: List[int]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    """Endpoint de health check"""
    return {
        "status": "OK"
    }

def validate_flight_data(flight: FlightData) -> bool:
    """
    Valida que los datos del vuelo sean válidos según las reglas del modelo
    
    Args:
        flight: Datos del vuelo a validar
        
    Returns:
        bool: True si los datos son válidos, False en caso contrario
    """
    # Validar OPERA (aerolíneas válidas)
    valid_airlines = [
        "Aerolineas Argentinas", "Aeromexico", "Air Canada", "Air France", 
        "Alitalia", "American Airlines", "Austral", "Avianca", "British Airways",
        "Copa Air", "Delta Air", "Gol Trans", "Grupo LATAM", "Iberia", 
        "JetSmart SPA", "K.L.M.", "Lacsa", "Latin American Wings", "Oceanair Linhas Aereas",
        "Plus Ultra Lineas Aereas", "Qantas Airways", "Sky Airline", "United Airlines"
    ]
    
    if flight.OPERA not in valid_airlines:
        return False
    
    # Validar TIPOVUELO (tipos válidos)
    valid_types = ["I", "N"]  # Internacional, Nacional
    if flight.TIPOVUELO not in valid_types:
        return False
    
    # Validar MES (meses válidos: 1-12)
    if flight.MES < 1 or flight.MES > 12:
        return False
    
    return True

@app.post("/predict", status_code=200, response_model=PredictResponse)
async def post_predict(request: PredictRequest) -> PredictResponse:
    """
    Endpoint para predecir retrasos de vuelos
    
    Args:
        request: Request con lista de vuelos a predecir
        
    Returns:
        PredictResponse: Lista de predicciones (0: sin retraso, 1: con retraso)
    """
    try:
        # Validar cada vuelo
        for flight in request.flights:
            if not validate_flight_data(flight):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Datos de vuelo inválidos: {flight.dict()}"
                )
        
        # Convertir a DataFrame
        flights_data = [flight.dict() for flight in request.flights]
        df = pd.DataFrame(flights_data)
        
        # Agregar columnas requeridas por el modelo (con valores por defecto)
        # Estas columnas se generan en el preprocesamiento pero necesitamos valores iniciales
        df['Fecha-I'] = '2023-01-01 10:00:00'  # Fecha por defecto
        df['Fecha-O'] = '2023-01-01 10:00:00'  # Fecha por defecto
        df['SIGLADES'] = 'SCL'  # Destino por defecto
        df['DIANOM'] = 'Lunes'  # Día por defecto
        
        # Preprocesar datos
        features = model.preprocess(df)
        features = pd.DataFrame(features)  # Asegurar que sea DataFrame
        
        # Hacer predicciones
        predictions = model.predict(features)
        
        return PredictResponse(predict=predictions)
        
    except HTTPException:
        # Re-lanzar HTTPExceptions
        raise
    except Exception as e:
        # Manejar otros errores
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno del servidor: {str(e)}"
        )