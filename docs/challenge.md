# Challenge MLE - Documentación

## Parte I: Implementación del Modelo

### Análisis del Notebook de Exploración

El notebook `exploration.ipynb` presenta un análisis completo de datos de vuelos para predecir retrasos. A continuación se detalla el proceso y las decisiones tomadas:

#### 1. Análisis de Datos

**Datos utilizados:**
- Dataset de vuelos con información sobre aerolíneas, fechas, destinos, tipos de vuelo
- Variable objetivo: `delay` (1 si hay retraso > 15 minutos, 0 en caso contrario)

**Features generadas:**
- `period_day`: Período del día (mañana, tarde, noche)
- `high_season`: Temporada alta (1) o baja (0)
- `min_diff`: Diferencia en minutos entre llegada y salida programada

#### 2. Modelos Evaluados

Se probaron 6 configuraciones diferentes de modelos:

1. **XGBoost básico** (todas las features)
2. **Logistic Regression básico** (todas las features)
3. **XGBoost con top 10 features y balance de clases**
4. **XGBoost con top 10 features sin balance de clases**
5. **Logistic Regression con top 10 features y balance de clases**
6. **Logistic Regression con top 10 features sin balance de clases**

#### 3. Selección del Mejor Modelo

**Criterios de selección:**
- Recall de clase "1" (retrasos) > 0.60
- F1-score de clase "1" > 0.30
- Recall de clase "0" (sin retrasos) < 0.60
- F1-score de clase "0" < 0.70

**Modelo seleccionado:** XGBoost con las siguientes características:
- **Top 10 features más importantes:**
  - OPERA_Latin American Wings
  - MES_7
  - MES_10
  - OPERA_Grupo LATAM
  - MES_12
  - TIPOVUELO_I
  - MES_4
  - MES_11
  - OPERA_Sky Airline
  - OPERA_Copa Air

- **Balance de clases:** `scale_pos_weight = n_y0/n_y1`
- **Learning rate:** 0.01
- **Random state:** 1

**Justificación:**
- El balance de clases mejora significativamente el recall de la clase minoritaria (retrasos)
- La reducción a top 10 features no disminuye el rendimiento
- XGBoost y Logistic Regression tienen rendimientos similares, pero XGBoost es más robusto para este tipo de datos

#### 4. Implementación en model.py

**Características de la implementación:**
- Preprocesamiento automático de datos con generación de features
- Codificación one-hot para variables categóricas
- Selección automática de las top 10 features
- Balance de clases integrado en el modelo
- Métodos `preprocess`, `fit` y `predict` implementados según especificaciones

**Buenas prácticas aplicadas:**
- Tipado estático con type hints
- Documentación completa de métodos
- Manejo de errores y validaciones
- Código modular y reutilizable
- Nombres de variables descriptivos

### Estructura del Modelo

```python
class DelayModel:
    def __init__(self):
        # Inicialización del modelo XGBoost con parámetros optimizados
    
    def preprocess(self, data, target_column=None):
        # Generación de features y preparación de datos
    
    def fit(self, features, target):
        # Entrenamiento del modelo con balance de clases
    
    def predict(self, features):
        # Predicción de retrasos para nuevos vuelos
```

### Resultados de Testing

**Tests ejecutados:** `make model-test`

**Resultados:**
- ✅ **Todos los tests pasan** (4/4)
- 📊 **Cobertura del código: 95%**
- ⚠️ **4 warnings** (relacionados con tipos de datos mixtos en CSV)

**Detalle de cobertura:**
- `challenge\__init__.py`: 100% (2/2 statements)
- `challenge\api.py`: 75% (6/8 statements) - No cubierto por tests del modelo
- `challenge\model.py`: 97% (70/72 statements) - Excelente cobertura

**Tests que pasan:**
1. `test_model_preprocess_for_training` - Preprocesamiento para entrenamiento
2. `test_model_preprocess_for_serving` - Preprocesamiento para predicción
3. `test_model_fit` - Entrenamiento del modelo
4. `test_model_predict` - Predicción con modelo entrenado

### Configuración para Windows

**Problemas resueltos:**
1. **Makefile adaptado** para Windows con comandos compatibles
2. **Rutas de archivos corregidas** en tests
3. **Dependencias instaladas** (pytest, pytest-cov)
4. **Comandos de PowerShell** adaptados

**Comandos utilizados:**
```bash
# Instalar dependencias
pip install pytest pytest-cov

# Ejecutar tests
make model-test

# O directamente
python -m pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/model
```

### Resultados Esperados

El modelo implementado debe cumplir con los siguientes criterios de rendimiento:
- Recall clase "0" < 0.60
- F1-score clase "0" < 0.70  
- Recall clase "1" > 0.60
- F1-score clase "1" > 0.30

Estos criterios aseguran que el modelo tenga un buen balance entre precisión y recall, especialmente para la detección de retrasos (clase minoritaria).

### Archivos Generados

**Reportes de cobertura:**
- `reports/html/` - Reporte HTML detallado
- `reports/coverage.xml` - Reporte XML para CI/CD
- `reports/junit.xml` - Reporte de tests para CI/CD

**Estado del proyecto:**
- ✅ Modelo implementado y funcionando
- ✅ Tests pasando al 100%
- ✅ Cobertura de código excelente (95%)
- ✅ Documentación completa
- ✅ Configuración para Windows lista
