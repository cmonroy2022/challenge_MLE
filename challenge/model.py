import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Union, List
import xgboost as xgb

class DelayModel:

    def __init__(
        self
    ):
        self._model = None
        self._top_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    def _get_period_day(self, date: str) -> str:
        """
        Determina el período del día basado en la hora.
        
        Args:
            date (str): Fecha en formato 'YYYY-MM-DD HH:MM:SS'
            
        Returns:
            str: Período del día ('mañana', 'tarde', 'noche')
        """
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()
        
        if date_time > morning_min and date_time < morning_max:
            return 'mañana'
        elif date_time > afternoon_min and date_time < afternoon_max:
            return 'tarde'
        elif ((date_time > evening_min and date_time < evening_max) or
              (date_time > night_min and date_time < night_max)):
            return 'noche'

    def _is_high_season(self, fecha: str) -> int:
        """
        Determina si la fecha está en temporada alta.
        
        Args:
            fecha (str): Fecha en formato 'YYYY-MM-DD HH:MM:SS'
            
        Returns:
            int: 1 si es temporada alta, 0 en caso contrario
        """
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)
        
        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0

    def _get_min_diff(self, data: pd.Series) -> float:
        """
        Calcula la diferencia en minutos entre llegada y salida programada.
        
        Args:
            data (pd.Series): Fila de datos con Fecha-O y Fecha-I
            
        Returns:
            float: Diferencia en minutos
        """
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Crear copia para no modificar el original
        df = data.copy()
        
        # Generar features adicionales
        df['period_day'] = df['Fecha-I'].apply(self._get_period_day)
        df['high_season'] = df['Fecha-I'].apply(self._is_high_season)
        df['min_diff'] = df.apply(self._get_min_diff, axis=1)
        
        # Crear features one-hot
        features = pd.concat([
            pd.get_dummies(df['OPERA'], prefix='OPERA'),
            pd.get_dummies(df['TIPOVUELO'], prefix='TIPOVUELO'), 
            pd.get_dummies(df['MES'], prefix='MES')
        ], axis=1)
        
        # Seleccionar solo las top 10 features
        available_features = [col for col in self._top_features if col in features.columns]
        features = features[available_features]
        
        # Asegurar que todas las features estén presentes (rellenar con 0 si no existen)
        for feature in self._top_features:
            if feature not in features.columns:
                features[feature] = 0
        
        # Reordenar columnas según el orden definido
        features = features[self._top_features]
        
        if target_column is not None:
            # Para entrenamiento
            threshold_in_minutes = 15
            target = pd.DataFrame({
                'delay': np.where(df['min_diff'] > threshold_in_minutes, 1, 0)
            })
            return features, target
        else:
            # Para predicción
            return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Calcular balance de clases
        n_y0 = len(target[target['delay'] == 0])
        n_y1 = len(target[target['delay'] == 1])
        scale = n_y0 / n_y1
        
        # Inicializar y entrenar modelo XGBoost
        self._model = xgb.XGBClassifier(
            random_state=1, 
            learning_rate=0.01, 
            scale_pos_weight=scale
        )
        
        self._model.fit(features, target['delay'])

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self._model.predict(features)
        return predictions.tolist()