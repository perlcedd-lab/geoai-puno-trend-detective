"""
Configuración global del sistema de Monitoreo y Predicción Espacial de Sequías Agrícolas - Puno
Versión corregida según correccion1.txt (VHI, Regresión Cuantil 90%, Random Forest, XGBoost)
"""
import os

# ============================================
# CONFIGURACIÓN REGIONAL - PUNO
# ============================================
REGION_CONFIG = {
    'name': 'Puno, Perú',
    'bounds': {
        'min_lon': -71.5,
        'max_lon': -68.5,
        'min_lat': -17.0,
        'max_lat': -13.5
    },
    'center': [-70.02, -15.84],
    'zoom': 8,
    # Polígono aproximado de exclusión del Lago Titicaca (cuerpo de agua)
    'titicaca_mask': {
        'min_lon': -70.1,
        'max_lon': -68.6,
        'min_lat': -16.6,
        'max_lat': -15.2
    }
}

# ============================================
# CONFIGURACIÓN DE DATOS Y VARIABLES
# ============================================
DATA_CONFIG = {
    'spatial_resolution': 5000,  # 5 km
    'target_variable': 'vhi',    # Vegetation Health Index (0 - 100%)
    'feature_variables': [
        'vci',          # Vegetation Condition Index (0 - 100%)
        'tci',          # Thermal Condition Index (0 - 100%)
        'lst',          # Land Surface Temperature (°C)
        'sm',           # Soil Moisture (m³/m³)
        'spi_1',        # Antecedent SPI 1-month
        'spi_3',        # Antecedent SPI 3-month
        'dem',          # Digital Elevation Model (m.s.n.m.)
        'lon',          # Longitud
        'lat'           # Latitud
    ],
    'raw_data_path': 'app/data/datasets/',
    'processed_data_path': 'app/data/processed/',
    'results_path': 'app/data/results/',
    'models_path': 'app/models/checkpoints/',
    'ml_models_path': 'app/models/ml_checkpoints/'
}

# ============================================
# CLASIFICACIÓN DE SEQUÍA AGRÍCOLA (FAO / NOAA)
# ============================================
DROUGHT_CATEGORIES = {
    'bins': [-float('inf'), 10.0, 20.0, 30.0, 40.0, float('inf')],
    'labels': [
        'Sequía extrema',
        'Sequía severa',
        'Sequía moderada',
        'Sequía leve',
        'Normal / Óptima'
    ],
    'colors': [
        '#d73027',  # Rojo
        '#fc8d59',  # Naranja
        '#fee08b',  # Amarillo
        '#d9ef8b',  # Verde claro
        '#1a9850'   # Verde oscuro
    ]
}

# ============================================
# CONFIGURACIÓN DE MODELOS
# ============================================
MODEL_CONFIG = {
    # Red Neuronal Tabular Cuantil (Quantile Regression al 90% de confianza)
    'quantile_nn': {
        'hidden_units': [64, 32, 16],
        'dropouts': [0.25, 0.20],
        'activation': 'swish',
        'quantiles': [0.05, 0.50, 0.95],  # [límite inferior, mediana, límite superior]
        'learning_rate': 1e-3,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 15,
        'model_name': 'quantile_nn_model.keras',
        'scaler_name': 'quantile_scaler.pkl'
    },
    # Random Forest Regressor
    'random_forest': {
        'n_estimators': 250,
        'max_depth': 12,
        'min_samples_split': 5,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
        'model_name': 'random_forest_model.pkl'
    },
    # XGBoost Regressor
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.08,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'model_name': 'xgboost_model.pkl'
    },
    'model_path': 'app/models/checkpoints/',
    'ml_model_path': 'app/models/ml_checkpoints/',
    'scaler_path': 'app/models/checkpoints/scalers.pkl'
}

# ============================================
# GEE (Google Earth Engine)
# ============================================
GEE_CONFIG = {
    'service_account': 'earth-engine-public-apps@appspot.gserviceaccount.com',
    'key_file': 'service-account.json',
    'chirps': 'UCSB-CHG/CHIRPS/DAILY',
    'modis_ndvi': 'MODIS/061/MOD13Q1',
    'modis_lst': 'MODIS/061/MOD11A2',
    'smap': 'NASA/SMAP/SPL4SMGP/007',
    'era5_land': 'ECMWF/ERA5_LAND/DAILY_RAW',
    'srtm': 'USGS/SRTMGL1_003',
    'jrc_water': 'JRC/GSW1_4/GlobalSurfaceWater',
    'default_time_window': 90
}

# ============================================
# VISUALIZACIÓN EN FRONTEND
# ============================================
VIS_CONFIG = {
    'vhi': {
        'min': 0,
        'max': 100,
        'palette': ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#1a9850']
    },
    'uncertainty': {
        'min': 0,
        'max': 35,
        'palette': ['#1a9850', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027']
    }
}

# ============================================
# FLASK SERVER
# ============================================
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'static_folder': 'static',
    'template_folder': 'templates'
}

def get_puno_bounds():
    """Retorna [min_lon, min_lat, max_lon, max_lat] de Puno"""
    b = REGION_CONFIG['bounds']
    return [b['min_lon'], b['min_lat'], b['max_lon'], b['max_lat']]

def ensure_directories():
    """Crea todas las carpetas necesarias si no existen"""
    paths = [
        DATA_CONFIG['raw_data_path'],
        DATA_CONFIG['processed_data_path'],
        DATA_CONFIG['results_path'],
        DATA_CONFIG['models_path'],
        DATA_CONFIG['ml_models_path'],
        os.path.join(DATA_CONFIG['results_path'], 'ml_models'),
        os.path.join(DATA_CONFIG['results_path'], 'model_comparison'),
        os.path.join(DATA_CONFIG['results_path'], 'pipeline_logs')
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
    return True

if __name__ == '__main__':
    ensure_directories()
    print("✅ Estructura de directorios y configuración verificadas.")
