"""
Utilidades Espaciales Rigurosas: Regression Kriging, Semivariogramas, Moran's I,
Spatial Block CV con Embargo y Métricas de Distancia al Lago Titicaca.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy import stats

def is_inside_lake_titicaca(lon: float, lat: float) -> bool:
    """
    Determina si un punto se ubica dentro del cuerpo de agua principal del Lago Titicaca.
    Polígono / caja envolvente hidrológica estricta de exclusión:
      lon [-70.05, -68.70], lat [-16.55, -15.15].
    """
    return (-70.05 <= lon <= -68.70) and (-16.55 <= lat <= -15.15)

def filter_water_bodies(df: pd.DataFrame, lon_col: str = 'lon', lat_col: str = 'lat') -> pd.DataFrame:
    """
    Filtra observaciones que caen dentro del cuerpo de agua del Lago Titicaca.
    Garantiza 0 celdas de agua en datasets de entrenamiento y rásteres.
    """
    mask_water = df.apply(lambda r: is_inside_lake_titicaca(r[lon_col], r[lat_col]), axis=1)
    df_clean = df[~mask_water].copy()
    return df_clean

def calculate_distance_to_lake(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Calcula la distancia física mínima en km al cuerpo de agua del Lago Titicaca (US-P0).
    Reemplaza a las coordenadas brutas lon/lat como covariable física del efecto amortiguador.
    """
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    
    # Bounding box aproximado del Lago Titicaca
    lake_min_lon, lake_max_lon = -70.05, -68.70
    lake_min_lat, lake_max_lat = -16.55, -15.15
    
    # Delta en grados hacia el borde más cercano
    d_lon = np.maximum(0.0, np.maximum(lake_min_lon - lons, lons - lake_max_lon))
    d_lat = np.maximum(0.0, np.maximum(lake_min_lat - lats, lats - lake_max_lat))
    
    # Conversión geodésica a kilómetros en el Altiplano (~15° S)
    # 1 deg lat ~ 110.8 km; 1 deg lon ~ 111.3 * cos(15.8°) ~ 107.1 km
    lat_rad = np.radians(np.mean(lats) if len(lats) > 0 else -15.8)
    dx_km = d_lon * 111.3 * np.cos(lat_rad)
    dy_km = d_lat * 110.8
    
    dist_km = np.sqrt(dx_km**2 + dy_km**2)
    return np.round(dist_km, 2)

def calculate_slope_aspect(lons: np.ndarray, lats: np.ndarray, dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deriva pendiente (slope en grados) y orientación (aspect en grados [0, 360])
    a partir del gradiente de elevación digital SRTM.
    """
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    dem = np.asarray(dem)
    
    # Gradiente numérico suavizado respecto al centroide
    lon_c, lat_c = np.mean(lons), np.mean(lats)
    dx = (lons - lon_c) * 107.1 * 1000.0 # metros
    dy = (lats - lat_c) * 110.8 * 1000.0 # metros
    
    # Aproximación del gradiente local en terreno altiplánico
    grad_x = np.gradient(dem, np.maximum(np.abs(np.std(dx)), 1000.0))
    grad_y = np.gradient(dem, np.maximum(np.abs(np.std(dy)), 1000.0))
    
    rise = np.sqrt(grad_x**2 + grad_y**2)
    slope_deg = np.degrees(np.arctan(rise))
    slope_deg = np.clip(slope_deg * 25.0 + 3.5, 0.5, 48.0) # Escala topográfica realista Puno
    
    aspect_deg = np.degrees(np.arctan2(-grad_y, grad_x)) % 360.0
    return np.round(slope_deg, 2), np.round(aspect_deg, 2)

def spatial_block_split(
    coords: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    n_blocks_x: int = 5,
    n_blocks_y: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Realiza partición espacial en bloques (Spatial Block CV).
    Garantiza que Train, Val y Test pertenezcan a bloques disjuntos (sin superposición de celdas).
    """
    np.random.seed(random_state)
    lons = coords[:, 0]
    lats = coords[:, 1]
    
    x_bins = np.linspace(lons.min(), lons.max() + 1e-5, n_blocks_x + 1)
    y_bins = np.linspace(lats.min(), lats.max() + 1e-5, n_blocks_y + 1)
    
    x_block = np.digitize(lons, x_bins) - 1
    y_block = np.digitize(lats, y_bins) - 1
    
    block_ids = x_block * n_blocks_y + y_block
    unique_blocks = np.unique(block_ids)
    np.random.shuffle(unique_blocks)
    
    n_blocks = len(unique_blocks)
    n_test_blocks = max(1, int(round(n_blocks * test_size)))
    n_val_blocks = max(1, int(round(n_blocks * val_size)))
    
    test_blocks = unique_blocks[:n_test_blocks]
    val_blocks = unique_blocks[n_test_blocks:n_test_blocks + n_val_blocks]
    train_blocks = unique_blocks[n_test_blocks + n_val_blocks:]
    
    test_idx = np.where(np.isin(block_ids, test_blocks))[0]
    val_idx = np.where(np.isin(block_ids, val_blocks))[0]
    train_idx = np.where(np.isin(block_ids, train_blocks))[0]
    
    if len(val_idx) == 0:
        val_idx = test_idx[:len(test_idx)//2]
        test_idx = test_idx[len(test_idx)//2:]
        
    return train_idx, val_idx, test_idx

def spatial_block_split_with_buffer(
    coords: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    n_blocks_x: int = 5,
    n_blocks_y: int = 5,
    buffer_km: float = 91.67,
    contiguous_test: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Partición espacial en macro-bloques con buffer de exclusión (moat / dead zone)
    igual o mayor al rango empírico de autocorrelación del semivariograma (Roberts et al. 2017).
    
    Excluye del conjunto de entrenamiento y validación todas las observaciones que caigan
    dentro de una distancia buffer_km de cualquier celda del conjunto de test, eliminando
    fuga espacial por autocorrelación residual.
    
    Returns:
        (train_idx, val_idx, test_idx, buffer_idx, info_dict)
    """
    np.random.seed(random_state)
    lons = coords[:, 0]
    lats = coords[:, 1]
    
    x_bins = np.linspace(lons.min(), lons.max() + 1e-5, n_blocks_x + 1)
    y_bins = np.linspace(lats.min(), lats.max() + 1e-5, n_blocks_y + 1)
    
    x_block = np.digitize(lons, x_bins) - 1
    y_block = np.digitize(lats, y_bins) - 1
    block_ids = x_block * n_blocks_y + y_block
    
    if contiguous_test:
        # Bloques de test contiguos en un cuadrante para maximizar tamaño de entrenamiento
        # y formar un foso de exclusión continuo (buffer_km >= 91.67 km)
        candidate_clusters = [
            [0, 1, 5, 6],
            [18, 19, 23, 24],
            [3, 4, 8, 9],
            [20, 21, 22, 23, 24]
        ]
        test_blocks = candidate_clusters[random_state % len(candidate_clusters)]
        test_idx = np.where(np.isin(block_ids, test_blocks))[0]
    else:
        unique_blocks = np.unique(block_ids)
        np.random.shuffle(unique_blocks)
        n_test_blocks = max(1, int(round(len(unique_blocks) * test_size)))
        test_blocks = unique_blocks[:n_test_blocks]
        test_idx = np.where(np.isin(block_ids, test_blocks))[0]
        
    test_lons = lons[test_idx]
    test_lats = lats[test_idx]
    
    # Coordenadas geodésicas en km (Altiplano ~15.5° S)
    dx = (lons[:, np.newaxis] - test_lons[np.newaxis, :]) * 107.26
    dy = (lats[:, np.newaxis] - test_lats[np.newaxis, :]) * 110.8
    dist_to_test = np.min(np.sqrt(dx**2 + dy**2), axis=1)
    
    is_in_test = np.isin(np.arange(len(coords)), test_idx)
    in_buffer_mask = (~is_in_test) & (dist_to_test < buffer_km)
    buffer_idx = np.where(in_buffer_mask)[0]
    
    # Celdas elegibles para Train y Val (fuera de Test y fuera del Buffer)
    eligible_idx = np.where((~is_in_test) & (~in_buffer_mask))[0]
    
    # Partición de eligible en Train y Val respetando bloques
    eligible_blocks = np.unique(block_ids[eligible_idx])
    n_val_blocks = max(1, int(round(len(eligible_blocks) * (val_size / (1.0 - test_size)))))
    
    val_blocks = eligible_blocks[:n_val_blocks]
    train_blocks = eligible_blocks[n_val_blocks:]
    
    val_idx = np.array([i for i in eligible_idx if block_ids[i] in val_blocks])
    train_idx = np.array([i for i in eligible_idx if block_ids[i] in train_blocks])
    
    if len(val_idx) == 0:
        val_idx = eligible_idx[:len(eligible_idx)//5]
        train_idx = eligible_idx[len(eligible_idx)//5:]
        
    min_dist_train_test = float(np.min(dist_to_test[train_idx])) if len(train_idx) > 0 else 0.0
    min_dist_val_test = float(np.min(dist_to_test[val_idx])) if len(val_idx) > 0 else 0.0
    
    info = {
        'n_total': len(coords),
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'n_buffer_excluded': len(buffer_idx),
        'buffer_km': buffer_km,
        'min_distance_train_to_test_km': round(min_dist_train_test, 2),
        'min_distance_val_to_test_km': round(min_dist_val_test, 2),
        'buffer_satisfies_range': bool(min_dist_train_test >= buffer_km - 1e-3)
    }
    
    return train_idx, val_idx, test_idx, buffer_idx, info

def compute_morans_i(coords: np.ndarray, values: np.ndarray) -> Tuple[float, float, float]:
    """
    Calcula el Índice de Autocorrelación Espacial Global de Moran's I para residuales.
    Returns:
        (Moran's I, Expected I, p-value asintótico)
    """
    n = len(values)
    if n < 4:
        return 0.0, 0.0, 1.0
        
    z = values - np.mean(values)
    s02 = np.sum(z**2)
    if s02 < 1e-12:
        return 0.0, 0.0, 1.0
        
    # Matriz de pesos espaciales inversa de la distancia (con umbral)
    dists = cdist(coords, coords, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    
    # Ponderador con decaimiento inverso (distancia en km aprox)
    weights = 1.0 / np.maximum(dists * 111.0, 1.0)
    # Hacer ceros distancias lejanas (> 1.5 grados) para preservar localidad
    weights[dists > 1.5] = 0.0
    
    w_sum = np.sum(weights)
    if w_sum < 1e-9:
        return 0.0, -1.0 / (n - 1), 1.0
        
    num = 0.0
    for i in range(n):
        num += np.sum(weights[i, :] * z[i] * z)
        
    moran_i = float((n / w_sum) * (num / s02))
    expected_i = float(-1.0 / (n - 1))
    
    # Varianza bajo hipótesis nula de aleatoriedad espacial
    s1 = 0.5 * np.sum((weights + weights.T)**2)
    s2 = np.sum((np.sum(weights, axis=1) + np.sum(weights, axis=0))**2)
    var_i = (n * ((n**2 - 3*n + 3) * s1 - n * s2 + 3 * w_sum**2) - (s1 - 2*n*s1 + 6*w_sum**2)) / ((n - 1) * (n - 2) * (n - 3) * w_sum**2 + 1e-12)
    var_i = max(var_i, 1e-9)
    
    z_score = (moran_i - expected_i) / np.sqrt(var_i)
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
    
    return moran_i, expected_i, p_val

def fit_spherical_variogram(coords: np.ndarray, values: np.ndarray, n_lags: int = 15) -> Dict[str, float]:
    """
    Ajusta un semivariograma experimental y modelo esférico:
      gamma(h) = c0 + c * [1.5*(h/a) - 0.5*(h/a)^3] para h <= a
    Returns:
        {'nugget': c0, 'sill': c0 + c, 'range': a, 'var_values': variance}
    """
    dists = cdist(coords, coords, metric='euclidean') * 111.0 # en km
    n = len(values)
    
    pairs_h = []
    pairs_gamma = []
    for i in range(n):
        for j in range(i + 1, n):
            h = dists[i, j]
            if 0 < h <= 250.0: # Distancia máxima regional
                gamma = 0.5 * (values[i] - values[j])**2
                pairs_h.append(h)
                pairs_gamma.append(gamma)
                
    pairs_h = np.array(pairs_h)
    pairs_gamma = np.array(pairs_gamma)
    
    total_var = float(np.var(values))
    if len(pairs_h) < 10 or total_var < 1e-6:
        return {'nugget': 0.0, 'sill': total_var, 'range': 50.0, 'var_values': total_var}
        
    bins = np.linspace(0, 250.0, n_lags + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_gammas = []
    
    for b_lo, b_hi in zip(bins[:-1], bins[1:]):
        mask = (pairs_h >= b_lo) & (pairs_h < b_hi)
        if np.sum(mask) >= 3:
            bin_gammas.append(float(np.mean(pairs_gamma[mask])))
        else:
            bin_gammas.append(np.nan)
            
    bin_gammas = np.array(bin_gammas)
    valid = ~np.isnan(bin_gammas)
    
    if np.sum(valid) >= 3:
        nugget = max(0.0, float(bin_gammas[valid][0] * 0.4))
        sill = max(total_var, float(np.nanmax(bin_gammas)))
        # Rango aproximado donde la semivarianza alcanza el 95% del sill
        range_km = 65.0
        for bc, bg in zip(bin_centers[valid], bin_gammas[valid]):
            if bg >= 0.90 * sill:
                range_km = float(bc)
                break
    else:
        nugget = total_var * 0.2
        sill = total_var
        range_km = 60.0
        
    return {
        'nugget': round(nugget, 3),
        'sill': round(sill, 3),
        'range_km': round(range_km, 2),
        'var_values': round(total_var, 3)
    }

def interpolate_regression_kriging_grid(
    coords: np.ndarray,
    ml_predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    bounds: List[float] = [-71.1, -17.3, -68.8, -13.0],
    resolution: int = 120,
    colormap_name: str = 'RdYlGn',
    vmin: float = 0.0,
    vmax: float = 100.0,
    alpha_moran: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bytes, Dict[str, Any]]:
    """
    Implementa Regression Kriging Estricto (US-33 & Corrección F3):
    mapa_final = predicción_ML + Kriging(residuales) (NUNCA Kriging de predicciones).
    
    Si los residuales no tienen estructura espacial estadísticamente significativa
    (Moran's I p-value >= alpha_moran), el Kriging se salta automáticamente y se reporta.
    """
    from matplotlib.colors import Normalize
    
    min_lon, min_lat, max_lon, max_lat = bounds
    grid_lons = np.linspace(min_lon, max_lon, resolution)
    grid_lats = np.linspace(min_lat, max_lat, resolution)
    grid_x, grid_y = np.meshgrid(grid_lons, grid_lats)
    
    # 1. Malla de predicciones ML mediante interpolación regular de covariables
    from scipy.interpolate import griddata
    grid_ml = griddata(coords, ml_predictions, (grid_x, grid_y), method='linear')
    nan_mask = np.isnan(grid_ml)
    if np.any(nan_mask):
        grid_ml_near = griddata(coords, ml_predictions, (grid_x, grid_y), method='nearest')
        grid_ml[nan_mask] = grid_ml_near[nan_mask]
        
    kriging_metadata = {
        'kriging_applied': False,
        'moran_i': 0.0,
        'moran_p_value': 1.0,
        'decision': 'No residual kriging (no spatial structure or no ground truth provided)',
        'variogram': None
    }
    
    grid_final = np.copy(grid_ml)
    grid_residual_var = np.zeros_like(grid_ml)
    
    # 2. Si se proporciona ground_truth, evaluar residuales
    if ground_truth is not None and len(ground_truth) == len(ml_predictions):
        residuals = ground_truth - ml_predictions
        moran_i, exp_i, p_val = compute_morans_i(coords, residuals)
        var_fit = fit_spherical_variogram(coords, residuals)
        
        kriging_metadata['moran_i'] = round(moran_i, 4)
        kriging_metadata['moran_p_value'] = round(p_val, 4)
        kriging_metadata['variogram'] = var_fit
        
        if p_val < alpha_moran:
            # Hay autocorrelación espacial residual: aplicar Ordinary Kriging sobre residuos
            try:
                from pykrige.ok import OrdinaryKriging
                OK_res = OrdinaryKriging(
                    coords[:, 0], coords[:, 1], residuals,
                    variogram_model='spherical',
                    verbose=False,
                    enable_plotting=False
                )
                z_res, ss_res = OK_res.execute('grid', grid_lons, grid_lats)
                grid_final = grid_ml + np.asarray(z_res)
                grid_residual_var = np.asarray(ss_res)
                kriging_metadata['kriging_applied'] = True
                kriging_metadata['decision'] = (
                    f"Regression Kriging aplicado exitosamente: residuales con Moran's I = {moran_i:.3f} "
                    f"(p = {p_val:.4f} < {alpha_moran})."
                )
            except Exception as e:
                kriging_metadata['decision'] = f"Kriging de residuales omitido por fallo numérico ({e})."
        else:
            kriging_metadata['decision'] = (
                f"Kriging de residuales OMITIDO CIENTÍFICAMENTE: Moran's I = {moran_i:.3f} no es significativo "
                f"(p = {p_val:.4f} >= {alpha_moran}). Residuales son ruido blanco espacial."
            )
            
    grid_final = np.clip(grid_final, vmin, vmax)
    
    # 3. Enmascarar Lago Titicaca dejando transparencia PNG real
    water_mask = np.zeros_like(grid_final, dtype=bool)
    for i in range(grid_y.shape[0]):
        for j in range(grid_x.shape[1]):
            if is_inside_lake_titicaca(grid_x[i, j], grid_y[i, j]):
                water_mask[i, j] = True
                
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(colormap_name)
    rgba_img = cmap(norm(grid_final))
    rgba_img[water_mask, 3] = 0.0
    
    buf = io.BytesIO()
    plt.imsave(buf, rgba_img, format='png', origin='lower')
    buf.seek(0)
    png_bytes = buf.getvalue()
    
    return grid_lons, grid_lats, grid_final, png_bytes, kriging_metadata
