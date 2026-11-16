"""
Métricas de Evaluación Científica y Geoespacial:
- Métricas Puntuales: RMSE, MAE, R² (sklearn), NSE (Nash-Sutcliffe), Pearson r², Bias medio.
- Métricas de Intervalos de Incertidumbre: PICP (cobertura empírica), PINAW (ancho normalizado),
  Winkler Score (Interval Score formal), CWC (Coverage Width-based Criterion).
- Conformalized Quantile Regression (CQR Romano et al. 2019) y Rearreglo Monotónico.
- Bootstrap Espacial en Bloques con Intervalos de Confianza al 95%.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Union, Any

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas estadísticas formales para regresión ambiental.
    
    Nota sobre NSE vs R²:
    Por definición matemática, NSE = 1 - Σ(y - ŷ)² / Σ(y - ȳ)² es IDÉNTICO a r2_score de scikit-learn.
    Ambos difieren del cuadrado del coeficiente de correlación de Pearson (r²_pearson) en presencia de sesgo.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    # Nash-Sutcliffe Efficiency (NSE)
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    ss_res = float(np.sum((y_true - y_pred)**2))
    nse = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0
    
    # Sesgo medio (Bias)
    bias = float(np.mean(y_pred - y_true))
    
    # Coeficiente de correlación de Pearson al cuadrado (r² Pearson)
    if np.std(y_true) > 1e-9 and np.std(y_pred) > 1e-9:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
        pearson_r2 = float(corr**2)
    else:
        corr = 0.0
        pearson_r2 = 0.0
        
    return {
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4),
        'NSE': round(nse, 4),
        'Bias': round(bias, 4),
        'Pearson_R2': round(pearson_r2, 4),
        'Correlation': round(corr, 4)
    }

def calculate_winkler_score(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    alpha: float = 0.10
) -> float:
    """
    Calcula el Winkler Score (Interval Score para nivel 1 - alpha, p.ej. alpha = 0.10 para 90% de confianza):
      S_alpha(l, u; y) = (u - l) + (2/alpha)*(l - y)*I(y < l) + (2/alpha)*(y - u)*I(y > u)
    Penaliza tanto la amplitud del intervalo como la lejanía de las observaciones que escapan del rango.
    Menor valor = intervalo más nítido y mejor calibrado.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    l = np.asarray(q_lower, dtype=np.float64).flatten()
    u = np.asarray(q_upper, dtype=np.float64).flatten()
    
    width = u - l
    penalty_low = (2.0 / alpha) * (l - y_true) * (y_true < l)
    penalty_high = (2.0 / alpha) * (y_true - u) * (y_true > u)
    
    scores = width + penalty_low + penalty_high
    return float(np.mean(scores))

def calculate_cwc(
    picp_pct: float,
    pinaw_pct: float,
    nominal_coverage: float = 90.0,
    eta: float = 50.0
) -> float:
    """
    Coverage Width-based Criterion (CWC):
    CWC = PINAW * (1 + gamma * exp(eta * (nominal - PICP) / 100))
    donde gamma = 1 si PICP < nominal, 0 si PICP >= nominal.
    """
    if picp_pct < nominal_coverage:
        gamma = 1.0
        penalty = np.exp(eta * (nominal_coverage - picp_pct) / 100.0)
    else:
        gamma = 0.0
        penalty = 0.0
    return float(pinaw_pct * (1.0 + gamma * penalty))

def calculate_uncertainty_metrics(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    nominal_coverage: float = 0.90,
    y_range: Optional[float] = None
) -> Dict[str, float]:
    """
    Calcula la suite completa de métricas de calidad para intervalos de incertidumbre:
    - PICP: Prediction Interval Coverage Probability (%)
    - PINAW: Prediction Interval Normalized Average Width (%)
    - ACE: Average Coverage Error (|PICP - nominal|)
    - Winkler_Score: Puntuación de intervalo
    - CWC: Coverage Width-based Criterion
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    q_lower = np.asarray(q_lower, dtype=np.float64).flatten()
    q_upper = np.asarray(q_upper, dtype=np.float64).flatten()
    
    # Monotonicidad forzada previa a métricas
    q_upper = np.maximum(q_upper, q_lower)
    
    inside = (y_true >= q_lower) & (y_true <= q_upper)
    picp = float(np.mean(inside)) * 100.0
    
    if y_range is None:
        y_range = float(np.max(y_true) - np.min(y_true))
    if y_range < 1e-6:
        y_range = 100.0 # Rango completo VHI [0, 100]
        
    width = q_upper - q_lower
    mean_width = float(np.mean(width))
    pinaw = float(mean_width / y_range) * 100.0
    
    nominal_pct = nominal_coverage * 100.0
    ace = float(abs(picp - nominal_pct))
    
    alpha = 1.0 - nominal_coverage
    winkler = calculate_winkler_score(y_true, q_lower, q_upper, alpha=alpha)
    cwc = calculate_cwc(picp, pinaw, nominal_coverage=nominal_pct)
    
    return {
        'PICP_Percent': round(picp, 2),
        'Target_Coverage': round(nominal_pct, 2),
        'Mean_Interval_Width': round(mean_width, 2),
        'PINAW_Percent': round(pinaw, 2),
        'ACE': round(ace, 2),
        'Winkler_Score': round(winkler, 2),
        'CWC': round(cwc, 2)
    }

def apply_conformalized_quantile_regression(
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    y_cal: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    q_mid_test: Optional[np.ndarray] = None,
    alpha: float = 0.10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Conformalized Quantile Regression (CQR, Romano et al. 2019):
    1. Calcula residuos de no-conformidad sobre el set de calibración:
       s_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
    2. Calcula el factor de ajuste conforme q_hat = cuantil ceil((n+1)(1-alpha))/n de {s_i}.
    3. Ajusta intervalos en test: [q_lo - q_hat, q_hi + q_hat].
       ARITMÉTICA EXACTA DEL ANCHO:
       W_crudo = q_hi - q_lo
       W_CQR = (q_hi + q_hat) - (q_lo - q_hat) = W_crudo + 2*q_hat
       Si q_hat < 0 (sobre-cobertura del modelo crudo), el ancho se CONTRAE en 2*|q_hat| unidades de VHI.
       La variación de PINAW es exactamente: ΔPINAW = (2*q_hat / Rango_VHI) * 100%.
       Por ejemplo, con q_hat = -1.046 y Rango_VHI ≈ 70.8, la reducción de PINAW es:
       2 * (-1.046) / 70.8 * 100% = -2.95 puntos porcentuales de PINAW.
    4. Aplica rearreglo monotónico estricto: q_lo <= q_mid <= q_hi.
    
    Returns:
        (q_lo_conf, q_mid_conf, q_hi_conf, q_hat)
    """
    q_lo_cal = np.asarray(q_lo_cal, dtype=np.float64)
    q_hi_cal = np.asarray(q_hi_cal, dtype=np.float64)
    y_cal = np.asarray(y_cal, dtype=np.float64)
    
    n_cal = len(y_cal)
    if n_cal < 3:
        raise ValueError("Conjunto de calibración insuficiente (mínimo 3 observaciones).")
        
    # Scores de no-conformidad
    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    
    # Cuantil empírico con corrección de muestra finita
    # k = ceil((n + 1) * (1 - alpha)) / n
    quantile_level = min(1.0, float(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal))
    q_hat = float(np.quantile(scores, quantile_level, method='higher'))
    
    # Ajuste en test
    q_lo_conf = q_lo_test - q_hat
    q_hi_conf = q_hi_test + q_hat
    
    if q_mid_test is None:
        q_mid_test = 0.5 * (q_lo_conf + q_hi_conf)
    else:
        q_mid_test = np.copy(q_mid_test)
        
    # Rearreglo monotónico y acotado biofísico [0, 100]
    q_lo_conf = np.clip(q_lo_conf, 0.0, 100.0)
    q_hi_conf = np.clip(q_hi_conf, 0.0, 100.0)
    q_mid_test = np.clip(q_mid_test, 0.0, 100.0)
    
    q_mid_conf = np.maximum(q_lo_conf, np.minimum(q_mid_test, q_hi_conf))
    q_hi_conf = np.maximum(q_hi_conf, q_mid_conf)
    
    return q_lo_conf, q_mid_conf, q_hi_conf, q_hat

def estimate_prob_vhi_below_threshold(
    q_lo: np.ndarray,
    q_mid: np.ndarray,
    q_hi: np.ndarray,
    tau_lo: float = 0.05,
    tau_mid: float = 0.50,
    tau_hi: float = 0.95,
    threshold: float = 20.0
) -> np.ndarray:
    """
    Estima la probabilidad P(VHI < threshold) mediante interpolación monotónica continua
    de la Función de Distribución Acumulada Empírica (CDF) a partir de los cuantiles predichos.
    
    Nudos de la CDF para cada celda:
      (VHI=0.0, F=0.0), (q_lo, tau_lo), (q_mid, tau_mid), (q_hi, tau_hi), (VHI=100.0, F=1.0)
    
    Garantiza estricta monotonía y acotamiento en [0.0, 1.0].
    """
    q_lo = np.asarray(q_lo, dtype=np.float64).flatten()
    q_mid = np.asarray(q_mid, dtype=np.float64).flatten()
    q_hi = np.asarray(q_hi, dtype=np.float64).flatten()
    
    n = len(q_lo)
    probs = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        knots_x = np.array([0.0, q_lo[i], q_mid[i], q_hi[i], 100.0])
        knots_y = np.array([0.0, tau_lo, tau_mid, tau_hi, 1.0])
        
        # Corrección de nudos coincidentes para interpolación estable
        for k in range(1, len(knots_x)):
            if knots_x[k] <= knots_x[k-1]:
                knots_x[k] = knots_x[k-1] + 1e-4
                
        p = float(np.interp(threshold, knots_x, knots_y))
        probs[i] = np.clip(p, 0.0, 1.0)
        
    return probs

def calculate_persistence_baseline(
    y_true: np.ndarray,
    vhi_lag: np.ndarray,
    block_ids: Optional[np.ndarray] = None,
    delta_threshold: float = 10.0,
    n_replications: int = 2000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Calcula el baseline de PERSISTENCIA: ŷ = VHI(t) = vhi_lag.
    Evalúa métricas globales con IC95% (Spatial Block Bootstrap) y en el subconjunto crítico
    |ΔVHI| = |y_true - vhi_lag| > delta_threshold (onset o recuperación de sequía).
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    vhi_lag = np.asarray(vhi_lag, dtype=np.float64).flatten()
    
    # 1. Métricas globales de persistencia
    m_global = calculate_metrics(y_true, vhi_lag)
    
    ci_global = None
    if block_ids is not None:
        ci_global = block_bootstrap_confidence_intervals(
            y_true, vhi_lag, block_ids, n_replications=n_replications, random_state=random_state
        )
        
    # 2. Subconjunto de transición / estrés dinámico |ΔVHI| > delta_threshold
    delta = np.abs(y_true - vhi_lag)
    mask_stress = delta > delta_threshold
    n_stress = int(np.sum(mask_stress))
    
    if n_stress > 0:
        m_stress = calculate_metrics(y_true[mask_stress], vhi_lag[mask_stress])
    else:
        m_stress = {'RMSE': 0.0, 'MAE': 0.0, 'R2': 0.0, 'NSE': 0.0}
        
    return {
        'global_metrics': m_global,
        'ci95': ci_global,
        'subset_stress': {
            'threshold': delta_threshold,
            'n_samples': n_stress,
            'pct_samples': round((n_stress / len(y_true)) * 100.0, 2),
            'metrics': m_stress
        }
    }

def calculate_skill_score(rmse_model: float, rmse_persistence: float) -> float:
    """Skill score de RMSE vs persistencia: SS = 1 - (RMSE_modelo / RMSE_persistencia)"""
    if rmse_persistence < 1e-9:
        return 0.0
    return float(1.0 - (rmse_model / rmse_persistence))

def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    block_ids: np.ndarray,
    n_replications: int = 2000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Test pareado por Bootstrap en Bloques Espaciales (Spatial Block Bootstrap).
    Evalúa la significancia estadística de la diferencia ΔRMSE = RMSE_A - RMSE_B.
    Si el IC95% de la diferencia incluye el 0 (p >= 0.05), la diferencia NO es estadísticamente
    significativa y está metodológicamente prohibido etiquetar un modelo como 'SOTA'.
    """
    np.random.seed(random_state)
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred_a = np.asarray(y_pred_a, dtype=np.float64).flatten()
    y_pred_b = np.asarray(y_pred_b, dtype=np.float64).flatten()
    
    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)
    
    delta_rmse_list = []
    
    for _ in range(n_replications):
        boot_blocks = np.random.choice(unique_blocks, size=n_blocks, replace=True)
        boot_idx = []
        for b in boot_blocks:
            boot_idx.extend(np.where(block_ids == b)[0])
        boot_idx = np.array(boot_idx)
        
        yt = y_true[boot_idx]
        ya = y_pred_a[boot_idx]
        yb = y_pred_b[boot_idx]
        
        if len(np.unique(yt)) < 2:
            continue
            
        rmse_a = np.sqrt(mean_squared_error(yt, ya))
        rmse_b = np.sqrt(mean_squared_error(yt, yb))
        delta_rmse_list.append(rmse_a - rmse_b)
        
    delta_arr = np.array(delta_rmse_list)
    ci_lo = float(np.percentile(delta_arr, 2.5))
    ci_hi = float(np.percentile(delta_arr, 97.5))
    median_diff = float(np.median(delta_arr))
    
    # p-valor empírico bilateral
    p_val = float(2.0 * min(np.mean(delta_arr <= 0), np.mean(delta_arr >= 0)))
    p_val = min(1.0, max(0.0, p_val))
    
    is_significant = not (ci_lo <= 0.0 <= ci_hi)
    
    return {
        'delta_rmse_ci95': [round(ci_lo, 4), round(ci_hi, 4)],
        'delta_rmse_median': round(median_diff, 4),
        'p_value': round(p_val, 4),
        'is_significant': is_significant,
        'conclusion': "Estadísticamente Significativo (p < 0.05)" if is_significant else "Estadísticamente Indistinguible (IC incluye el 0, p >= 0.05)"
    }

def block_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_ids: np.ndarray,
    q_lower: Optional[np.ndarray] = None,
    q_upper: Optional[np.ndarray] = None,
    n_replications: int = 2000,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap por BLOQUES ESPACIALES (resampling de bloques completos para preservar autocorrelación).
    Calcula Intervalos de Confianza al 95% para R², RMSE, MAE, NSE, PICP y PINAW.
    """
    np.random.seed(random_state)
    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)
    
    metrics_dist = {
        'R2': [],
        'RMSE': [],
        'MAE': [],
        'NSE': [],
        'Pearson_R2': []
    }
    has_intervals = (q_lower is not None) and (q_upper is not None)
    fixed_y_range = float(np.max(y_true) - np.min(y_true))
    if fixed_y_range < 1e-6:
        fixed_y_range = 100.0
        
    if has_intervals:
        metrics_dist['PICP'] = []
        metrics_dist['PINAW'] = []
        metrics_dist['Winkler'] = []
        
    for _ in range(n_replications):
        boot_blocks = np.random.choice(unique_blocks, size=n_blocks, replace=True)
        # Extraer índices correspondientes a los bloques seleccionados
        boot_idx = []
        for b in boot_blocks:
            boot_idx.extend(np.where(block_ids == b)[0])
        boot_idx = np.array(boot_idx)
        
        y_t_b = y_true[boot_idx]
        y_p_b = y_pred[boot_idx]
        
        if len(np.unique(y_t_b)) < 2:
            continue
            
        m = calculate_metrics(y_t_b, y_p_b)
        for k in ['R2', 'RMSE', 'MAE', 'NSE', 'Pearson_R2']:
            metrics_dist[k].append(m[k])
            
        if has_intervals:
            unc = calculate_uncertainty_metrics(
                y_t_b, q_lower[boot_idx], q_upper[boot_idx], y_range=fixed_y_range
            )
            metrics_dist['PICP'].append(unc['PICP_Percent'])
            metrics_dist['PINAW'].append(unc['PINAW_Percent'])
            metrics_dist['Winkler'].append(unc['Winkler_Score'])
            
    ci_results = {}
    for k, vals in metrics_dist.items():
        if len(vals) == 0:
            continue
        v_lo = float(np.percentile(vals, 2.5))
        v_hi = float(np.percentile(vals, 97.5))
        v_med = float(np.median(vals))
        ci_results[k] = {
            'ci_lower': round(v_lo, 4),
            'median': round(v_med, 4),
            'ci_upper': round(v_hi, 4)
        }
        
    return ci_results
