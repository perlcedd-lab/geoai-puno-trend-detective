"""
Modelo Baseline: LightGBM con Regresión por Cuantiles (tau = [0.05, 0.50, 0.95])
Entrenado sobre las mismas features y split para comparación rigurosa multi-modelo.
"""
import lightgbm as lgb
import numpy as np
import pickle
import os
from typing import Dict, Tuple, Optional

class LightGBMQuantileModel:
    """
    Ensamble de 3 modelos LightGBM con función de pérdida por cuantiles (Pinball Loss)
    con soporte para calibración conforme CQR (Romano et al. 2019) y corrección anti-crossing.
    """
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.models = {}
        self.q_hat_cqr: Optional[float] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Entrena independientemente los tres cuantiles tau = [0.05, 0.50, 0.95]"""
        quantiles = [0.05, 0.50, 0.95]
        for q in quantiles:
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                verbosity=-1
            )
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
                )
            else:
                model.fit(X_train, y_train)
            self.models[q] = model
            
        if X_val is not None and y_val is not None:
            self.calibrate_cqr(X_val, y_val, alpha=0.10)

    def calibrate_cqr(self, X_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.10):
        """Calcula el factor de corrección conforme q_hat sobre el conjunto de calibración"""
        raw_preds = self.predict_quantiles(X_cal, apply_cqr=False)
        q05 = raw_preds['q05']
        q95 = raw_preds['q95']
        n_cal = len(y_cal)
        scores = np.maximum(q05 - y_cal, y_cal - q95)
        k = min(1.0, float(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal))
        self.q_hat_cqr = float(np.quantile(scores, k, method='higher'))

    def predict_quantiles(self, X: np.ndarray, apply_cqr: bool = True) -> Dict[str, np.ndarray]:
        """Predice cuantiles y asegura monotonicidad (anti-crossing)"""
        q05 = self.models[0.05].predict(X)
        q50 = self.models[0.50].predict(X)
        q95 = self.models[0.95].predict(X)
        
        # Guardar versiones crudas
        q05_raw = np.copy(q05)
        q95_raw = np.copy(q95)
        
        if apply_cqr and (self.q_hat_cqr is not None):
            q05 = q05 - self.q_hat_cqr
            q95 = q95 + self.q_hat_cqr
            
        # Rearreglo monotónico estricto
        q05 = np.clip(q05, 0.0, 100.0)
        q50 = np.clip(q50, 0.0, 100.0)
        q95 = np.clip(q95, 0.0, 100.0)
        
        q50 = np.maximum(q05, np.minimum(q50, q95))
        q95 = np.maximum(q95, q50)
        
        return {
            'q05': q05,
            'q50': q50,
            'q95': q95,
            'q05_raw': q05_raw,
            'q95_raw': q95_raw,
            'uncertainty_90': q95 - q05,
            'q_hat': self.q_hat_cqr
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
