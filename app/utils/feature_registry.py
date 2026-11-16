"""
Feature Registry Central — Única fuente de verdad de variables del modelo.
Garantiza el cumplimiento estricto del protocolo anti-fuga (Target Leakage Prevention).
"""
import os
import yaml
from typing import List, Dict, Any, Optional

DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../experiment_config.yaml'))

class FeatureRegistry:
    """
    Controlador central de variables permitidas y prohibidas según el modo experimental.
    """
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Archivo de configuración no encontrado en: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        self.modes = self.cfg.get('modes', {})
        self.active_mode = self.cfg.get('active_mode', 'mode_b_forecast')

    @property
    def config(self) -> Dict[str, Any]:
        return self.cfg

    def get_target(self, mode: Optional[str] = None) -> str:
        mode = mode or self.active_mode
        if mode not in self.modes:
            raise KeyError(f"Modo desconocido: {mode}. Modos disponibles: {list(self.modes.keys())}")
        return self.modes[mode]['target']

    def get_features(self, mode: Optional[str] = None) -> List[str]:
        mode = mode or self.active_mode
        if mode not in self.modes:
            raise KeyError(f"Modo desconocido: {mode}. Modos disponibles: {list(self.modes.keys())}")
        return list(self.modes[mode]['features'])

    def get_forbidden_features(self, mode: Optional[str] = None) -> List[str]:
        mode = mode or self.active_mode
        if mode not in self.modes:
            raise KeyError(f"Modo desconocido: {mode}")
        return list(self.modes[mode].get('forbidden_features', []))

    def get_allowed_features(self, mode: Optional[str] = None) -> List[str]:
        mode = mode or self.active_mode
        if mode not in self.modes:
            raise KeyError(f"Modo desconocido: {mode}")
        return list(self.modes[mode].get('allowed_features', self.modes[mode].get('features', [])))

    def validate_features(self, features: List[str], mode: Optional[str] = None) -> bool:
        """
        Ejecuta aserciones duras para garantizar que ninguna variable prohibida entre al modelo.
        """
        mode = mode or self.active_mode
        forbidden = set(self.get_forbidden_features(mode))
        current = set(features)
        
        leakage = current.intersection(forbidden)
        if leakage:
            raise AssertionError(
                f"🚨 [TARGET LEAKAGE DETECTADO] En modo '{mode}', las siguientes variables están "
                f"ESTRICTAMENTE PROHIBIDAS y violan el protocolo científico: {sorted(list(leakage))}"
            )
            
        allowed = set(self.get_allowed_features(mode))
        if mode != 'legacy_leaky':
            unrecognized = current - allowed
            if unrecognized:
                raise ValueError(
                    f"⚠️ Variables no registradas en la lista de permitidas para '{mode}': {sorted(list(unrecognized))}"
                )
        return True

    def validate_dataframe(self, df, mode: Optional[str] = None) -> bool:
        """Verifica que el subconjunto de features a extraer de un DataFrame no contenga fugas"""
        feats = self.get_features(mode)
        self.validate_features(feats, mode)
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise KeyError(f"Columnas requeridas para '{mode}' faltantes en el DataFrame: {missing}")
        return True

# Instancia singleton para uso general
registry = FeatureRegistry()
