"""
Red Neuronal Tabular Compacta con Regresión por Cuantiles (Incertidumbre al 90% de Confianza)
Cumple con US-20, US-21, US-22, US-24 de correccion1.txt
Arquitectura: Input(8) -> Dense(64) + Swish + BN + Dropout(0.25) -> Dense(32) + Swish + BN + Dropout(0.20) -> Dense(16) -> Dense(3)
Parámetros totales: < 5,000 parámetros (adiós sobreparametrización de 12 millones de pesos)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import os
import pickle
from typing import List, Tuple, Dict, Optional

QUANTILES = [0.05, 0.50, 0.95]

@keras.utils.register_keras_serializable(package="CustomLoss")
def pinball_loss_multiquantile(y_true, y_pred):
    """
    Función de pérdida Pinball Loss (Quantile Loss) multicuantil para tau = [0.05, 0.50, 0.95] (US-21).
    Calcula simultáneamente el cuantil 5% (límite inferior), 50% (mediana) y 95% (límite superior).
    """
    tau = tf.constant([0.05, 0.50, 0.95], dtype=tf.float32)
    # y_true shape: (batch, 1), y_pred shape: (batch, 3)
    y_true_broadcast = tf.cast(y_true, tf.float32)
    if tf.rank(y_true_broadcast) == 1:
        y_true_broadcast = tf.expand_dims(y_true_broadcast, -1)
    
    err = y_true_broadcast - y_pred  # Broadcast: (batch, 3)
    loss = tf.maximum(tau * err, (tau - 1.0) * err)
    return tf.reduce_mean(loss)

def build_quantile_model(
    input_dim: int = 8,
    hidden_units: List[int] = [64, 32, 16],
    dropouts: List[float] = [0.25, 0.20],
    learning_rate: float = 1e-3
) -> Model:
    """
    Construye la arquitectura tabular compacta (US-20).
    Parámetros totales < 5,000 para prevenir sobreajuste en datasets espaciales.
    """
    inputs = layers.Input(shape=(input_dim,), name='input_features')
    
    # Capa 1
    x = layers.Dense(hidden_units[0], activation='swish', kernel_regularizer=keras.regularizers.l2(1e-4), name='dense_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(dropouts[0], name='drop_1')(x)
    
    # Capa 2
    x = layers.Dense(hidden_units[1], activation='swish', kernel_regularizer=keras.regularizers.l2(1e-4), name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(dropouts[1], name='drop_2')(x)
    
    # Capa 3
    x = layers.Dense(hidden_units[2], activation='linear', name='dense_3')(x)
    
    # Salida: 3 neuronas lineales correspondientes a [q_0.05, q_0.50, q_0.95]
    outputs = layers.Dense(3, activation='linear', name='quantiles_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Quantile_FCNN_Puno')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=pinball_loss_multiquantile,
        metrics=['mae']
    )
    
    return model

class QuantileNNWrapper:
    """
    Envoltorio de alto nivel para entrenamiento e inferencia de la Red Cuantil (US-22).
    """
    def __init__(self, input_dim: int = 8, model: Optional[Model] = None):
        self.input_dim = input_dim
        self.model = model if model is not None else build_quantile_model(input_dim=input_dim)
        self.quantiles = QUANTILES
        
    def summary(self):
        return self.model.summary()
        
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None, verbose=1):
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
            ]
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
    def predict_quantiles(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predice los tres cuantiles y calcula la banda de incertidumbre al 90% (US-22).
        
        Returns:
            dict con:
                - 'q05': Límite inferior (cuantil 5%)
                - 'q50': Mediana predicha (cuantil 50% / valor central)
                - 'q95': Límite superior (cuantil 95%)
                - 'uncertainty_90': Ancho del intervalo (q95 - q05)
        """
        preds = self.model.predict(X, verbose=0)
        q05 = preds[:, 0]
        q50 = preds[:, 1]
        q95 = preds[:, 2]
        
        # Corrección de monotonía de cuantiles (garantizar q05 <= q50 <= q95)
        q50 = np.maximum(q50, q05)
        q95 = np.maximum(q95, q50)
        uncertainty_90 = q95 - q05
        
        return {
            'q05': q05,
            'q50': q50,
            'q95': q95,
            'uncertainty_90': uncertainty_90
        }
        
    def save(self, filepath: str):
        """Guarda modelo Keras (US-24)"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"💾 Modelo Quantile NN guardado en: {filepath}")
        
    @classmethod
    def load(cls, filepath: str):
        """Carga modelo serializado"""
        model = keras.models.load_model(
            filepath,
            custom_objects={'pinball_loss_multiquantile': pinball_loss_multiquantile}
        )
        instance = cls(input_dim=model.input_shape[-1], model=model)
        print(f"✅ Modelo Quantile NN cargado exitosamente desde: {filepath}")
        return instance

if __name__ == '__main__':
    wrapper = QuantileNNWrapper(input_dim=8)
    wrapper.summary()
    print("✅ QuantileNNWrapper verificado exitosamente.")
