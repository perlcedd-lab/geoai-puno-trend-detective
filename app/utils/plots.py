"""
Utilidades de visualización
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple
import os

def plot_spatial_map(
    coords: np.ndarray,
    values: np.ndarray,
    title: str = "Mapa Espacial",
    cmap: str = 'viridis',
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Grafica mapa espacial de puntos
    
    Args:
        coords: Array (N, 2) con [lon, lat]
        values: Array (N,) con valores
        title: Título del gráfico
        cmap: Colormap
        save_path: Ruta para guardar
        figsize: Tamaño de figura
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        cmap=cmap,
        s=50,
        alpha=0.7,
        edgecolors='k',
        linewidths=0.5
    )
    
    plt.colorbar(scatter, ax=ax, label='Valor')
    
    ax.set_xlabel('Longitud', fontsize=12)
    ax.set_ylabel('Latitud', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Mapa guardado: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison_maps(
    coords: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    uncertainty: np.ndarray = None,
    save_path: str = None
):
    """
    Gráfico comparativo: observado, predicho, error, incertidumbre
    """
    n_plots = 4 if uncertainty is not None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    
    error = observed - predicted
    
    # Observado
    sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=observed, cmap='viridis', s=30)
    axes[0].set_title('Observado')
    plt.colorbar(sc1, ax=axes[0])
    
    # Predicho
    sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=predicted, cmap='viridis', s=30)
    axes[1].set_title('Predicho')
    plt.colorbar(sc2, ax=axes[1])
    
    # Error
    sc3 = axes[2].scatter(coords[:, 0], coords[:, 1], c=error, cmap='RdBu_r', s=30)
    axes[2].set_title('Error (Obs - Pred)')
    plt.colorbar(sc3, ax=axes[2])
    
    # Incertidumbre
    if uncertainty is not None:
        sc4 = axes[3].scatter(coords[:, 0], coords[:, 1], c=uncertainty, cmap='Reds', s=30)
        axes[3].set_title('Incertidumbre')
        plt.colorbar(sc4, ax=axes[3])
    
    for ax in axes:
        ax.set_xlabel('Lon')
        ax.set_ylabel('Lat')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Comparación guardada: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_comparison(
    results_dict: dict,
    metric: str = 'RMSE',
    save_path: str = None
):
    """
    Compara múltiples modelos en un gráfico de barras
    
    Args:
        results_dict: Dict con {model_name: metrics_dict}
        metric: Métrica a comparar
        save_path: Ruta para guardar
    """
    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # Agregar valores encima de barras
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Comparación de Modelos - {metric}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Comparación guardada: {save_path}")
    else:
        plt.show()
    
    plt.close()