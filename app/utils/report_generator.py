"""
Generador de Reportes Ejecutivos en PDF para Sequías Agrícolas en Puno (US-41)
Utiliza ReportLab y Matplotlib para generar un informe técnico oficial para MIDAGRI / DREP / COER Puno.
"""
import io
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

from app.config import DROUGHT_CATEGORIES

# Distritos de referencia representativos de Puno
PUNO_DISTRICTS_REF = [
    {"distrito": "Taraco", "provincia": "Huancané", "lon": -69.98, "lat": -15.30},
    {"distrito": "Huancané", "provincia": "Huancané", "lon": -69.76, "lat": -15.20},
    {"distrito": "Juliaca", "provincia": "San Román", "lon": -70.13, "lat": -15.50},
    {"distrito": "Ilave", "provincia": "El Collao", "lon": -69.64, "lat": -16.08},
    {"distrito": "Juli", "provincia": "Chucuito", "lon": -69.46, "lat": -16.21},
    {"distrito": "Azángaro", "provincia": "Azángaro", "lon": -70.19, "lat": -14.91},
    {"distrito": "Ayaviri", "provincia": "Melgar", "lon": -70.59, "lat": -14.88},
    {"distrito": "Lampa", "provincia": "Lampa", "lon": -70.36, "lat": -15.36},
    {"distrito": "Yunguyo", "provincia": "Yunguyo", "lon": -69.09, "lat": -16.24},
    {"distrito": "Macusani", "provincia": "Carabaya", "lon": -70.43, "lat": -14.07},
    {"distrito": "Sandia", "provincia": "Sandia", "lon": -69.43, "lat": -14.33},
    {"distrito": "Puno", "provincia": "Puno", "lon": -70.02, "lat": -15.84}
]

def generate_severity_chart_bytes(dist_dict: Dict[str, int]) -> bytes:
    """Genera imagen PNG del gráfico de distribución de severidad para incrustar en PDF"""
    labels = DROUGHT_CATEGORIES['labels']
    values = [dist_dict.get(cat, 0) for cat in labels]
    bar_colors = DROUGHT_CATEGORIES['colors']
    
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=200)
    bars = ax.bar(labels, values, color=bar_colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel('N° Puntos Evaluados', fontsize=9, fontweight='bold')
    ax.set_title('Distribución Espacial de Severidad de Sequía Agrícola (VHI)', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=12, labelsize=8)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
                    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def find_district_severities(coords: np.ndarray, q50: np.ndarray, q05: np.ndarray, q95: np.ndarray, labels: list) -> list:
    """Asocia las predicciones a los distritos de referencia de Puno"""
    district_data = []
    for d in PUNO_DISTRICTS_REF:
        d_lon, d_lat = d['lon'], d['lat']
        # Distancia euclidiana mínima
        dists = np.sqrt((coords[:, 0] - d_lon)**2 + (coords[:, 1] - d_lat)**2)
        idx_min = int(np.argmin(dists))
        district_data.append({
            'distrito': f"{d['distrito']} ({d['provincia']})",
            'lon': round(d_lon, 3),
            'lat': round(d_lat, 3),
            'vhi': round(float(q50[idx_min]), 1),
            'q05': round(float(q05[idx_min]), 1),
            'q95': round(float(q95[idx_min]), 1),
            'severidad': labels[idx_min]
        })
    # Ordenar por VHI ascendente (más críticos primero)
    district_data.sort(key=lambda x: x['vhi'])
    return district_data

def generate_pdf_report(latest_predictions: Dict[str, Any]) -> io.BytesIO:
    """
    Genera el informe técnico de alerta temprana en PDF (US-41).
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#1a365d'),
        alignment=1,
        fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#4a5568'),
        alignment=1,
        fontName='Helvetica-Oblique'
    )
    section_style = ParagraphStyle(
        'DocSection',
        parent=styles['Heading2'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#2b6cb0'),
        fontName='Helvetica-Bold',
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#2d3748')
    )
    
    elements = []
    
    # 1. Cabecera oficial
    elements.append(Paragraph("SISTEMA DE ALERTA TEMPRANA Y MONITOREO DE SEQUÍAS AGRÍCOLAS", title_style))
    elements.append(Paragraph("GOBIERNO REGIONAL DE PUNO — PLATAFORMA DE INTELIGENCIA ESPACIAL (GeoAI)", subtitle_style))
    elements.append(Spacer(1, 10))
    
    # Metadatos del informe
    stats = latest_predictions.get('stats', {})
    model_used = stats.get('model_used', 'Red Neuronal Cuantil')
    fecha_emision = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    meta_table_data = [
        [
            Paragraph(f"<b>Fecha de Emisión:</b> {fecha_emision}", body_style),
            Paragraph(f"<b>Ámbito Territorial:</b> Región Puno (Altiplano)", body_style)
        ],
        [
            Paragraph(f"<b>Modelo Predictivo:</b> {model_used.upper()}", body_style),
            Paragraph("<b>Variable Objetivo:</b> VHI (Vegetation Health Index, 0-100%)", body_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#edf2f7')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e0')),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))
    
    # 2. Resumen Ejecutivo
    elements.append(Paragraph("1. RESUMEN EJECUTIVO Y SITUACIÓN AGROMETEOROLÓGICA", section_style))
    dist = stats.get('severity_distribution', {})
    total_pts = stats.get('total_points', 400)
    extrema_pts = dist.get('Sequía extrema', 0)
    severa_pts = dist.get('Sequía severa', 0)
    critica_pct = ((extrema_pts + severa_pts) / max(1, total_pts)) * 100.0
    mean_vhi = stats.get('mean_prediction', 50.0)
    mean_unc = stats.get('mean_uncertainty', 9.0)
    
    resumen_txt = (
        f"El monitoreo satelital e inferencia espacial mediante GeoAI para la región de Puno registra un "
        f"<b>Índice de Salud de la Vegetación promedio de {mean_vhi:.1f}%</b>. "
        f"Se estima que el <b>{critica_pct:.1f}%</b> de la superficie agrícola evaluada se encuentra bajo "
        f"condición de <b>sequía crítica (severa o extrema)</b>, requiriendo atención prioritaria en los comités "
        f"de gestión de riesgos agrarios. La incertidumbre predictiva media al 90% de confianza es de ±{mean_unc/2.0:.1f} puntos de VHI, "
        f"lo que evidencia alta confiabilidad en los modelos biofísicos implementados."
    )
    elements.append(Paragraph(resumen_txt, body_style))
    elements.append(Spacer(1, 8))
    
    # 3. Gráfico de Severidad y Estadísticas Regionales
    elements.append(Paragraph("2. DISTRIBUCIÓN DE SEVERIDAD DE SEQUÍA (NORMA FAO/OMM)", section_style))
    chart_bytes = generate_severity_chart_bytes(dist)
    chart_img = Image(io.BytesIO(chart_bytes), width=420, height=190)
    chart_img.hAlign = 'CENTER'
    elements.append(chart_img)
    elements.append(Spacer(1, 8))
    
    # 4. Tabla de Distritos en Riesgo
    elements.append(Paragraph("3. DISTRITOS Y SECTORES AGRARIOS PRIORITARIOS EN PUNO", section_style))
    coords = latest_predictions['coords']
    q50 = latest_predictions['q50']
    q05 = latest_predictions['q05']
    q95 = latest_predictions['q95']
    severity_labels = latest_predictions['severity_labels']
    
    district_list = find_district_severities(coords, q50, q05, q95, severity_labels)
    
    dist_table_data = [
        ["Distrito / Sector", "Coord. (Lon, Lat)", "VHI Est.", "IC 90% [q₀.₀₅ - q₀.₉₅]", "Nivel de Riesgo"]
    ]
    for d in district_list[:8]:  # Mostrar los 8 más prioritarios
        dist_table_data.append([
            d['distrito'],
            f"{d['lon']}, {d['lat']}",
            f"{d['vhi']}%",
            f"[{d['q05']} - {d['q95']}%]",
            d['severidad']
        ])
        
    dist_table = Table(dist_table_data, colWidths=[150, 110, 60, 120, 100])
    dist_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('PADDING', (0,0), (-1,-1), 4),
        ('FONTSIZE', (0,1), (-1,-1), 7.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')])
    ]))
    elements.append(dist_table)
    elements.append(Spacer(1, 10))
    
    # 5. Recomendaciones Agronómicas
    elements.append(Paragraph("4. RECOMENDACIONES DE ALERTA TEMPRANA PARA AGROMETEOROLOGÍA", section_style))
    recs = (
        "• <b>Sectores en Sequía Severa/Extrema:</b> Activar monitoreo continuo de humedad en suelo a 0-20 cm, "
        "restringir turnos de riego en valles interandinos y coordinar asistencia con el Seguro Agrícola Catastrófico (SAC).<br/>"
        "• <b>Sectores en Sequía Moderada:</b> Promover prácticas de labranza mínima para conservación de humedad edáfica "
        "y aplicación de mulch o coberturas vegetales en cultivos de quinua y papa.<br/>"
        "• <b>Incertidumbre:</b> Para áreas con ancho de banda [q₀.₀₅, q₀.₉₅] > 30%, complementar con mediciones de estaciones automáticas SENAMHI."
    )
    elements.append(Paragraph(recs, body_style))
    elements.append(Spacer(1, 10))
    
    # Pie de página técnico
    footer_text = Paragraph(
        "<i>Documento técnico generado automáticamente por el Sistema GeoAI Puno. "
        "Válido para gestión de riesgos agrarios ante el COER Puno y MIDAGRI.</i>",
        subtitle_style
    )
    elements.append(footer_text)
    
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer
