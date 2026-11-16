"""
Validación de Esquema de Datos Geoespaciales con Pandera.
Garantiza coherencia física, tipos de datos, ausencia de nulos y unicidad espacio-temporal.
"""
import os
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
try:
    import pandera.pandas as pa
except ImportError:
    import pandera as pa
from pandera import Column, Check, DataFrameSchema
import pandas as pd
from typing import Tuple

PunoDataSchema = DataFrameSchema(
    columns={
        "cell_id": Column(str, nullable=False),
        "date": Column(str, nullable=False),
        "lon": Column(float, Check.in_range(-71.5, -68.5), nullable=False),
        "lat": Column(float, Check.in_range(-17.5, -13.0), nullable=False),
        "vci": Column(float, Check.in_range(0.0, 100.0), nullable=False),
        "tci": Column(float, Check.in_range(0.0, 100.0), nullable=False),
        "lst": Column(float, Check.in_range(-15.0, 45.0), nullable=False), # Permite heladas nocturnas < 0°C
        "sm": Column(float, Check.in_range(0.02, 0.50), nullable=False),
        "spi_1": Column(float, Check.in_range(-4.5, 4.5), nullable=False),
        "spi_3": Column(float, Check.in_range(-4.5, 4.5), nullable=False),
        "dem": Column(float, Check.in_range(3812.0, 5500.0), nullable=False),
        "slope": Column(float, Check.in_range(0.0, 60.0), nullable=False),
        "aspect": Column(float, Check.in_range(0.0, 360.0), nullable=False),
        "dist_lake": Column(float, Check.in_range(0.0, 300.0), nullable=False),
        "precip_lag": Column(float, Check.greater_than_or_equal_to(0.0), nullable=False),
        "vhi": Column(float, Check.in_range(0.0, 100.0), nullable=False),
        "vhi_lead16": Column(float, Check.in_range(0.0, 100.0), nullable=True),
        "vci_lag16": Column(float, Check.in_range(0.0, 100.0), nullable=True),
        "tci_lag16": Column(float, Check.in_range(0.0, 100.0), nullable=True),
        "vhi_lag16": Column(float, Check.in_range(0.0, 100.0), nullable=True),
        "doy_sin": Column(float, Check.in_range(-1.0, 1.0), nullable=False),
        "doy_cos": Column(float, Check.in_range(-1.0, 1.0), nullable=False),
        "smap_qa_pass": Column(int, Check.isin([0, 1]), nullable=False)
    },
    unique=["cell_id", "date"],
    coerce=True,
    strict=False
)

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Valida un DataFrame frente al esquema oficial de Pandera.
    """
    try:
        PunoDataSchema.validate(df, lazy=True)
        return True, "Dataset cumple al 100% con el esquema de Pandera (tipos, rangos, unicidad y no-nulos)."
    except Exception as ex:
        return False, f"Fallo de validación de esquema: {str(ex)}"
