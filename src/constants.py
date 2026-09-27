"""
constants.py
------------
Constantes globales y contratos para el procesamiento de datos GRD.
"""

from pathlib import Path

# Directorios predeterminados
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
DOCS_DIR = BASE_DIR / "docs"

# Alias conocidos entre extracciones de FONASA
ALIAS_COLUMNAS = {
    "ID_BENEFICIARIO": "CIP_ENCRIPTADO"
}

# Formatos de fecha observados en la base de datos
FORMATO_FECHA_NACIMIENTO = "%Y-%m-%d"  # yyyy-mm-dd
FORMATO_FECHAS_DMY = "%d-%m-%Y"        # dd-mm-yyyy (ingreso, alta, traslados)

# Columnas con información posterior al ingreso que NUNCA deben entrar al modelo
COLUMNAS_PROHIBIDAS_FUGA = {
    # Procedimientos quirúrgicos y fechas médicas
    *[f"PROCEDIMIENTO{i}" for i in range(1, 31)],
    *[f"FECHATRASLADO{i}" for i in range(1, 10)],
    *[f"SERVICIOTRASLADO{i}" for i in range(1, 10)],
    "FECHAINTERV1",
    "ESPECIALIDADINTERVENCION",
    "MEDICOINTERV1_ENCRIPTADO",
    "MEDICOALTA_ENCRIPTADO",
    "USOSPABELLON",
    "SERVICIOALTA",
    # Resultados o fechas de egreso
    "FECHAALTA",
    "TIPOALTA",
    "IR_29301_MORTALIDAD",
    "IR_29301_PESO",
    "IR_29301_SEVERIDAD",
    # Identificadores institucionales (para evitar efecto prestador en entrenamiento)
    "COD_HOSPITAL",
    "SERVICIO_SALUD",
    "ID_EPISODIO",
    "CIP_ENCRIPTADO"
}
