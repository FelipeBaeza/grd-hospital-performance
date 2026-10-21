"""Configuración por variables de entorno (con valores para desarrollo local)."""
from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://indicadores:indicadores@localhost:5432/indicadores")
DIR_MODELOS = Path(os.environ.get("DIR_MODELOS", RAIZ.parent / "fase4_modelado" / "modelos"))
DIR_DATOS = Path(os.environ.get("DIR_DATOS", RAIZ / "datos" / "publicables"))
CORS_ORIGENES = [o for o in os.environ.get("CORS_ORIGENES", "http://localhost:8080,http://localhost:5173").split(",") if o]
