import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def registro_temporal(tmp_path, monkeypatch):
    """El registro de accesos de cada prueba vive en una carpeta temporal y la autorización de 2024 nunca está activa."""
    monkeypatch.setenv("FASE3_REGISTRO_ACCESOS", str(tmp_path / "registro_accesos.csv"))
    monkeypatch.delenv("FASE3_AUTORIZA_DESENLACE_2024", raising=False)
