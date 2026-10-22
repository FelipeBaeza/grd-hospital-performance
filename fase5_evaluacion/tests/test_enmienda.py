"""La enmienda 01 solo vale con su registro, el registro no se sobrescribe y la corrida exige la enmienda sin cambios."""
import datetime as dt
import json
import shutil
import tomllib
from pathlib import Path

import pytest

from evaluacion import enmienda
from evaluacion.config import RAIZ

T = dt.datetime(2026, 10, 22, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def raiz(tmp_path: Path) -> Path:
    for rel in (enmienda.PROTOCOLO_V1, enmienda.PROTOCOLO_V2, enmienda.TEXTO):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(RAIZ / rel, tmp_path / rel)
    (tmp_path / "evaluacion").mkdir()
    (tmp_path / "evaluacion" / "modulo.py").write_text("x = 1\n")
    return tmp_path


def test_la_enmienda_real_es_coherente():
    d = tomllib.load(open(RAIZ / enmienda.PROTOCOLO_V2, "rb"))
    ids = [a["id"] for a in d["analisis"]]
    assert len(ids) == len(set(ids)) == 15
    assert d["regla_lectura"]["min_misma_zona"] == 0.80 and d["regla_lectura"]["min_rho"] == 0.90
    assert d["disenos"]["anidado"]["probar"] == 2023 and 2024 not in d["disenos"]["anidado"]["entrenar"]
    assert d["huella_base"] == enmienda.sha256_archivo(RAIZ / enmienda.PROTOCOLO_V1)


def test_firmar_exige_constancia_y_no_sobrescribe(raiz):
    with pytest.raises(enmienda.ErrorEnmienda, match="constancia"):
        enmienda.firmar("Autor", "Profesor", "2026-10-22", raiz=raiz, ahora=T)
    (raiz / enmienda.CONSTANCIA).write_bytes(b"%PDF-1.4 constancia")
    ruta = enmienda.firmar("Autor", "Profesor", "2026-10-22", raiz=raiz, ahora=T)
    reg = json.loads(ruta.read_text())
    assert reg["huellas"] == enmienda.huellas(raiz)
    assert reg["constancia"]["sha256"] and reg["codigo_evaluacion"] == {"evaluacion/modulo.py": enmienda.sha256_archivo(raiz / "evaluacion/modulo.py")}
    assert reg["momento_utc"].startswith("2026-10-22T12:00:00") and len(reg["analisis"]) == 15
    with pytest.raises(enmienda.ErrorEnmienda, match="ya está registrada"):
        enmienda.firmar("Autor", "Profesor", "2026-10-22", raiz=raiz, ahora=T)


def test_firmar_rechaza_protocolo_base_cambiado(raiz):
    (raiz / enmienda.PROTOCOLO_V1).write_text("version = 'otra'\n")
    with pytest.raises(enmienda.ErrorEnmienda, match="huella base"):
        enmienda.firmar("Autor", "Profesor", "2026-10-22", raiz=raiz, sin_constancia=True, ahora=T)


def test_la_nota_queda_en_el_registro(raiz):
    ruta = enmienda.firmar("Autor", "Profesor", "2026-10-22", raiz=raiz, sin_constancia=True, ahora=T, nota="aprobación verbal")
    reg = json.loads(ruta.read_text())
    assert reg["nota"] == "aprobación verbal" and reg["constancia"]["sha256"] is None


def test_corrida_exige_firma_y_enmienda_intacta(raiz):
    script = raiz / "sens.py"
    script.write_text("print('analisis')\n")
    with pytest.raises(enmienda.ErrorEnmienda, match="no está firmada"):
        enmienda.antes_de_correr(["sens.py"], raiz=raiz, ahora=T)
    enmienda.firmar("Autor", "Profesor", "2026-10-22", raiz=raiz, sin_constancia=True, ahora=T)
    (raiz / "evaluacion" / "modulo.py").write_text("x = 2\n")
    ruta = enmienda.antes_de_correr(["sens.py"], raiz=raiz, ahora=T)
    reg = json.loads(ruta.read_text())
    assert reg["scripts"] == {"sens.py": enmienda.sha256_archivo(script)}
    assert reg["codigo_cambiado_desde_firma"] == ["evaluacion/modulo.py"]


def test_corrida_falla_si_la_enmienda_cambio(raiz):
    enmienda.firmar("Autor", "Profesor", "2026-10-22", raiz=raiz, sin_constancia=True, ahora=T)
    with open(raiz / enmienda.TEXTO, "a", encoding="utf8") as f:
        f.write("\nCambio posterior a la firma.\n")
    (raiz / "sens.py").write_text("print(1)\n")
    with pytest.raises(enmienda.ErrorEnmienda, match="cambió después de firmarse"):
        enmienda.antes_de_correr(["sens.py"], raiz=raiz, ahora=T)
