import csv
from pathlib import Path

import polars as pl
import pytest

from evaluacion import apertura
from evaluacion.config import Protocolo, cargar_protocolo


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    sellado = tmp_path / "sellado"
    sellado.mkdir()
    pl.DataFrame({"ID_EPISODIO": ["a", "b"], "ROL": ["EVENTO", "NO_EVENTO"], "ESTANCIA_DIAS": [1, 2]}).write_parquet(sellado / "desenlace_2024.parquet")
    registro = tmp_path / "registro.csv"
    monkeypatch.setenv("FASE3_REGISTRO_ACCESOS", str(registro))
    prot = cargar_protocolo()
    pre = tmp_path / "pre.json"
    apertura.registrar_preapertura(pre, prot, {})
    return {"tmp": tmp_path, "sellado": sellado, "registro": registro, "prot": prot, "pre": pre}


def _auth(ruta: Path, texto="autorizo-fase-5\nfirmante: Profesor guía\nfecha: hoy\n"):
    ruta.write_text(texto, encoding="utf8")
    return ruta


def test_sin_autorizacion_no_se_abre(entorno):
    e = entorno
    with pytest.raises(apertura.AperturaNoAutorizada):
        apertura.abrir_2024(e["sellado"], e["prot"], e["pre"], e["tmp"] / "no_existe.txt", e["registro"], lambda: None, e["tmp"] / "out" / "d.parquet")
    assert apertura.lecturas_concedidas(e["registro"]) == 0


def test_autorizacion_incompleta_se_rechaza(entorno):
    e = entorno
    a = _auth(e["tmp"] / "a.txt", "autorizo-fase-5\n")
    with pytest.raises(apertura.AperturaNoAutorizada):
        apertura.abrir_2024(e["sellado"], e["prot"], e["pre"], a, e["registro"], lambda: None, e["tmp"] / "out" / "d.parquet")


def test_apertura_unica(entorno):
    e = entorno
    a = _auth(e["tmp"] / "a.txt")
    d = apertura.abrir_2024(e["sellado"], e["prot"], e["pre"], a, e["registro"], lambda: None, e["tmp"] / "out" / "d.parquet")
    assert d.height == 2 and apertura.lecturas_concedidas(e["registro"]) == 1
    with pytest.raises(apertura.AperturaNoAutorizada):
        apertura.abrir_2024(e["sellado"], e["prot"], e["pre"], a, e["registro"], lambda: None, e["tmp"] / "out2" / "d.parquet")


def test_protocolo_modificado_impide_abrir(entorno, tmp_path):
    e = entorno
    copia = tmp_path / "protocolo.toml"
    copia.write_text(e["prot"].ruta.read_text(encoding="utf8"), encoding="utf8")
    p1 = cargar_protocolo(copia)
    pre = tmp_path / "pre2.json"
    apertura.registrar_preapertura(pre, p1, {})
    copia.write_text(copia.read_text(encoding="utf8") + "\n# cambio posterior\n", encoding="utf8")
    p2 = cargar_protocolo(copia)
    a = _auth(tmp_path / "a.txt")
    with pytest.raises(apertura.AperturaNoAutorizada):
        apertura.abrir_2024(e["sellado"], p2, pre, a, e["registro"], lambda: None, tmp_path / "o" / "d.parquet")


def test_el_codigo_de_la_fase_5_no_entrena_ni_recalibra_con_2024():
    import re
    for f in (Path(__file__).resolve().parents[1] / "evaluacion").glob("*.py"):
        if f.name in ("planes_b.py", "verificacion.py"):          # planes_b recalibra con 2023; verificacion contiene los patrones a modo de búsqueda
            continue
        assert not re.search(r"\.fit\(|ajustar_final|entrenar\(", f.read_text(encoding="utf8")), f.name


def test_la_enmienda_03_permite_una_segunda_lectura_y_no_una_tercera(entorno, tmp_path):
    from evaluacion.config import RAIZ
    e = entorno
    a1 = _auth(e["tmp"] / "a.txt")
    apertura.abrir_2024(e["sellado"], e["prot"], e["pre"], a1, e["registro"], lambda: None, e["tmp"] / "o1" / "d.parquet")
    p4 = cargar_protocolo(RAIZ / "config" / "protocolo_fase5_v4.toml")
    pre4 = tmp_path / "pre4.json"
    apertura.registrar_preapertura(pre4, p4, {})
    a2 = _auth(tmp_path / "a2.txt", "autorizo-reapertura-2024\nfirmante: Profesor guía\nfecha: hoy\n")
    with pytest.raises(apertura.AperturaNoAutorizada):          # la frase del v1.0 no sirve para la segunda
        apertura.abrir_2024(e["sellado"], p4, pre4, a1, e["registro"], lambda: None, tmp_path / "x" / "d.parquet")
    d = apertura.abrir_2024(e["sellado"], p4, pre4, a2, e["registro"], lambda: None, tmp_path / "o2" / "d.parquet")
    assert d.height == 2 and apertura.lecturas_concedidas(e["registro"]) == 2
    with pytest.raises(apertura.AperturaNoAutorizada):
        apertura.abrir_2024(e["sellado"], p4, pre4, a2, e["registro"], lambda: None, tmp_path / "o3" / "d.parquet")
