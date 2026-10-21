"""F3: consistencia API / base de datos / tablas de la Fase 5, ausencia de ranking y de nombres, validación de entradas."""
import csv

import pytest

from conftest import DIR_DATOS


def _csv(n):
    with open(DIR_DATOS / n, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("tarea", ["mortalidad", "estadia"])
def test_api_coincide_con_tablas_fase5(cliente, tarea):
    api = cliente.get(f"/indicadores/{tarea}").json()
    ref = {f["cod_hospital"]: f for f in _csv("indicador_hospital.csv") if f["tarea"] == tarea}
    assert api["n_hospitales"] == len(api["puntos"]) == len(ref) == 69
    for p in api["puntos"]:
        f = ref[p["cod_hospital"]]
        assert p["n_egresos"] == int(f["n_egresos"])
        assert p["esperado"] == pytest.approx(float(f["esperado"]))
        if f["oe"] == "":
            assert p["oe"] is None
        else:
            assert p["oe"] == pytest.approx(float(f["oe"]))
            assert p["zona"] == f["zona"]


def test_ficha_coincide_con_lista(cliente):
    cods = cliente.get("/hospitales").json()["hospitales"]
    assert len(cods) == 69
    for c in cods[:10]:
        f = cliente.get(f"/hospitales/{c}").json()
        assert set(f["indicadores"]) == {"mortalidad", "estadia"}


def test_hospital_inexistente_404(cliente):
    assert cliente.get("/hospitales/999999").status_code == 404
    assert cliente.get("/indicadores/otra").status_code == 404


def test_orden_por_codigo_y_sin_ranking(cliente):
    for t in ("mortalidad", "estadia"):
        d = cliente.get(f"/indicadores/{t}").json()
        cods = [p["cod_hospital"] for p in d["puntos"]]
        assert cods == sorted(cods)
        for p in d["puntos"]:
            assert not {"ranking", "posicion", "rank", "lugar", "puesto"} & set(p)
    assert cliente.get("/hospitales").json()["hospitales"] == sorted(cliente.get("/hospitales").json()["hospitales"])


def test_sin_nombres_de_hospital(cliente):
    cuerpo = "".join(cliente.get(r).text.lower() for r in ("/hospitales", "/indicadores/mortalidad", "/hospitales/101100", "/metodologia"))
    for palabra in ("hospital clínico", "hospital de ", "complejo hospitalario", "hospital base", "rut"):
        assert palabra not in cuerpo, palabra


def test_advertencias_visibles(cliente):
    adv = cliente.get("/metodologia").json()["advertencias"]
    assert len(adv) >= 4 and all(a["texto"] for a in adv)


def test_demo_valida_entradas_sin_ecoar(cliente, perfil):
    malo = {**perfil, "edad": 400, "diagnostico_principal": "SECRETO_XYZ"}
    r = cliente.post("/demo/riesgo", json=malo)
    assert r.status_code == 422 and "SECRETO_XYZ" not in r.text
    r = cliente.post("/demo/riesgo", json={**perfil, "diagnostico_principal": "ZZZ"})
    assert r.status_code == 422 and "ZZZ" not in r.text


def test_demo_sin_cache(cliente, perfil):
    r = cliente.post("/demo/riesgo", json=perfil)
    assert r.status_code == 200 and "no-store" in r.headers["cache-control"]
