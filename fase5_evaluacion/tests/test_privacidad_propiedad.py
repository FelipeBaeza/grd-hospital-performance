"""Plan 4.2: escáner de privacidad por propiedad. Cada prueba planta una fuga (canario) y exige que se detecte; la
última corre el escáner sobre las salidas reales del proyecto."""
from pathlib import Path

import pandas as pd

from evaluacion import privacidad_propiedad as pp
from evaluacion.config import RAIZ

DECLARADOS = RAIZ / "config" / "privacidad_hallazgos_declarados.csv"


def _escribir(carpeta: Path, nombre: str, df: pd.DataFrame) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    df.to_csv(carpeta / nombre, index=False)


def test_canario_tasa_por_tamano_en_otra_tabla(tmp_path):
    c = tmp_path / "anio"
    _escribir(c, "tamanos.csv", pd.DataFrame({"cod_hospital": ["A", "B"], "n_egresos": [1234, 5000]}))
    _escribir(c, "tasas.csv", pd.DataFrame({"cod_hospital": ["A", "B"], "tasa_traslado": [7 / 1234, 0.0]}))
    h = pp.reconstrucciones(tmp_path)
    assert [(x["cod_hospital"], x["conteo"]) for x in h] == [("A", 7)]


def test_canario_tasa_redondeada_que_igual_fija_el_conteo(tmp_path):
    _escribir(tmp_path, "t.csv", pd.DataFrame({"cod_hospital": ["A", "B"], "n_egresos": [1234, 1234], "tasa_x_pct": ["0.57", "0.6"]}))
    h = pp.reconstrucciones(tmp_path)
    assert [(x["cod_hospital"], x["conteo"]) for x in h] == [("A", 7)]      # 0,6 % deja dos enteros posibles: no fija nada


def test_canario_oe_por_esperado(tmp_path):
    _escribir(tmp_path, "oe_hospital.csv", pd.DataFrame({"cod_hospital": ["A"], "observado": ["<10"], "esperado": [20.0], "oe": [0.25]}))
    assert [x["tipo"] for x in pp.reconstrucciones(tmp_path)] == ["O/E × esperado"]


def test_canario_complemento_de_subgrupos(tmp_path):
    d = pd.DataFrame({"variable": ["edad", "edad", "sexo", "sexo", "sexo"], "grupo": ["<60", "60+", "H", "M", "S"],
                      "n": [500, 600, 540, 520, 40], "observado": [30, 50, 40, 37, None]})
    _escribir(tmp_path, "subgrupos.csv", d)
    h = pp.reconstrucciones(tmp_path)
    assert [(x["variable"], x["conteo"]) for x in h] == [("sexo", 3)]


def test_sin_fuga_no_hay_hallazgos(tmp_path):
    _escribir(tmp_path, "t.csv", pd.DataFrame({"cod_hospital": ["A", "B"], "n_egresos": [1000, 2000], "tasa_x": [0.0, 15 / 2000]}))
    _escribir(tmp_path, "s.csv", pd.DataFrame({"variable": ["edad", "edad", "sexo", "sexo"], "grupo": ["a", "b", "H", "M"],
                                               "observado": [30, 50, 40, 40]}))
    assert pp.reconstrucciones(tmp_path) == []


def test_canario_valor_real_en_una_prueba(tmp_path):
    sal = tmp_path / "salidas"
    _escribir(sal, "oe_hospital_mortalidad.csv", pd.DataFrame({"cod_hospital": ["A"], "observado": ["<10"], "esperado": [31.2345678912], "oe": [None]}))
    pruebas = tmp_path / "tests"
    pruebas.mkdir()
    (pruebas / "test_x.py").write_text("ESPERADO = 31.2345678912\n")
    r = pp.verificar([sal], [pruebas])
    assert not r["ok"] and r["valores_en_textos"][0]["origen"] == "oe_hospital_mortalidad.csv:esperado"


def test_canario_mismo_conteo_con_dos_cohortes(tmp_path):
    # I-12: el mismo conteo por hospital publicado con la cohorte sin reglas y con ellas; la resta da lo que quitó la regla
    cods = [f"1{i:05d}" for i in range(6)]
    _escribir(tmp_path / "fase2", "embudo.csv", pd.DataFrame({"periodo": ["2023"] * 6, "cod_hospital": cods, "k": [400, 380, 512, 233, 701, 99]}))
    _escribir(tmp_path / "fase3", "catalogo.csv", pd.DataFrame({"anio": ["2023"] * 6, "cod_hospital": cods, "n_fallecidos_d": [396, 380, 512, 230, 701, 99]}))
    h = pp.entre_tablas([tmp_path / "fase2", tmp_path / "fase3"])
    assert sorted(x["conteo"] for x in h) == [3, 4]
    # redondeado a la decena: la diferencia ya no fija un entero
    _escribir(tmp_path / "fase3", "catalogo.csv", pd.DataFrame({"anio": ["2023"] * 6, "cod_hospital": cods, "n_fallecidos_d": [400, 380, 510, 230, 700, 100]}))
    assert pp.entre_tablas([tmp_path / "fase2", tmp_path / "fase3"]) == []


def test_canario_misma_tabla_en_dos_versiones_del_protocolo(tmp_path):
    from evaluacion import publicacion
    cods = [f"1{i:05d}" for i in range(5)]
    v1 = pd.DataFrame({"cod_hospital": cods, "n_egresos": [1200, 5400, 830, 2210, 9001], "observado": [40, 200, 31, 75, 360], "esperado": [41.0, 190.2, 30.0, 80.1, 350.3]})
    v4 = v1.assign(n_egresos=[1195, 5400, 828, 2210, 8990], observado=[38, 199, 31, 70, 352])
    _escribir(tmp_path / "evaluacion_2024" / "tablas", "oe_hospital_mortalidad.csv", v1)
    _escribir(tmp_path / "evaluacion_2024_v4" / "tablas", "oe_hospital_mortalidad.csv", v4)
    assert len(pp.entre_tablas([tmp_path])) == 6          # 2 en egresos y 4 en muertes (la diferencia de 11 no cuenta)
    publicacion.publicar(tmp_path / "evaluacion_2024_v4", tmp_path / "trabajo")
    pub = pd.read_csv(tmp_path / "evaluacion_2024_v4" / "tablas" / "oe_hospital_mortalidad.csv")
    assert "observado" not in pub.columns and "esperado" not in pub.columns and (pub["n_egresos"] % 10 == 0).all()
    assert pp.entre_tablas([tmp_path / "evaluacion_2024", tmp_path / "evaluacion_2024_v4"]) == []
    # las completas quedan en la carpeta de trabajo y se vuelve a publicar desde ellas
    assert publicacion.leer(tmp_path / "evaluacion_2024_v4", tmp_path / "trabajo", "oe_hospital_mortalidad")["observado"].tolist() == v4["observado"].tolist()


def test_salidas_reales_sin_hallazgos_nuevos():
    """Puerta real: lo publicado por las Fases 3 a 6, las pruebas y los documentos. Los hallazgos del incidente I-11
    están declarados; cualquier otro hace fallar la prueba."""
    t = RAIZ.parent
    r = pp.verificar([RAIZ / "salidas", t / "fase3_preparacion" / "salidas", t / "fase4_modelado" / "salidas", t / "fase6_despliegue" / "datos"],
                     [RAIZ / "tests", t / "fase3_preparacion" / "tests", t / "fase4_modelado" / "tests", t / "docs"], ruta_declarados=DECLARADOS,
                     cruces=[t / "fase2_eda" / "salidas", t / "fase2_eda" / "analisis_complementario", t / "fase3_preparacion" / "salidas", RAIZ / "salidas",
                             t / "fase4_modelado" / "salidas", t / "fase6_despliegue" / "datos"])
    assert r["ok"], r["nuevos"]
    assert all(pp.declarados(DECLARADOS)[pp.clave_hallazgo(h)] == "I-11" for h in r["declarados"])
