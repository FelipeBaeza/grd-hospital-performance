"""Tarea 13, parte D: el runner (argumentos, limpieza protegida, orden de ejecución, códigos de salida) y la corrida de punta a punta sobre datos sintéticos."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from ayudas_t13 import clonar
from eda import (
    asociacion, calidad, calidad_catalogos, catalogos, cierre, cohorte, comorbilidad, descripcion, ejecucion, hospitales, informes_finales, lector, orquestador, privacidad,
    recoleccion, rutas, seleccion, verificadores,
)
from eda import panel as modulo_panel
from eda.contexto import Contexto, _limpiar_numericos
from eda.orquestador import ErrorDeProtocolo, Parametros, SalidaProtegida
from eda.verificadores import ResultadoDT
from sintetico import crear_fuente_sintetica

TAREAS = [f"T{n}" for n in range(3, 13)]


# ============================================================================================================================ argumentos
@pytest.fixture()
def archivo_falso(tmp_path):
    p = tmp_path / "GRD_PUBLICO_2019.txt"
    p.write_text("x", encoding="utf8")
    return p


def error_de_uso(argv, capsys) -> str:
    with pytest.raises(SystemExit) as e:
        orquestador.analizar_argumentos(argv)
    assert e.value.code == 2
    return capsys.readouterr().err


def test_los_valores_por_defecto_son_extracto_con_la_carpeta_de_salidas_de_la_suite():
    p = orquestador.analizar_argumentos([])
    assert p == Parametros(fuente="extracto", data_root=None, archivos={}, salida_base=rutas.DIR_SALIDAS, sin_pytest=False, sin_determinismo=False, solo=None)
    assert rutas.DIR_SALIDAS == rutas.RAIZ_EDA / "salidas"


def test_se_leen_todas_las_opciones(tmp_path, archivo_falso):
    otro = tmp_path / "GRD_PUBLICO_2024.txt"
    otro.write_text("y", encoding="utf8")
    p = orquestador.analizar_argumentos(
        ["--fuente", "completo", "--data-root", str(tmp_path), "--archivo", f"2024={otro}", "--archivo", f"2019={archivo_falso}", "--salida-base", str(tmp_path / "s"),
         "--sin-pytest", "--sin-determinismo", "--solo", "T5,T6"]
    )
    assert p.fuente == "completo" and p.data_root == tmp_path and p.archivos == {2019: archivo_falso, 2024: otro} and p.salida_base == tmp_path / "s"
    assert p.sin_pytest and p.sin_determinismo and p.solo == ("T5", "T6")


def test_con_fuente_completo_hace_falta_la_carpeta_de_datos(capsys, tmp_path):
    err = error_de_uso(["--fuente", "completo"], capsys)
    assert "error de uso: con --fuente completo hace falta --data-root" in err and err.startswith("uso: run_fase2_eda.py")
    assert "la carpeta de --data-root no existe" in error_de_uso(["--data-root", str(tmp_path / "no_existe")], capsys)


def test_las_fuentes_invalidas_se_rechazan_en_espanol(capsys):
    err = error_de_uso(["--fuente", "otra"], capsys)
    assert "argumento --fuente: debe ser extracto o completo, no 'otra'" in err


def test_el_archivo_por_anio_debe_tener_la_forma_anio_igual_ruta_con_un_anio_valido_y_un_archivo_que_existe(capsys, tmp_path, archivo_falso):
    assert "debe tener la forma ANIO=RUTA" in error_de_uso(["--archivo", str(archivo_falso)], capsys)
    assert "debe tener la forma ANIO=RUTA" in error_de_uso(["--archivo", "dosmil=x"], capsys)
    assert "debe tener la forma ANIO=RUTA" in error_de_uso(["--archivo", "2019="], capsys)
    assert "el año 2018 no está entre 2019 y 2024" in error_de_uso(["--archivo", f"2018={archivo_falso}"], capsys)
    assert "el año 2025 no está entre 2019 y 2024" in error_de_uso(["--archivo", f"2025={archivo_falso}"], capsys)
    assert "el archivo del año 2020 no existe" in error_de_uso(["--archivo", f"2020={tmp_path / 'no_existe.txt'}"], capsys)
    assert "el año 2019 aparece más de una vez en --archivo" in error_de_uso(["--archivo", f"2019={archivo_falso}", "--archivo", f"2019={archivo_falso}"], capsys)
    # el borde de los años válidos se acepta
    p = orquestador.analizar_argumentos(["--archivo", f"2019={archivo_falso}", "--archivo", f"2024={archivo_falso}"])
    assert sorted(p.archivos) == [2019, 2024]


def test_solo_acepta_t3_a_t12_en_cualquier_escritura_y_los_deja_en_orden(capsys):
    assert orquestador.normalizar_solo("t12,T3,t03, T5 ,T10,T3") == ("T3", "T5", "T10", "T12")
    assert orquestador.normalizar_solo("T3") == ("T3",) and orquestador.normalizar_solo(",".join(TAREAS)) == tuple(TAREAS)
    assert orquestador.analizar_argumentos(["--solo", "t4,T3"]).solo == ("T3", "T4")
    assert "T13 es el propio runner" in error_de_uso(["--solo", "T3,T13"], capsys)
    for malo in ("T2", "T14", "X", "3", "", "T3,,T4", "T3,T99"):
        assert "tarea desconocida" in error_de_uso(["--solo", malo], capsys), malo


def test_las_opciones_desconocidas_y_la_ayuda_estan_en_espanol(capsys):
    assert "argumentos no reconocidos: --otra" in error_de_uso(["--otra"], capsys)
    assert "se esperaba un valor" in error_de_uso(["--fuente"], capsys)
    with pytest.raises(SystemExit) as e:
        orquestador.analizar_argumentos(["--help"])
    assert e.value.code == 0
    ayuda = capsys.readouterr().out
    assert ayuda.startswith("uso: run_fase2_eda.py") and "opciones:" in ayuda and "muestra esta ayuda y termina" in ayuda and "Códigos de salida" in ayuda
    for opcion in ("--fuente", "--data-root", "--archivo", "--salida-base", "--sin-pytest", "--sin-determinismo", "--solo"):
        assert opcion in ayuda


def test_el_script_solo_orquesta_y_su_ayuda_sale_con_codigo_cero():
    p = subprocess.run([sys.executable, "-B", str(rutas.RAIZ_EDA / "run_fase2_eda.py"), "--help"], cwd=rutas.RAIZ_EDA, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and p.stdout.startswith("uso: run_fase2_eda.py")
    codigo = (rutas.RAIZ_EDA / "run_fase2_eda.py").read_text(encoding="utf8")
    assert "orquestador.main()" in codigo and "argparse" not in codigo and len(codigo.splitlines()) < 20
    malo = subprocess.run([sys.executable, "-B", str(rutas.RAIZ_EDA / "run_fase2_eda.py"), "--fuente", "completo"], cwd=rutas.RAIZ_EDA, capture_output=True, text=True, timeout=120)
    assert malo.returncode == 2 and "hace falta --data-root" in malo.stderr


# ============================================================================================================================ limpieza protegida
@pytest.fixture()
def proyecto(tmp_path, monkeypatch):
    """Un proyecto falso (raíz, suite y directorio personal en tmp_path) para probar la limpieza sin tocar el real."""
    raiz, casa = tmp_path / "proyecto", tmp_path / "casa"
    suite = raiz / "fase2_eda"
    for d in (".git", "docs", "data seminario", ".agents", "fase2_eda/config", "fase2_eda/eda", "fase2_eda/tests", "fase2_eda/.agents", "fase2_eda/salidas"):
        (raiz / d).mkdir(parents=True)
    casa.mkdir()
    monkeypatch.setattr(rutas, "RAIZ_PROYECTO", raiz)
    monkeypatch.setattr(rutas, "RAIZ_EDA", suite)
    monkeypatch.setenv("HOME", str(casa))
    return SimpleNamespace(raiz=raiz, suite=suite, casa=casa, tmp=tmp_path)


def sembrar(destino: Path, con_manifiesto: bool = True) -> dict[str, bytes]:
    """Una salida de corrida con sus tres subcarpetas y el manifiesto; devuelve ruta relativa -> contenido."""
    archivos = {"tablas/a.csv": b"a\n1\n", "figuras/F01.png": b"\x89PNG", "informes/i.md": b"# i\n", "tablas/subcarpeta/b.csv": b"b\n"}
    if con_manifiesto:
        archivos["manifiesto.json"] = b"{}"
    for rel, contenido in archivos.items():
        (destino / rel).parent.mkdir(parents=True, exist_ok=True)
        (destino / rel).write_bytes(contenido)
    return archivos


def contenido(carpeta: Path) -> dict[str, bytes]:
    return {p.relative_to(carpeta).as_posix(): p.read_bytes() for p in sorted(carpeta.rglob("*")) if p.is_file()}


def test_limpiar_vacia_la_carpeta_de_la_fuente_y_recrea_las_subcarpetas(proyecto):
    base = proyecto.suite / "salidas"
    sembrar(base / "extracto")
    r = orquestador.limpiar_salida(base, "extracto")
    assert r == base / "extracto" and contenido(r) == {}
    assert sorted(p.name for p in r.iterdir()) == ["figuras", "informes", "tablas"] and all(p.is_dir() for p in r.iterdir())


def test_limpiar_una_fuente_no_toca_a_su_hermana(proyecto):
    base = proyecto.suite / "salidas"
    hermana = sembrar(base / "completo")
    sembrar(base / "extracto")
    orquestador.limpiar_salida(base, "extracto")
    assert contenido(base / "completo") == hermana                                    # intacta, byte a byte
    extracto = sembrar(base / "extracto")
    orquestador.limpiar_salida(base, "completo")
    assert contenido(base / "completo") == {} and contenido(base / "extracto") == extracto          # y al revés


def test_limpiar_una_carpeta_que_no_existe_la_crea_con_sus_subcarpetas(proyecto):
    r = orquestador.limpiar_salida(proyecto.tmp / "otra" / "base", "completo")
    assert r == proyecto.tmp / "otra" / "base" / "completo" and sorted(p.name for p in r.iterdir()) == ["figuras", "informes", "tablas"]


def test_limpiar_no_borra_nada_si_la_carpeta_trae_archivos_ajenos(proyecto):
    base = proyecto.tmp / "salidas"
    antes = sembrar(base / "extracto")
    (base / "extracto" / "notas.txt").write_text("mis notas", encoding="utf8")
    antes["notas.txt"] = b"mis notas"
    with pytest.raises(SalidaProtegida, match="archivos ajenos a una corrida"):
        orquestador.limpiar_salida(base, "extracto")
    assert contenido(base / "extracto") == antes
    (base / "extracto" / "notas.txt").unlink()
    (base / "extracto" / "otra_carpeta").mkdir()
    with pytest.raises(SalidaProtegida, match="otra_carpeta"):
        orquestador.limpiar_salida(base, "extracto")
    assert (base / "extracto" / "tablas" / "a.csv").is_file()


def test_limpiar_rechaza_un_destino_que_es_un_enlace_simbolico_y_no_toca_su_destino(proyecto):
    real = proyecto.tmp / "real"
    antes = sembrar(real)
    base = proyecto.tmp / "salidas"
    base.mkdir()
    (base / "extracto").symlink_to(real, target_is_directory=True)
    with pytest.raises(SalidaProtegida, match="enlace simbólico"):
        orquestador.limpiar_salida(base, "extracto")
    assert contenido(real) == antes


def test_limpiar_rechaza_enlaces_simbolicos_dentro_de_la_carpeta(proyecto):
    ajena = proyecto.tmp / "ajena"
    antes = sembrar(ajena)
    base = proyecto.tmp / "salidas"
    (base / "extracto").mkdir(parents=True)
    (base / "extracto" / "tablas").symlink_to(ajena / "tablas", target_is_directory=True)
    with pytest.raises(SalidaProtegida, match="enlaces simbólicos"):
        orquestador.limpiar_salida(base, "extracto")
    assert contenido(ajena) == antes


def test_limpiar_rechaza_una_base_que_es_un_enlace_a_una_carpeta_protegida(proyecto):
    centinela = proyecto.raiz / "docs" / "extracto" / "tablas" / "c.csv"
    centinela.parent.mkdir(parents=True)
    centinela.write_text("no borrar", encoding="utf8")
    enlace = proyecto.tmp / "enlace_a_docs"
    enlace.symlink_to(proyecto.raiz / "docs", target_is_directory=True)
    with pytest.raises(SalidaProtegida, match="protegida"):
        orquestador.limpiar_salida(enlace, "extracto")
    assert centinela.read_text(encoding="utf8") == "no borrar"


def test_limpiar_rechaza_una_fuente_invalida_y_un_destino_que_es_un_archivo(proyecto):
    with pytest.raises(SalidaProtegida, match="la fuente 'otra' no es válida"):
        orquestador.limpiar_salida(proyecto.tmp / "s", "otra")
    base = proyecto.tmp / "s"
    base.mkdir()
    (base / "extracto").write_text("soy un archivo", encoding="utf8")
    with pytest.raises(SalidaProtegida, match="no es una carpeta"):
        orquestador.limpiar_salida(base, "extracto")
    assert (base / "extracto").read_text(encoding="utf8") == "soy un archivo"


def bases_protegidas(p) -> list[tuple[str, Path]]:
    return [
        ("la raíz del proyecto", p.raiz), ("la raíz de la suite", p.suite), ("un ancestro del proyecto", p.tmp), ("el directorio personal", p.casa),
        ("docs", p.raiz / "docs"), ("dentro de docs", p.raiz / "docs" / "sub"), ("data seminario", p.raiz / "data seminario"), (".git", p.raiz / ".git"),
        (".agents del proyecto", p.raiz / ".agents"), ("config", p.suite / "config"), ("eda", p.suite / "eda"), ("tests", p.suite / "tests"), (".agents de la suite", p.suite / ".agents"),
        ("dentro de .git", p.raiz / ".git" / "objetos"),
    ]


def test_limpiar_rechaza_las_bases_sensibles_y_las_carpetas_protegidas_sin_borrar_nada(proyecto):
    for nombre, base in bases_protegidas(proyecto):
        for fuente in rutas.FUENTES_VALIDAS:
            centinela = base / fuente / "tablas" / "centinela.csv"
            centinela.parent.mkdir(parents=True, exist_ok=True)
            centinela.write_text("no borrar", encoding="utf8")
            with pytest.raises(SalidaProtegida):
                orquestador.limpiar_salida(base, fuente)
            assert centinela.read_text(encoding="utf8") == "no borrar", nombre
    with pytest.raises(SalidaProtegida, match="la raíz del sistema"):
        orquestador.limpiar_salida(Path("/"), "extracto")


def test_limpiar_rechaza_cualquier_carpeta_dentro_de_un_repositorio_git_aunque_no_sea_el_del_proyecto(proyecto):
    base = proyecto.tmp / "otro_repositorio" / ".git" / "dentro"
    centinela = base / "extracto" / "tablas" / "centinela.csv"
    centinela.parent.mkdir(parents=True)
    centinela.write_text("no borrar", encoding="utf8")
    with pytest.raises(SalidaProtegida, match="dentro de un repositorio git"):
        orquestador.limpiar_salida(base, "extracto")
    assert centinela.read_text(encoding="utf8") == "no borrar"


def test_limpiar_acepta_la_carpeta_de_salidas_de_la_suite_y_cualquier_carpeta_aparte(proyecto):
    for base in (proyecto.suite / "salidas", proyecto.tmp / "fuera", proyecto.raiz / "otra_carpeta"):
        sembrar(base / "extracto")
        assert orquestador.limpiar_salida(base, "extracto") == base / "extracto"
        assert contenido(base / "extracto") == {}


# ============================================================================================================================ registros de una corrida anterior
def test_hidratar_los_registros_reproduce_los_del_contexto_que_los_escribio(pipeline_sintetico, tmp_path):
    ctx0 = pipeline_sintetico.ctx
    c = clonar(ctx0, tmp_path)
    assert c.indice_tablas().equals(ctx0.indice_tablas())
    assert c.metricas().equals(_limpiar_numericos(ctx0.metricas()))
    assert c.figuras.indice().equals(ctx0.figuras.indice())
    assert c.cortafuegos.registro().equals(ctx0.cortafuegos.registro())
    assert c.cifras_informe().equals(ctx0.cifras_informe())
    c.escribir_registros()                                                         # al escribirlos de nuevo no cambia nada
    for nombre in ("tablas_indice.csv", "metricas_eda.csv", "figuras_indice.csv", "registro_accesos.csv", "cifras_informe.csv"):
        assert (c.dir_tablas / nombre).read_bytes() == (ctx0.dir_tablas / nombre).read_bytes(), nombre


def test_hidratar_descarta_los_accesos_de_las_tareas_que_se_vuelven_a_ejecutar_y_renumera(pipeline_sintetico, tmp_path):
    ctx0 = pipeline_sintetico.ctx
    shutil.copytree(ctx0.dir_salida, tmp_path / "extracto")
    c = Contexto.crear("extracto", fuente=ctx0.fuente, dir_salida_base=tmp_path)
    orquestador.hidratar_registros(c, ["T5", "T10"])
    reg0, reg = ctx0.cortafuegos.registro(), c.cortafuegos.registro()
    esperado = reg0.filter(~pl.col("tarea").is_in(["T05", "T10"]))
    assert reg0.filter(pl.col("tarea") == "T05").height > 0 and reg0.filter(pl.col("tarea") == "T10").height > 0
    assert reg.height == esperado.height and reg["secuencia"].to_list() == list(range(1, esperado.height + 1))
    assert reg.drop("secuencia").equals(esperado.drop("secuencia"))
    # una tarea nueva sigue la numeración
    c.cortafuegos.autorizar(2019, ["COD_HOSPITAL"], "estructura", tarea="T05")
    assert c.cortafuegos.registro()["secuencia"][-1] == esperado.height + 1


def test_hidratar_una_carpeta_vacia_no_carga_nada(fuente_sintetica, tmp_path):
    c = Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path)
    orquestador.hidratar_registros(c, ["T3"])
    assert c.indice_tablas().height == 0 and c.metricas().height == 0 and c.figuras.indice().height == 0 and c.cortafuegos.registro().height == 0 and c.cifras_informe().height == 0


def test_los_espacios_de_identificadores_se_leen_de_la_tarea_5_o_de_su_tabla_publicada(fuente_sintetica, tmp_path):
    r5 = {"regimenes": pl.DataFrame({"anios": ["2019, 2020", "2021, 2022, 2023, 2024"]})}
    c = Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path)
    assert orquestador._regimenes_de_t5(c, r5) == [[2019, 2020], [2021, 2022, 2023, 2024]]
    assert orquestador._regimenes_de_t5(c, None) is None                            # sin la tarea ni su tabla, la Tarea 9 los calcula sola
    c.escribir_tabla("t24_id_regimenes.csv", pl.DataFrame({"regimen": [1], "anios": ["2019, 2020, 2021"]}), tarea=5, descripcion="x")
    assert orquestador._regimenes_de_t5(c, None) == [[2019, 2020, 2021]]
    assert orquestador._regimenes_de_t5(c, r5) == [[2019, 2020], [2021, 2022, 2023, 2024]]       # la tarea recién ejecutada manda


# ============================================================================================================================ orden de ejecución con tareas falsas
RESULTADOS_PASS = [ResultadoDT(f"DT{i:02d}", f"criterio {i}", "PASS", "bien", "x") for i in range(1, 13)]


@pytest.fixture()
def falsas(monkeypatch):
    """Reemplaza cada tarea, el panel, los informes, los verificadores y el manifiesto por funciones que solo anotan su llamada."""
    f = SimpleNamespace(llamadas=[], args={}, resultados=list(RESULTADOS_PASS), sello=list(RESULTADOS_PASS[1:2] + RESULTADOS_PASS[3:4] + RESULTADOS_PASS[7:8]))

    def falsa(nombre, retorno=None):
        def llamar(*a, **k):
            f.llamadas.append(nombre)
            f.args.setdefault(nombre, []).append((a, k))
            return retorno(*a, **k) if callable(retorno) else retorno
        return llamar

    def verificar(ctx, banderas=None, *, ids=None):
        f.llamadas.append("sello" if ids is not None else "verificar")
        f.args.setdefault("sello" if ids is not None else "verificar", []).append(((ctx, banderas), {"ids": ids}))
        return f.sello if ids is not None else f.resultados

    for modulo, nombre, etiqueta, retorno in (
        (catalogos, "cargar_catalogos", "catalogos", "CATALOGOS"), (recoleccion, "ejecutar", "T3", None), (descripcion, "ejecutar", "T4", None),
        (calidad, "ejecutar", "T5", {"regimenes": pl.DataFrame({"anios": ["2019, 2020", "2021, 2022, 2023, 2024"]})}), (calidad_catalogos, "ejecutar", "T6", None),
        (cohorte, "ejecutar", "T7", {"cascada": "CASCADA", "cascada_pipeline": "PIPELINE"}), (comorbilidad, "ejecutar", "T8", None),
        (hospitales, "ejecutar", "T9", {"regimenes": [[2019, 2020]]}), (modulo_panel, "construir_panel", "panel", "PANEL"), (asociacion, "ejecutar", "T10", {"ASOC": 1}),
        (seleccion, "ejecutar", "T11", None), (cierre, "ejecutar", "T12", None), (informes_finales, "componer_informes", "informes", []),
        (informes_finales, "componer_cierre", "cierre", None), (ejecucion, "escribir_manifiesto_corrida", "manifiesto", None),
    ):
        monkeypatch.setattr(modulo, nombre, falsa(etiqueta, retorno))
    monkeypatch.setattr(verificadores, "verificar_todo", verificar)
    return f


def parametros_de(tmp_path, datos, **cambios) -> Parametros:
    base = dict(fuente="extracto", data_root=datos.raiz, archivos=dict(datos.archivos), salida_base=tmp_path / "salidas", sin_pytest=True, sin_determinismo=True)
    return Parametros(**{**base, **cambios})


def test_el_orden_de_ejecucion_es_el_del_encargo_y_el_panel_se_construye_una_sola_vez(falsas, fuente_sintetica, tmp_path):
    res = orquestador.ejecutar_corrida(parametros_de(tmp_path, fuente_sintetica), salida=io.StringIO())
    assert falsas.llamadas == ["catalogos", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "panel", "T10", "T11", "T12", "informes", "verificar", "cierre", "manifiesto", "sello"]
    assert falsas.llamadas.count("panel") == 1 and falsas.llamadas.index("T9") < falsas.llamadas.index("panel") < falsas.llamadas.index("T10")
    assert res.codigo == 0 and not res.parcial and res.tareas == tuple(TAREAS) and [n for n, _ in res.pasos][0] == "catálogos"


def test_cada_tarea_recibe_lo_que_produjo_la_anterior(falsas, fuente_sintetica, tmp_path):
    orquestador.ejecutar_corrida(parametros_de(tmp_path, fuente_sintetica), salida=io.StringIO())
    a = falsas.args
    # los espacios de identificadores de la Tarea 5 (texto con comas) llegan a la Tarea 9 como listas de años
    assert a["T9"][0][1]["regimenes"] == [[2019, 2020], [2021, 2022, 2023, 2024]] and a["T9"][0][0][1] == "CATALOGOS"
    # con extractos el panel no recibe espacios (no calcula historial); es el único punto de lectura de las Tareas 10 y 11 y se etiqueta T10
    assert a["panel"][0][1] == {"regimenes": None, "tarea": "T10"} and a["panel"][0][0][1] == "CATALOGOS"
    # la Tarea 10 recibe el panel; la 11, el panel y lo que calculó la 10
    assert a["T10"][0][0][1] == "PANEL" and a["T11"][0][0][1:3] == ("PANEL", {"ASOC": 1})
    # la Tarea 12 recibe la cascada sin suprimir de la Tarea 7, igual que el verificador DT05
    assert a["T12"][0][1] == {"cascada": "CASCADA"}
    banderas = a["verificar"][0][0][1]
    assert banderas.cascada == "CASCADA" and banderas.cascada_pipeline == "PIPELINE" and banderas.sin_pytest and banderas.sin_determinismo
    assert banderas.informes_esperados == informes_finales.informes_esperados()[:-1]               # antes del cierre faltan t26
    assert a["cierre"][0][0][1] == falsas.resultados and a["sello"][0][1] == {"ids": orquestador.SELLO}
    assert a["informes"][0][0][1] is None                                                           # corrida completa: todos los informes


def test_con_datos_completos_el_panel_recibe_los_espacios_de_la_tarea_9(falsas, fuente_sintetica, tmp_path):
    p = parametros_de(tmp_path, fuente_sintetica, fuente="completo")
    orquestador.ejecutar_corrida(p, salida=io.StringIO())
    assert falsas.args["panel"][0][1] == {"regimenes": [[2019, 2020]], "tarea": "T10"}


def test_el_sello_reevalua_dt02_dt04_y_dt08_con_los_siete_informes(falsas, fuente_sintetica, tmp_path):
    orquestador.ejecutar_corrida(parametros_de(tmp_path, fuente_sintetica), salida=io.StringIO())
    assert orquestador.SELLO == ("DT02", "DT04", "DT08")
    banderas_sello = falsas.args["sello"][0][0][1]
    assert banderas_sello.informes_esperados == informes_finales.informes_esperados() and len(banderas_sello.informes_esperados) == 7


def test_el_codigo_de_salida_es_cero_salvo_que_un_criterio_o_el_sello_falle(falsas, fuente_sintetica, tmp_path):
    pendiente = ResultadoDT("DT12", "c", "PENDIENTE", "d", "e")
    falsas.resultados = RESULTADOS_PASS[:-1] + [pendiente]
    assert orquestador.ejecutar_corrida(parametros_de(tmp_path, fuente_sintetica), salida=io.StringIO()).codigo == 0         # PENDIENTE no es un fallo
    falla = ResultadoDT("DT05", "c", "FAIL", "d", "e")
    falsas.resultados = RESULTADOS_PASS[:4] + [falla] + RESULTADOS_PASS[5:]
    res = orquestador.ejecutar_corrida(parametros_de(tmp_path, fuente_sintetica), salida=io.StringIO())
    assert res.codigo == 1 and [r.id for r in res.resultados if r.estado == "FAIL"] == ["DT05"]
    falsas.resultados = list(RESULTADOS_PASS)
    falsas.sello = [ResultadoDT("DT04", "c", "FAIL", "d", "e")]
    assert orquestador.ejecutar_corrida(parametros_de(tmp_path, fuente_sintetica), salida=io.StringIO()).codigo == 1             # el sello también cuenta


def test_solo_ejecuta_las_tareas_pedidas_sin_limpiar_ni_verificar_ni_escribir_el_manifiesto(falsas, fuente_sintetica, tmp_path):
    base = tmp_path / "salidas"
    antes = sembrar(base / "extracto", con_manifiesto=False)
    (base / "extracto" / "notas.txt").write_text("ajeno", encoding="utf8")                     # con una carpeta limpiable esto la bloquearía
    res = orquestador.ejecutar_corrida(parametros_de(tmp_path, fuente_sintetica, solo=("T5", "T6")), salida=io.StringIO())
    assert falsas.llamadas == ["catalogos", "T5", "T6", "informes"] and res.parcial and res.codigo == 0 and res.resultados == []
    assert falsas.args["informes"][0][0][1] == ("T5", "T6")
    assert (base / "extracto" / "notas.txt").read_text(encoding="utf8") == "ajeno"
    assert all((base / "extracto" / rel).read_bytes() == c for rel, c in antes.items())        # nada se borró
    assert not (base / "extracto" / "manifiesto.json").exists()


def test_solo_construye_el_panel_si_se_pide_la_tarea_10_o_la_11_y_no_otra_cosa(falsas, fuente_sintetica, tmp_path):
    for solo, esperado in ((("T7",), ["catalogos", "T7", "informes"]), (("T10",), ["catalogos", "panel", "T10", "informes"]),
                           (("T11",), ["catalogos", "panel", "T11", "informes"]), (("T10", "T11"), ["catalogos", "panel", "T10", "T11", "informes"]),
                           (("T12",), ["catalogos", "T12", "informes"]), (("T3", "T4"), ["catalogos", "T3", "T4", "informes"])):
        falsas.llamadas.clear()
        orquestador.ejecutar_corrida(parametros_de(tmp_path, fuente_sintetica, solo=solo), salida=io.StringIO())
        assert falsas.llamadas == esperado, solo


def test_solo_t12_no_recibe_la_cascada_y_lee_lo_demas_de_disco(falsas, fuente_sintetica, tmp_path):
    orquestador.ejecutar_corrida(parametros_de(tmp_path, fuente_sintetica, solo=("T12",)), salida=io.StringIO())
    assert falsas.args["T12"][0][1] == {"cascada": None}


def test_un_protocolo_invalido_detiene_la_corrida_antes_de_limpiar_nada(falsas, fuente_sintetica, tmp_path, monkeypatch):
    base = tmp_path / "salidas"
    antes = sembrar(base / "extracto")
    monkeypatch.setattr(orquestador, "validar_protocolo", lambda: ["umbrales: falta la justificación de x"])
    with pytest.raises(ErrorDeProtocolo, match="el protocolo no es válido: umbrales: falta la justificación de x"):
        orquestador.ejecutar_corrida(parametros_de(tmp_path, fuente_sintetica), salida=io.StringIO())
    assert falsas.llamadas == [] and contenido(base / "extracto") == antes


def test_el_protocolo_y_las_cifras_documentadas_reales_son_validos():
    assert orquestador.validar_protocolo() == []


def test_si_faltan_los_datos_la_corrida_no_limpia_la_salida_anterior(falsas, fuente_sintetica, tmp_path):
    base = tmp_path / "salidas"
    antes = sembrar(base / "extracto")
    p = parametros_de(tmp_path, fuente_sintetica, data_root=tmp_path / "no_hay_datos", archivos={})
    with pytest.raises(lector.ArchivoNoEncontrado):
        orquestador.ejecutar_corrida(p, salida=io.StringIO())
    assert falsas.llamadas == [] and contenido(base / "extracto") == antes


def test_main_devuelve_dos_en_los_errores_de_protocolo_de_salida_y_de_datos(falsas, fuente_sintetica, tmp_path, monkeypatch, capsys):
    args = ["--data-root", str(fuente_sintetica.raiz), "--salida-base", str(tmp_path / "s"), "--sin-pytest", "--sin-determinismo"]
    assert orquestador.main(args) == 0
    original = orquestador.validar_protocolo
    monkeypatch.setattr(orquestador, "validar_protocolo", lambda: ["x"])
    assert orquestador.main(args) == 2 and "error: el protocolo no es válido: x" in capsys.readouterr().err
    monkeypatch.setattr(orquestador, "validar_protocolo", original)
    sembrar(tmp_path / "s2" / "extracto")
    (tmp_path / "s2" / "extracto" / "ajeno.txt").write_text("a", encoding="utf8")
    assert orquestador.main(["--data-root", str(fuente_sintetica.raiz), "--salida-base", str(tmp_path / "s2")]) == 2
    assert "archivos ajenos" in capsys.readouterr().err
    assert orquestador.main(["--data-root", str(tmp_path), "--salida-base", str(tmp_path / "s3")]) == 2
    assert "faltan archivos" in capsys.readouterr().err


# ============================================================================================================================ el lanzador de DT11 y el resumen
def test_el_lanzador_de_la_corrida_determinista_corre_el_mismo_runner_con_otra_carpeta_y_sin_fijar_la_semilla_de_hash(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    llamadas = []

    def falso(cmd, **kw):
        llamadas.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr="Traceback\nboom\n")

    monkeypatch.setattr(subprocess, "run", falso)
    a, b = tmp_path / "2019.txt", tmp_path / "2020.txt"
    p = Parametros(fuente="completo", data_root=tmp_path, archivos={2020: b, 2019: a}, salida_base=tmp_path / "ignorada", sin_pytest=False, sin_determinismo=False)
    codigo = orquestador.lanzador_corrida(p)(tmp_path / "temporal")
    (cmd, kw), = llamadas
    assert codigo == 3
    assert cmd == [sys.executable, "-B", str(rutas.RAIZ_EDA / "run_fase2_eda.py"), "--fuente", "completo", "--salida-base", str(tmp_path / "temporal"), "--sin-pytest",
                   "--sin-determinismo", "--data-root", str(tmp_path), "--archivo", f"2019={a}", "--archivo", f"2020={b}"]
    assert kw["cwd"] == rutas.RAIZ_EDA and kw["env"]["PYTHONDONTWRITEBYTECODE"] == "1" and "PYTHONHASHSEED" not in kw["env"]
    assert "boom" in capsys.readouterr().err                                      # un código inesperado deja el rastro en la consola
    sin_datos = Parametros(fuente="extracto", archivos={}, salida_base=tmp_path)
    llamadas.clear()
    orquestador.lanzador_corrida(sin_datos, script=tmp_path / "otro.py")(tmp_path / "t")
    assert llamadas[0][0] == [sys.executable, "-B", str(tmp_path / "otro.py"), "--fuente", "extracto", "--salida-base", str(tmp_path / "t"), "--sin-pytest", "--sin-determinismo"]


def test_el_resumen_de_una_corrida_completa_lista_los_criterios_el_sello_y_la_linea_final():
    res = orquestador.ResultadoCorrida(
        codigo=0, parametros=Parametros(), pasos=[("T3 recolección inicial", 1.25), ("informes", 0.5)], tareas=tuple(TAREAS),
        resultados=[ResultadoDT("DT01", "Criterio uno", "PASS", "bien", "x"), ResultadoDT("DT10", "Pruebas", "PENDIENTE", "no se ejecutó", "x"), ResultadoDT("DT05", "Cascada", "FAIL", "no cuadra", "x")],
        sello=[ResultadoDT("DT02", "c", "PASS", "d", "e"), ResultadoDT("DT04", "c", "PASS", "d", "e")],
    )
    salida = io.StringIO()
    orquestador.imprimir_resumen(res, salida)
    texto = salida.getvalue()
    assert "DT01 PASS       Criterio uno. bien" in texto and "DT10 PENDIENTE  Pruebas. no se ejecutó" in texto and "DT05 FAIL       Cascada. no cuadra" in texto
    assert "Sello final sobre el árbol de salidas: DT02, DT04 sin fallos." in texto and "Tiempo total: 1,8 s" in texto
    assert texto.rstrip().endswith("Resultado: 1 PASS, 1 PENDIENTE, 1 FAIL")
    res.sello = [ResultadoDT("DT04", "Privacidad", "FAIL", "una fecha", "x")]
    salida = io.StringIO()
    orquestador.imprimir_resumen(res, salida)
    assert "El sello final sobre el árbol de salidas encontró fallos:" in salida.getvalue() and "DT04 FAIL       Privacidad. una fecha" in salida.getvalue()


def test_el_resumen_de_una_corrida_parcial_dice_que_no_verifico_nada():
    res = orquestador.ResultadoCorrida(codigo=0, parametros=Parametros(solo=("T5",)), pasos=[("T5 calidad", 2.0)], tareas=("T5",), parcial=True)
    salida = io.StringIO()
    orquestador.imprimir_resumen(res, salida)
    t = salida.getvalue()
    assert t.startswith("\nCorrida parcial (T5): no se limpió la carpeta, no se corrieron los criterios de terminado ni se escribió el manifiesto.") and "Resultado:" not in t


# ============================================================================================================================ punta a punta con datos sintéticos
@pytest.fixture(scope="module")
def datos_sinteticos(tmp_path_factory):
    """La fuente sintética de las pruebas de tubería (700 filas por año, semilla 29) en una carpeta propia."""
    return crear_fuente_sintetica(tmp_path_factory.mktemp("datos_t13"), n=700, semilla=29)


def argumentos_de(fuente, base, *extra) -> list[str]:
    out = ["--data-root", str(fuente.raiz), "--salida-base", str(base)]
    for anio, ruta in sorted(fuente.archivos.items()):
        out += ["--archivo", f"{anio}={ruta}"]
    return out + list(extra)


@pytest.fixture(scope="module")
def extracto(datos_sinteticos, tmp_path_factory):
    """Una corrida completa del runner sobre la fuente sintética, con la salida en otra carpeta, y lo que imprimió."""
    base = tmp_path_factory.mktemp("salidas_extracto")
    p = orquestador.analizar_argumentos(argumentos_de(datos_sinteticos, base, "--sin-pytest", "--sin-determinismo"))
    # una tabla de una versión anterior del código: el runner debe limpiarla
    (base / "extracto" / "tablas").mkdir(parents=True)
    (base / "extracto" / "tablas" / "t99_resto_de_otra_version.csv").write_text("x\n1\n", encoding="utf8")
    salida = io.StringIO()
    res = orquestador.ejecutar_corrida(p, salida=salida)
    orquestador.imprimir_resumen(res, salida)                                  # lo que imprime main al terminar
    return SimpleNamespace(base=base, dir=base / "extracto", res=res, salida=salida.getvalue(), params=p)


def test_la_corrida_sintetica_termina_en_cero_con_nueve_criterios_en_pass_y_tres_pendientes(extracto):
    res = extracto.res
    assert res.codigo == 0 and not res.parcial
    assert [r.id for r in res.resultados] == [f"DT{i:02d}" for i in range(1, 13)]
    assert [r.estado for r in res.resultados] == ["PASS"] * 9 + ["PENDIENTE"] * 3
    assert {r.id: r.estado for r in res.sello} == {"DT02": "PASS", "DT04": "PASS", "DT08": "PASS"}
    assert "Resultado: 9 PASS, 3 PENDIENTE, 0 FAIL" in extracto.salida


def test_la_salida_trae_las_33_figuras_los_siete_informes_y_nada_de_una_corrida_anterior(extracto):
    d = extracto.dir
    assert sorted(p.name for p in d.iterdir()) == ["figuras", "informes", "manifiesto.json", "tablas"]
    assert len(list((d / "figuras").glob("*.png"))) == 33
    assert sorted(p.name for p in (d / "informes").glob("*.md")) == sorted(informes_finales.informes_esperados())
    assert not (d / "tablas" / "t99_resto_de_otra_version.csv").exists()
    plan = pl.read_csv(rutas.DIR_CONFIG / "plan_figuras.csv")
    assert sorted(p.stem for p in (d / "figuras").glob("*.png")) == sorted(plan["id_figura"].to_list())


def test_el_runner_no_escribe_fuera_de_la_carpeta_de_la_fuente(extracto):
    assert [p.name for p in extracto.base.iterdir()] == ["extracto"]


def test_la_salida_cumple_privacidad_cortafuegos_y_no_tiene_filas_de_las_tareas_12_y_13(extracto):
    d = extracto.dir
    indice = pl.read_csv(d / "tablas" / "tablas_indice.csv", infer_schema=False)
    esc = privacidad.escanear_salidas(d, indice, 10)
    assert privacidad.total_violaciones(esc) == 0 and "manifiesto.json" in esc["archivo"].to_list()
    accesos = pl.read_csv(d / "tablas" / "registro_accesos.csv", infer_schema_length=1000, schema_overrides={"tarea": pl.String})
    assert accesos.filter(pl.col("tarea").is_in(["T12", "T13"])).height == 0
    assert accesos.filter((pl.col("anio") == 2024) & pl.col("tarea").is_in(["T10", "T11"])).height == 0
    manifiesto = (d / "manifiesto.json").read_text(encoding="utf8")
    assert str(extracto.base) not in manifiesto and '"bloqueos": 1' in manifiesto and '"violaciones_de_politica": 0' in manifiesto
    assert '"tareas": "T3 a T12"' in manifiesto and '"fail": 0' in manifiesto and '"pass": 9' in manifiesto and '"pendiente": 3' in manifiesto


def test_la_tabla_de_verificacion_y_el_cierre_de_la_corrida_sintetica_son_coherentes(extracto):
    d = extracto.dir
    v = pl.read_csv(d / "tablas" / "verificacion_dt.csv", infer_schema=False)
    assert v.columns == ["id", "criterio", "estado", "detalle", "evidencia"] and v.height == 12
    assert v["estado"].to_list() == ["PASS"] * 9 + ["PENDIENTE"] * 3
    resumen = pl.read_csv(d / "tablas" / "t26_resumen_cierre.csv")
    assert resumen.row(0, named=True)["n_pass"] == 9 and resumen.row(0, named=True)["n_figuras"] == 33 and resumen.row(0, named=True)["n_informes"] == 7
    md = (d / "informes" / "t26_cierre.md").read_text(encoding="utf8")
    assert informes_finales.numeracion_secciones(md) == [1, 2, 3, 4] and "9 pasan, 0 fallan y 3 quedan pendientes" in md


def test_la_consola_muestra_los_pasos_con_su_tiempo_los_criterios_y_el_resultado(extracto):
    s = extracto.salida
    for paso in ("catálogos", "T3 recolección inicial", "T9 hospitales", "panel de las Tareas 10 y 11", "T12 evidencia para decidir", "informes", "criterios de terminado", "manifiesto", "sello"):
        assert f"  {paso}: " in s, paso
    assert s.index("T8 comorbilidad") < s.index("T9 hospitales") < s.index("panel de las Tareas 10 y 11") < s.index("T10 asociación") < s.index("T12 evidencia")
    assert "DT10 PENDIENTE" in s and "no se ejecutó: la corrida usó --sin-pytest" in s and "Sello final sobre el árbol de salidas: DT02, DT04, DT08 sin fallos." in s


def test_volver_a_correr_sobre_la_misma_carpeta_deja_la_misma_salida(extracto, datos_sinteticos):
    antes = verificadores.huellas_arbol(extracto.dir)
    res = orquestador.ejecutar_corrida(extracto.params, salida=io.StringIO())
    assert res.codigo == 0 and verificadores.huellas_arbol(extracto.dir) == antes


def test_solo_fusiona_sus_registros_con_los_de_la_corrida_anterior(extracto, datos_sinteticos, tmp_path):
    copia = tmp_path / "salidas"
    shutil.copytree(extracto.base, copia)
    d = copia / "extracto"
    (d / "tablas" / "notas_mias.csv").write_text("x\n1\n", encoding="utf8")                  # un archivo ajeno: --solo no limpia, no lo toca
    antes = {rel: h for rel, h in verificadores.huellas_arbol(d).items()}
    res = orquestador.ejecutar_corrida(orquestador.analizar_argumentos(argumentos_de(datos_sinteticos, copia, "--solo", "T5,T6")), salida=io.StringIO())
    assert res.parcial and res.codigo == 0 and res.tareas == ("T5", "T6") and res.resultados == []
    despues = verificadores.huellas_arbol(d)
    assert despues["manifiesto.json"] == antes["manifiesto.json"] and (d / "tablas" / "notas_mias.csv").read_text(encoding="utf8") == "x\n1\n"
    # la tarea repetida produce lo mismo y el informe de calidad se vuelve a componer igual; lo de las demás tareas sigue ahí
    for rel in ("tablas/t24_reglas.csv", "tablas/t24_resumen_catalogos.csv", "informes/t24_informe.md", "tablas/t23_cascada.csv", "informes/t25_evidencia_decisiones.md"):
        assert despues[rel] == antes[rel], rel
    for rel in ("tablas/metricas_eda.csv", "tablas/tablas_indice.csv", "tablas/figuras_indice.csv"):
        assert despues[rel] == antes[rel], rel                                              # la fusión reproduce los registros completos
    # las filas de lo que se repitió quedan al final del registro (pasaron después): mismo contenido, otro orden
    ordenar = lambda csv, quitar: sorted(pl.read_csv(d / "tablas" / csv, infer_schema=False).drop(quitar).rows())      # noqa: E731
    assert ordenar("cifras_informe.csv", []) == sorted(pl.read_csv(extracto.dir / "tablas" / "cifras_informe.csv", infer_schema=False).rows())
    assert ordenar("registro_accesos.csv", ["secuencia"]) == sorted(pl.read_csv(extracto.dir / "tablas" / "registro_accesos.csv", infer_schema=False).drop("secuencia").rows())
    accesos = pl.read_csv(d / "tablas" / "registro_accesos.csv", infer_schema=False, schema_overrides={"secuencia": pl.Int64})
    assert accesos["secuencia"].to_list() == list(range(1, accesos.height + 1)) and accesos["tarea"].to_list()[-1] == "T06"


def test_una_corrida_en_modo_completo_deja_el_criterio_de_datos_completos_en_pass(datos_sinteticos, tmp_path):
    base = tmp_path / "salidas"
    p = orquestador.analizar_argumentos(["--fuente", "completo", *argumentos_de(datos_sinteticos, base, "--sin-pytest", "--sin-determinismo")])
    res = orquestador.ejecutar_corrida(p, salida=io.StringIO())
    estados = {r.id: r.estado for r in res.resultados}
    assert res.codigo == 0 and estados["DT12"] == "PASS" and estados["DT10"] == "PENDIENTE" and estados["DT11"] == "PENDIENTE"
    assert [p.name for p in base.iterdir()] == ["completo"] and all(estados[f"DT{i:02d}"] == "PASS" for i in range(1, 10))
    tablas = pl.read_csv(base / "completo" / "tablas" / "tablas_indice.csv", infer_schema=False)
    assert set(tablas["fuente"].to_list()) == {"completo"}
    assert (base / "completo" / "informes" / "t23_informe.md").read_text(encoding="utf8").splitlines()[2].startswith("**Fuente de los datos: completo.**")


# ============================================================================================================================ determinismo entre procesos
def test_el_runner_como_script_repite_la_corrida_en_otros_procesos_y_los_hashes_coinciden(datos_sinteticos, tmp_path):
    """La ruta real de DT11: el script lanza dos subprocesos del propio runner con otra semilla de hash de Python y compara el SHA-256 de todos sus archivos."""
    base = tmp_path / "salidas"
    temporales = tmp_path / "temporales"                                   # los subprocesos de DT11 crean sus carpetas aquí y no en el /tmp compartido
    temporales.mkdir()
    cmd = [sys.executable, "-B", str(rutas.RAIZ_EDA / "run_fase2_eda.py"), "--fuente", "extracto", *argumentos_de(datos_sinteticos, base, "--sin-pytest")]
    entorno = {k: v for k, v in os.environ.items() if k not in ("PYTEST_CURRENT_TEST", "PYTHONHASHSEED")}
    entorno["PYTHONDONTWRITEBYTECODE"] = "1"
    entorno["TMPDIR"] = str(temporales)
    p = subprocess.run(cmd, cwd=rutas.RAIZ_EDA, env=entorno, capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    v = pl.read_csv(base / "extracto" / "tablas" / "verificacion_dt.csv", infer_schema=False)
    estados = dict(zip(v["id"].to_list(), v["estado"].to_list()))
    assert estados["DT11"] == "PASS" and estados["DT10"] == "PENDIENTE" and estados["DT12"] == "PENDIENTE"
    assert "dos corridas completas dieron los mismos" in v.filter(pl.col("id") == "DT11")["detalle"][0]
    assert "Resultado: 10 PASS, 2 PENDIENTE, 0 FAIL" in p.stdout
    assert [q.name for q in temporales.glob("fase2_det*")] == []                                    # las carpetas temporales de DT11 se borraron


# FIN-test_t13_runner
