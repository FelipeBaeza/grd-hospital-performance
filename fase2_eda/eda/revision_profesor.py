"""Análisis que pidió el profesor guía en su revisión técnica de la Fase 1 (24-10-2026). Cambio 1.6 del protocolo.

Son análisis **posteriores**: se definieron después de ver los resultados de la Fase 2 y después de abrir 2024 (19-10-2026). Son descriptivos y no
cambian ninguna decisión ya ejecutada. Siguen las mismas reglas que el resto de la suite:

- el desenlace (tipo de alta y fecha de alta) solo se lee en 2019 a 2023; 2024 se lee solo con propósito ``estructura``;
- ninguna celda de 1 a 9 casos se publica (``Contexto.escribir_tabla`` suprime y anula las columnas derivadas);
- los grupos pequeños se juntan antes de publicar, para que una celda suprimida no se pueda reconstruir restando (institutos y pediátricos van juntos).

Análisis (tablas ``r*`` y figuras ``R*``):

- R1 (observación O8): mortalidad de la cohorte D y estadía de la cohorte E por año, en total, en los capítulos CIE-10 y en los códigos de 3 caracteres
  más frecuentes, con la razón entre 2020–2021 y el promedio de 2019, 2022 y 2023.
- R2 (O2): faltantes de las variables candidatas del modelo, por variable y año, en la cohorte C (2019 a 2024).
- R3 (O3): vocabulario del diagnóstico principal: códigos con menos de 100 egresos en 2019–2022 y códigos que aparecen en 2023 o 2024 sin haber
  estado en 2019–2022 (cohorte C).
- R4 (O7): traslados entrantes, salientes y hospitalización domiciliaria según el tipo de hospital del catálogo DEIS 2025, en la cohorte C de 2019–2023,
  y la mortalidad de los entrantes en la cohorte D.
- R5 (decisión 6): hospitales con al menos 500 egresos en las cohortes D y E, por año.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import catalogos, cohorte, elixhauser, esquema, graficos, rutas
from . import transformaciones as tr
from .contexto import Contexto

TAREA = "R"
MIN_FILAS_NIVEL = 100            # regla de la Fase 3 (umbrales_fase3.toml, min_filas_nivel): un código con menos egresos en 2019–2022 pasa a RARO
VOLUMEN_MINIMO = 500             # decisión 6
N_CAPITULOS = 10
N_CODIGOS = 15
ANIOS_PANDEMIA = (2020, 2021)
ANIOS_REFERENCIA = (2019, 2022, 2023)
RUTA_DEIS = rutas.RAIZ_PROYECTO / "fase3_preparacion" / "config" / "tipo_establecimiento.csv"
GRUPOS_DEIS = {
    "general de alta complejidad": "General de alta complejidad",
    "general de mediana complejidad": "General de mediana complejidad",
    "instituto monográfico": "Institutos y pediátricos",
    "pediátrico": "Institutos y pediátricos",
}
ORDEN_GRUPOS = ("General de alta complejidad", "General de mediana complejidad", "Institutos y pediátricos")
ETIQUETAS_PROPIAS = {"U07": "COVID-19 (U07)"}
CAPITULOS = {
    "I": "Infecciosas y parasitarias", "II": "Tumores", "III": "Sangre e inmunidad", "IV": "Endocrinas y metabólicas", "V": "Trastornos mentales",
    "VI": "Sistema nervioso", "VII": "Ojo", "VIII": "Oído", "IX": "Circulatorio", "X": "Respiratorio", "XI": "Digestivo", "XII": "Piel",
    "XIII": "Osteomuscular", "XIV": "Genitourinario", "XV": "Embarazo, parto y puerperio", "XVI": "Perinatal", "XVII": "Malformaciones congénitas",
    "XVIII": "Síntomas y hallazgos anormales", "XIX": "Traumatismos y envenenamientos", "XX": "Causas externas", "XXI": "Factores que influyen en la salud",
    "XXII": "Códigos especiales (COVID-19)",
}
VARIABLES_FALTANTES = (
    ("SEXO", "Sexo", "C"),
    ("TIPO_INGRESO", "Tipo de ingreso", "C"),
    ("TIPO_PROCEDENCIA", "Procedencia", "C"),
    ("SERVICIOINGRESO", "Servicio de ingreso (nivel de cuidado)", "C"),
    ("DIAGNOSTICO1", "Diagnóstico principal", "C"),
    ("FECHA_INGRESO", "Fecha de ingreso (fin de semana)", "todos"),
    ("EDAD", "Edad", "todos"),
)
SALIENTE = ("DERIV_HOSP_SERVICIO", "DERIV_HOSP_RED")


# ------------------------------------------------------------------------------------------------------------------------------- lectura
def _falta(col: str) -> pl.Expr:
    c = pl.col(col)
    return (c.is_null() | (c.str.strip_chars() == "") | esquema.es_centinela(c)).fill_null(True)


def leer_anio(ctx: Contexto, anio: int, cat: catalogos.Catalogos, e, e3) -> pl.DataFrame:
    """Base tipada de un año con los indicadores de cohorte, el diagnóstico de 3 caracteres y las marcas de faltante."""
    con_res = anio in set(ctx.umbrales.particion["anios_resultado"])
    cols = list(cohorte.COLS_BASE) + ["COD_HOSPITAL", "SERVICIOINGRESO"] + (list(cohorte.COLS_RESULTADO) if con_res else [])
    df = ctx.lector.leer(anio, cols, proposito="resultado" if con_res else "estructura", tarea=TAREA)
    extra = df.select(
        esquema.normalizar_cie10(pl.col("DIAGNOSTICO1")).str.slice(0, 3).alias("cat3"),
        (~pl.col("SEXO").str.strip_chars().is_in(["MUJER", "HOMBRE"])).fill_null(True).alias("falta_SEXO"),
        _falta("TIPO_INGRESO").alias("falta_TIPO_INGRESO"),
        _falta("TIPO_PROCEDENCIA").alias("falta_TIPO_PROCEDENCIA"),
        _falta("SERVICIOINGRESO").alias("falta_SERVICIOINGRESO"),
        _falta("DIAGNOSTICO1").alias("falta_DIAGNOSTICO1"),
        tr.a_fecha(pl.col("FECHA_INGRESO")).is_null().alias("falta_FECHA_INGRESO"),
    )
    base = cohorte.tipar(df, anio, cat, ctx.umbrales).hstack(extra)
    del df
    dx = ctx.lector.leer(anio, list(esquema.DIAG_SEC), proposito="estructura", tarea=TAREA)
    base = cohorte.agregar_comorbilidad(base, dx, e, e3)
    del dx
    base = cohorte.marcar(base, int(ctx.umbrales["edad_adulto"]))
    base = base.with_columns(pl.col("edad").is_null().alias("falta_EDAD"))
    if con_res:
        codigo = pl.col("etiqueta_roles").replace_strict(cohorte.CODIGO_TIPOALTA, default=None, return_dtype=pl.String)
        base = base.with_columns(
            codigo.alias("codigo_alta"),
            codigo.is_in(list(SALIENTE)).fill_null(False).alias("saliente"),
            (codigo == "HOSP_DOMICILIARIA").fill_null(False).alias("hosp_domiciliaria"),
            (pl.col("rol") == "EVENTO").alias("evento"),
        )
    return base.with_columns(pl.col("procedencia").str.to_uppercase().str.contains(cohorte_texto_entrante()).fill_null(False).alias("entrante"))


def cohorte_texto_entrante() -> str:
    from .hospitales import TEXTO_ENTRANTE

    return TEXTO_ENTRANTE


def cargar_deis(ruta: Path = RUTA_DEIS) -> dict[str, str]:
    """Código de hospital -> grupo de pares (catálogo DEIS 2025 de la Fase 3, enmienda 02). Institutos y pediátricos van juntos."""
    if not Path(ruta).is_file():
        return {}
    d = pl.read_csv(ruta, infer_schema=False)
    return {r["cod_hospital"]: GRUPOS_DEIS.get(r["tipo"], "Sin clasificar") for r in d.iter_rows(named=True)}


# ---------------------------------------------------------------------------------------------------------------------------- recolección
def recolectar(ctx: Contexto, cat: catalogos.Catalogos | None = None) -> dict[str, Any]:
    """Recorre los años una sola vez y guarda lo mínimo para las cinco tablas."""
    cat = cat or catalogos.cargar_catalogos()
    e, e3 = elixhauser.cargar(), elixhauser.cargar(solo_3c=True)
    anios_res = set(ctx.umbrales.particion["anios_resultado"])
    falt: list[dict[str, Any]] = []
    vocab: dict[int, Counter] = {}
    d_rows: list[pl.DataFrame] = []
    e_rows: list[pl.DataFrame] = []
    c_rows: list[pl.DataFrame] = []
    for anio in ctx.fuente.anios():
        b = leer_anio(ctx, anio, cat, e, e3)
        c = b.filter(pl.col("C"))
        for var, _, universo in VARIABLES_FALTANTES:
            u = c if universo == "C" else b
            falt.append({"anio": anio, "variable": var, "universo": "cohorte C" if universo == "C" else "todos los egresos",
                         "n": u.height, "n_faltante": int(u[f"falta_{var}"].sum())})
        vocab[anio] = Counter(dict(c.filter(pl.col("cat3").is_not_null() & (pl.col("cat3") != "")).group_by("cat3").len().iter_rows()))
        if anio in anios_res:
            c_rows.append(c.select(pl.lit(anio, dtype=pl.Int32).alias("anio"), "hospital", "entrante", "saliente", "hosp_domiciliaria", "D", "evento"))
            d_rows.append(b.filter(pl.col("D")).select(pl.lit(anio, dtype=pl.Int32).alias("anio"), "hospital", "capitulo", "cat3", "evento"))
            e_rows.append(b.filter(pl.col("E")).select(pl.lit(anio, dtype=pl.Int32).alias("anio"), "hospital", "capitulo", "cat3", "estancia"))
        del b, c
    return {"faltantes": falt, "vocab": vocab, "C": pl.concat(c_rows) if c_rows else None, "D": pl.concat(d_rows) if d_rows else None,
            "E": pl.concat(e_rows) if e_rows else None}


# --------------------------------------------------------------------------------------------------------------------------------- tablas
def _etiquetas_cat3() -> dict[str, str]:
    ruta = rutas.RAIZ_PROYECTO / "CIE-10.csv"
    if not ruta.is_file():
        return {}
    d = pl.read_csv(ruta, columns=["Código", "Descripción"], infer_schema=False, encoding="utf8")
    d = d.filter(pl.col("Código").str.strip_chars().str.len_chars() == 3)
    return {r[0].strip().upper(): r[1].strip() for r in d.iter_rows()}


def tope_p99(est: pl.DataFrame, anios: Sequence[int]) -> int:
    """Percentil 99 de la estadía de la cohorte E en los años de entrenamiento (regla de la decisión 7)."""
    v = est.filter(pl.col("anio").is_in(list(anios)))["estancia"].drop_nulls().to_numpy()
    return int(np.percentile(v, 99, method="higher")) if v.size else 0


def tabla_r1(r: dict[str, Any], u, etiquetas: dict[str, str], minimo: int = 10) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    d, est = r["D"], r["E"]
    tope = tope_p99(est, u.particion["anios_ranking"])
    est = est.with_columns(pl.min_horizontal(pl.col("estancia"), pl.lit(tope)).alias("estancia_tope"))
    valido = lambda col: pl.col(col).is_not_null() & ~pl.col(col).is_in([cohorte.SIN_DATO, "sin capítulo", ""])  # noqa: E731
    top_cap = d.filter(valido("capitulo")).group_by("capitulo").len().sort(["len", "capitulo"], descending=[True, False]).head(N_CAPITULOS)["capitulo"].to_list()
    top_cod = d.filter(valido("cat3")).group_by("cat3").len().sort(["len", "cat3"], descending=[True, False]).head(N_CODIGOS)["cat3"].to_list()
    cap_de = {c: k for c, k in d.filter(pl.col("cat3").is_in(top_cod)).group_by("cat3").agg(pl.col("capitulo").mode().first()).iter_rows()}
    grupos = [("total", "Todos", pl.lit(True))]
    grupos += [("capitulo", c, pl.col("capitulo") == c) for c in top_cap]
    grupos += [("codigo", c, pl.col("cat3") == c) for c in top_cod]
    solo_resumen = [("sin_covid", "Sin COVID-19", (pl.col("capitulo") != "XXII").fill_null(True))]   # se resume, no se publica por año
    z = float(u["z_95"])
    filas = []
    for nivel, g, filtro in grupos:
        dm = d.filter(filtro).group_by("anio").agg(pl.len().alias("n_d"), pl.col("evento").sum().alias("n_fallecidos"))
        em = est.filter(filtro).group_by("anio").agg(pl.len().alias("n_e"), pl.col("estancia").mean().alias("estancia_media"),
                                                     pl.col("estancia_tope").mean().alias("estancia_media_tope"))
        m = dm.join(em, on="anio", how="full", coalesce=True).sort("anio")
        etiqueta = "Todos los egresos" if nivel == "total" else (CAPITULOS.get(g, g) if nivel == "capitulo" else ETIQUETAS_PROPIAS.get(g, etiquetas.get(g, g)))
        cap = g if nivel == "capitulo" else (cap_de.get(g) if nivel == "codigo" else None)
        m = m.with_columns(pl.lit(nivel).alias("nivel"), pl.lit(g).alias("grupo"), pl.lit(etiqueta).alias("etiqueta"), pl.lit(cap, dtype=pl.String).alias("_cap"))
        filas.append(m)
    t = pl.concat(filas, how="diagonal_relaxed").with_columns((pl.col("n_fallecidos") / pl.col("n_d")).alias("mortalidad"))
    t = cohorte.con_wilson(t, "n_fallecidos", "n_d", "", z)
    t = t.select(["nivel", "grupo", "etiqueta", "anio", "n_d", "n_fallecidos", "mortalidad", "ic_bajo", "ic_alto", "n_e", "estancia_media", "estancia_media_tope", "_cap"])
    # resumen: razón entre la pandemia y los años de referencia (más el total sin COVID-19, que no se publica por año)
    extra = []
    for nivel, g, filtro in solo_resumen:
        dm = d.filter(filtro).group_by("anio").agg(pl.len().alias("n_d"), pl.col("evento").sum().alias("n_fallecidos"))
        em = est.filter(filtro).group_by("anio").agg(pl.len().alias("n_e"), pl.col("estancia_tope").mean().alias("estancia_media_tope"))
        extra.append(dm.join(em, on="anio", how="full", coalesce=True).with_columns((pl.col("n_fallecidos") / pl.col("n_d")).alias("mortalidad"),
                     pl.lit(nivel).alias("nivel"), pl.lit(g).alias("grupo"), pl.lit("Todos los egresos sin COVID-19 (capítulo XXII)").alias("etiqueta")))
    base_res = pl.concat([t.select("nivel", "grupo", "etiqueta", "anio", "n_d", "n_fallecidos", "mortalidad", "n_e", "estancia_media_tope"),
                          *[x.select("nivel", "grupo", "etiqueta", "anio", "n_d", "n_fallecidos", "mortalidad", "n_e", "estancia_media_tope") for x in extra]],
                         how="diagonal_relaxed")
    res = []
    for (nivel, g, etiqueta), s in base_res.group_by(["nivel", "grupo", "etiqueta"], maintain_order=True):
        val = {int(a): (m, e_, k, ne) for a, m, e_, k, ne in s.select("anio", "mortalidad", "estancia_media_tope", "n_fallecidos", "n_e").iter_rows()}
        anios = ANIOS_PANDEMIA + ANIOS_REFERENCIA

        def razon(i: int, conteo: int) -> float | None:
            # Solo si los cinco años tienen la tasa y un conteo publicable: con un año suprimido, la razón permitiría reconstruirlo.
            if any(a not in val or val[a][i] is None or val[a][conteo] is None or val[a][conteo] < minimo for a in anios):
                return None
            ref = float(np.mean([val[a][i] for a in ANIOS_REFERENCIA]))
            return float(np.mean([val[a][i] for a in ANIOS_PANDEMIA])) / ref if ref > 0 else None
        rm, re2 = razon(0, 2), razon(1, 3)
        umbral = float(u["cambio_relativo_relevante"])
        res.append({"nivel": nivel, "grupo": g, "etiqueta": etiqueta, "n_d_2019_2023": int(s["n_d"].sum()), "n_e_2019_2023": int(s["n_e"].fill_null(0).sum()),
                    "razon_mortalidad_pandemia": rm, "razon_estancia_pandemia": re2,
                    "cambio_mortalidad_relevante": "no evaluable" if rm is None else ("si" if abs(rm - 1) >= umbral else "no"),
                    "cambio_estancia_relevante": "no evaluable" if re2 is None else ("si" if abs(re2 - 1) >= umbral else "no")})
    return t, pl.DataFrame(res), tope


def sin_razon_con_anios_ocultos(res: pl.DataFrame, publicada: pl.DataFrame) -> pl.DataFrame:
    """Anula la razón de pandemia de un grupo si la tabla publicada ocultó la tasa de alguno de sus cinco años.

    La supresión complementaria puede ocultar un año con 10 o más casos; con la razón y las tasas de los otros cuatro años a la vista, la tasa
    oculta se despejaría (y con ella, los fallecidos del año).
    """
    anios = {*ANIOS_PANDEMIA, *ANIOS_REFERENCIA}
    ocultas_m, ocultas_e = set(), set()
    for f in publicada.filter(pl.col("anio").is_in(list(anios))).iter_rows(named=True):
        if f["mortalidad"] is None:
            ocultas_m.add((f["nivel"], f["grupo"]))
        if f["estancia_media_tope"] is None:
            ocultas_e.add((f["nivel"], f["grupo"]))
    clave = pl.concat_str([pl.col("nivel"), pl.col("grupo")], separator="\x1f")
    km = [f"{a}\x1f{b}" for a, b in ocultas_m]
    ke = [f"{a}\x1f{b}" for a, b in ocultas_e]
    return res.with_columns(
        pl.when(clave.is_in(km)).then(None).otherwise(pl.col("razon_mortalidad_pandemia")).alias("razon_mortalidad_pandemia"),
        pl.when(clave.is_in(km)).then(pl.lit("no evaluable")).otherwise(pl.col("cambio_mortalidad_relevante")).alias("cambio_mortalidad_relevante"),
        pl.when(clave.is_in(ke)).then(None).otherwise(pl.col("razon_estancia_pandemia")).alias("razon_estancia_pandemia"),
        pl.when(clave.is_in(ke)).then(pl.lit("no evaluable")).otherwise(pl.col("cambio_estancia_relevante")).alias("cambio_estancia_relevante"),
    )


def tabla_r2(r: dict[str, Any]) -> pl.DataFrame:
    etiquetas = {v: e for v, e, _ in VARIABLES_FALTANTES}
    t = pl.DataFrame(r["faltantes"]).with_columns(pl.col("variable").replace_strict(etiquetas, return_dtype=pl.String).alias("etiqueta"))
    return t.with_columns((pl.col("n_faltante") / pl.col("n")).alias("prop_faltante")).select(["variable", "etiqueta", "universo", "anio", "n", "n_faltante", "prop_faltante"])


def tabla_r3(r: dict[str, Any], u) -> pl.DataFrame:
    vocab: dict[int, Counter] = r["vocab"]
    base_anios = [a for a in u.particion["anios_ranking"] if a in vocab]
    base: Counter = Counter()
    for a in base_anios:
        base.update(vocab[a])
    raros = {k for k, v in base.items() if v < MIN_FILAS_NIVEL}
    n_base = sum(base.values())
    filas = [{"periodo": f"{min(base_anios)}-{max(base_anios)}", "n_egresos": n_base, "n_codigos": len(base), "n_codigos_conservados": len(base) - len(raros),
              "n_codigos_raros": len(raros), "egresos_codigos_raros": sum(base[k] for k in raros), "n_codigos_nuevos": None, "egresos_codigos_nuevos": None}]
    for a in sorted(vocab):
        if a in base_anios:
            continue
        nuevos = {k for k in vocab[a] if k not in base}
        filas.append({"periodo": str(a), "n_egresos": sum(vocab[a].values()), "n_codigos": len(vocab[a]), "n_codigos_conservados": None, "n_codigos_raros": None,
                      "egresos_codigos_raros": sum(v for k, v in vocab[a].items() if k in raros), "n_codigos_nuevos": len(nuevos),
                      "egresos_codigos_nuevos": sum(vocab[a][k] for k in nuevos)})
    t = pl.DataFrame(filas, schema={"periodo": pl.String, "n_egresos": pl.Int64, "n_codigos": pl.Int64, "n_codigos_conservados": pl.Int64, "n_codigos_raros": pl.Int64,
                                    "egresos_codigos_raros": pl.Int64, "n_codigos_nuevos": pl.Int64, "egresos_codigos_nuevos": pl.Int64})
    return t.with_columns((pl.col("egresos_codigos_raros") / pl.col("n_egresos")).alias("prop_egresos_raros"),
                          (pl.col("egresos_codigos_nuevos") / pl.col("n_egresos")).alias("prop_egresos_nuevos"))


def tabla_r4(r: dict[str, Any], deis: dict[str, str], z: float) -> pl.DataFrame:
    c = r["C"].with_columns(pl.col("hospital").replace_strict(deis, default="Sin clasificar", return_dtype=pl.String).alias("grupo"))
    g = c.group_by("grupo").agg(
        pl.col("hospital").n_unique().alias("n_hospitales"), pl.len().alias("n_c"), pl.col("entrante").sum().alias("n_entrantes"),
        pl.col("saliente").sum().alias("n_salientes"), pl.col("hosp_domiciliaria").sum().alias("n_hosp_domiciliaria"),
        (pl.col("D") & pl.col("entrante")).sum().alias("n_d_entrantes"), (pl.col("D") & pl.col("entrante") & pl.col("evento")).sum().alias("n_fallecidos_entrantes"),
        (pl.col("D") & ~pl.col("entrante")).sum().alias("n_d_no_entrantes"), (pl.col("D") & ~pl.col("entrante") & pl.col("evento")).sum().alias("n_fallecidos_no_entrantes"),
    )
    total_ent = int(c["entrante"].sum())
    g = g.with_columns(
        (pl.col("n_entrantes") / pl.col("n_c")).alias("prop_entrantes"),
        (pl.col("n_salientes") / pl.col("n_c")).alias("prop_salientes"),
        (pl.col("n_hosp_domiciliaria") / pl.col("n_c")).alias("prop_hosp_domiciliaria"),
        (pl.col("n_entrantes") / total_ent).alias("parte_de_todos_los_entrantes") if total_ent else pl.lit(None, dtype=pl.Float64).alias("parte_de_todos_los_entrantes"),
        (pl.col("n_fallecidos_entrantes") / pl.col("n_d_entrantes")).alias("mortalidad_entrantes"),
        (pl.col("n_fallecidos_no_entrantes") / pl.col("n_d_no_entrantes")).alias("mortalidad_no_entrantes"),
    ).with_columns((pl.col("mortalidad_entrantes") / pl.col("mortalidad_no_entrantes")).alias("razon_mortalidad_entrantes"))
    orden = {k: i for i, k in enumerate(ORDEN_GRUPOS)}
    anios = r["C"]["anio"]
    periodo = f"{int(anios.min())}-{int(anios.max())}"
    return g.with_columns(pl.lit(periodo).alias("periodo"), pl.col("grupo").replace_strict(orden, default=99, return_dtype=pl.Int32).alias("_o")).sort("_o").drop("_o").select(
        ["periodo", "grupo", "n_hospitales", "n_c", "n_entrantes", "prop_entrantes", "parte_de_todos_los_entrantes", "n_salientes", "prop_salientes",
         "n_hosp_domiciliaria", "prop_hosp_domiciliaria", "n_d_entrantes", "n_fallecidos_entrantes", "mortalidad_entrantes", "n_d_no_entrantes",
         "n_fallecidos_no_entrantes", "mortalidad_no_entrantes", "razon_mortalidad_entrantes"])


def tabla_r5(r: dict[str, Any]) -> pl.DataFrame:
    filas = []
    for anio in sorted(r["D"]["anio"].unique().to_list()):
        nd = r["D"].filter(pl.col("anio") == anio).group_by("hospital").len()
        ne = r["E"].filter(pl.col("anio") == anio).group_by("hospital").len()
        filas.append({"anio": int(anio), "hospitales": nd.height, "hospitales_d_500": int((nd["len"] >= VOLUMEN_MINIMO).sum()),
                      "hospitales_e_500": int((ne["len"] >= VOLUMEN_MINIMO).sum())})
    return pl.DataFrame(filas)


# --------------------------------------------------------------------------------------------------------------------------------- figuras
def figura_r1(ctx: Contexto, tope: int) -> None:
    t = ctx.leer_tabla("r1_resultado_anio_diagnostico.csv", columnas_texto=["nivel", "grupo", "etiqueta"])
    caps = t.filter(pl.col("nivel") == "capitulo")
    orden = caps.group_by("grupo", maintain_order=True).agg(pl.col("n_d").sum()).sort("n_d", descending=True).head(6)["grupo"].to_list()
    fig, (a1, a2) = graficos.nueva_figura_multiple(1, 2, figsize=(11.5, 4.8))
    def trazar(ax, col: str, escala: float, titulo: str, eje: str) -> None:
        tot = t.filter(pl.col("nivel") == "total").sort("anio")
        ax.plot(tot["anio"].to_list(), (tot[col].cast(pl.Float64) * escala).to_list(), color="black", linewidth=2.4, marker="o", label="Todos")
        for i, g in enumerate(orden):
            s = caps.filter(pl.col("grupo") == g).sort("anio")
            ax.plot(s["anio"].to_list(), (s[col].cast(pl.Float64) * escala).to_list(), marker="o", linewidth=1.4, color=graficos.OKABE_ITO[i % 7],
                    label=f"{g}. {CAPITULOS.get(g, g)}")
        ax.axvspan(2019.5, 2021.5, color="#EEEEEE", zorder=0)
        ax.set_title(titulo, fontsize=10)
        ax.set_ylabel(eje)
        ax.set_xticks(sorted(tot["anio"].to_list()))
        graficos.formato_decimal(ax)
    trazar(a1, "mortalidad", 100.0, "Mortalidad intrahospitalaria (cohorte D)", "Mortalidad (%)")
    trazar(a2, "estancia_media_tope", 1.0, f"Estadía media de los sobrevivientes (cohorte E, tope {tope} días)", "Días")
    a2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ctx.figuras.guardar_figura(
        fig, id_figura="R01", tarea=TAREA, titulo="Mortalidad y estadía por año en los capítulos CIE-10 más frecuentes (2019–2023)",
        descripcion_alt="Dos paneles de líneas por año, de 2019 a 2023: a la izquierda la mortalidad de la cohorte D y a la derecha la estadía media de la cohorte E, "
                        "para todos los egresos y los seis capítulos CIE-10 con más egresos. La franja gris marca 2020 y 2021.",
        csv="r1_resultado_anio_diagnostico.csv", anios="2019 a 2023", decision="17")


def figura_r2(ctx: Contexto) -> None:
    t = ctx.leer_tabla("r2_faltantes_variables.csv", columnas_texto=["variable", "etiqueta", "universo"])
    etiquetas = [e for _, e, _ in VARIABLES_FALTANTES]
    anios = sorted(t["anio"].unique().to_list())
    m = np.full((len(etiquetas), len(anios)), np.nan)
    for i, e in enumerate(etiquetas):
        for j, a in enumerate(anios):
            s = t.filter((pl.col("etiqueta") == e) & (pl.col("anio") == a))
            if s.height and s["prop_faltante"][0] is not None:
                m[i, j] = 100.0 * float(s["prop_faltante"][0])
    fig, ax = graficos.nueva_figura(figsize=(8.0, 4.2))
    barra = graficos.mapa_calor(fig, ax, m, etiquetas, [str(a) for a in anios], vmin=0.0, etiqueta_barra="Faltantes (%)", fmt="{:.2f}")
    graficos.formato_barra_decimal(barra)
    ctx.figuras.guardar_figura(
        fig, id_figura="R02", tarea=TAREA, titulo="Faltantes de las variables candidatas del modelo, por año (%)",
        descripcion_alt="Mapa de calor con el porcentaje de valores faltantes de siete variables candidatas por año, de 2019 a 2024. Todas las celdas están bajo 1 %.",
        csv="r2_faltantes_variables.csv", anios="2019 a 2024")


def figura_r4(ctx: Contexto) -> None:
    t = ctx.leer_tabla("r4_traslados_tipo_hospital.csv", columnas_texto=["periodo", "grupo"])
    t = t.filter(pl.col("grupo") != "Sin clasificar")
    grupos = t["grupo"].to_list()
    series = [("prop_entrantes", "Traslados entrantes"), ("prop_salientes", "Traslados salientes"), ("prop_hosp_domiciliaria", "Hospitalización domiciliaria")]
    x = np.arange(len(grupos))
    w = 0.26
    fig, ax = graficos.nueva_figura(figsize=(8.6, 4.6))
    for i, (col, nombre) in enumerate(series):
        v = np.array([np.nan if x_ is None else 100.0 * float(x_) for x_ in t[col].to_list()])
        b = ax.bar(x + (i - 1) * w, v, w, label=nombre, color=graficos.OKABE_ITO[i])
        ax.bar_label(b, labels=[("" if np.isnan(y) else f"{y:.1f}".replace(".", ",")) for y in v], fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace(" de ", "\nde ") for g in grupos], fontsize=8)
    ax.set_ylabel("% de los egresos de la cohorte C")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    graficos.formato_decimal(ax)
    ctx.figuras.guardar_figura(
        fig, id_figura="R03", tarea=TAREA, titulo="Traslados y hospitalización domiciliaria según el tipo de hospital (DEIS 2025), 2019–2023",
        descripcion_alt="Barras agrupadas con el porcentaje de traslados entrantes, traslados salientes y hospitalizaciones domiciliarias sobre los egresos de adultos "
                        "hospitalizados, para los hospitales generales de alta complejidad, los de mediana complejidad y los institutos y pediátricos.",
        csv="r4_traslados_tipo_hospital.csv", anios="2019 a 2023", decision="16")


# ----------------------------------------------------------------------------------------------------------------------------- ejecución
def ejecutar(ctx: Contexto, cat: catalogos.Catalogos | None = None, deis: dict[str, str] | None = None) -> dict[str, Any]:
    r = recolectar(ctx, cat)
    u = ctx.umbrales
    out: dict[str, Any] = {}
    r2 = tabla_r2(r)
    ctx.escribir_tabla("r2_faltantes_variables.csv", r2, tarea=TAREA, descripcion="Faltantes de las variables candidatas por variable y año (O2)",
                       cols_n=["n", "n_faltante"], dependientes=["prop_faltante"])
    r3 = tabla_r3(r, u)
    ctx.escribir_tabla("r3_vocabulario_diagnostico.csv", r3, tarea=TAREA, descripcion="Códigos del diagnóstico principal con pocos casos y códigos nuevos (O3)",
                       cols_n=["n_egresos", "egresos_codigos_raros", "egresos_codigos_nuevos"], dependientes=["prop_egresos_raros", "prop_egresos_nuevos"])
    out.update(r2=r2, r3=r3)
    if r["D"] is not None:
        r1, r1r, tope = tabla_r1(r, u, _etiquetas_cat3(), ctx.celda_minima)
        tot = r1.filter(pl.col("nivel") == "total").select("anio", pl.col("n_d").alias("_total_d"), pl.col("n_fallecidos").alias("_total_k"), pl.col("n_e").alias("_total_e"))
        # el capítulo de cada código frecuente: un capítulo publicado menos sus códigos publicados deja un resto que también se protege
        capt = r1.filter(pl.col("nivel") == "capitulo").select("anio", pl.col("grupo").alias("_cap"), pl.col("n_d").alias("_tcap_d"), pl.col("n_fallecidos").alias("_tcap_k"),
                                                               pl.col("n_e").alias("_tcap_e"))
        r1_aux = r1.join(tot, on="anio", how="left", maintain_order="left").join(capt, on=["anio", "_cap"], how="left", maintain_order="left").with_columns(
            [pl.when(pl.col("nivel") == "codigo").then(pl.col(c)).otherwise(None).alias(c) for c in ("_tcap_d", "_tcap_k", "_tcap_e")])
        ctx.escribir_tabla("r1_resultado_anio_diagnostico.csv", r1_aux, tarea=TAREA,
                           descripcion="Mortalidad y estadía por año en total, por capítulo y por código frecuente (O8)",
                           cols_n=["n_d", "n_fallecidos", "n_e"], dependientes=["mortalidad", "ic_bajo", "ic_alto", "estancia_media", "estancia_media_tope"],
                           auxiliares=["_total_d", "_total_k", "_total_e", "_cap", "_tcap_d", "_tcap_k", "_tcap_e"])
        r1r = sin_razon_con_anios_ocultos(r1r, ctx.leer_tabla("r1_resultado_anio_diagnostico.csv", columnas_texto=["nivel", "grupo", "etiqueta"]))
        ctx.escribir_tabla("r1_resumen_pandemia.csv", r1r, tarea=TAREA, descripcion="Razón entre 2020–2021 y 2019, 2022 y 2023 por grupo (O8)",
                           cols_n=["n_d_2019_2023", "n_e_2019_2023"], dependientes=["razon_mortalidad_pandemia", "razon_estancia_pandemia"])
        deis = cargar_deis() if deis is None else deis
        r4 = tabla_r4(r, deis, float(u["z_95"]))
        ctx.escribir_tabla("r4_traslados_tipo_hospital.csv", r4, tarea=TAREA, descripcion="Traslados según el tipo de hospital del catálogo DEIS 2025 (O7)",
                           cols_n=["n_c", "n_entrantes", "n_salientes", "n_hosp_domiciliaria", "n_d_entrantes", "n_fallecidos_entrantes", "n_d_no_entrantes",
                                   "n_fallecidos_no_entrantes"],
                           dependientes=["prop_entrantes", "parte_de_todos_los_entrantes", "prop_salientes", "prop_hosp_domiciliaria", "mortalidad_entrantes",
                                         "mortalidad_no_entrantes", "razon_mortalidad_entrantes"])
        r5 = tabla_r5(r)
        ctx.escribir_tabla("r5_volumen_cohortes.csv", r5, tarea=TAREA, descripcion="Hospitales con al menos 500 egresos en las cohortes D y E (decisión 6)")
        out.update(r1=r1, r1_resumen=r1r, tope=tope, r4=r4, r5=r5)
        figura_r1(ctx, tope)
        figura_r4(ctx)
    figura_r2(ctx)
    return out
