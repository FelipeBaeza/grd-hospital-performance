"""Cohortes A a E y cascada de exclusiones (Fase 1, sección 9.3; reproduce la Fase 2, tarea T07).

  EX01 GRD no agrupable (sin código, centinela, formato inválido o MDC 99)
  EX02 recién nacidos (MDC 15)
  EX03 obstétricos sin comorbilidades (MDC 14 sin categorías de Elixhauser; solo diagnósticos secundarios)
  A = cohorte dura;  B = A y adultos (18 años o más);  C = B y actividad de hospitalización
  D = C con desenlace conocido (modelo de mortalidad);  E = D, solo sobrevivientes y con estadía mayor que cero (modelo de estadía)

Las exclusiones se aplican en secuencia, por eso suman exactamente el total de egresos del año.
Para 2024 solo se construye hasta C: D y E necesitan el desenlace, que está sellado hasta la Fase 5.

Enmienda 02 (borrador, ``Opciones``; apagadas por defecto, y sin ellas todo queda igual byte a byte):
  - ``mdc14_completo``: EX03 excluye el MDC 14 completo (decisión conservadora sobre H-07).
  - ``excluir_tipoalta_e``: tipos de alta que salen de E en un paso nuevo, EX09 (H-08). Se toman de
    ``config/tipoalta_roles.csv`` con ``rol_estancia = INCLUIDO_SENSIBILIDAD`` (alta voluntaria y fuga).
    EX09 va al final de la tabla de la cascada para no mover el orden de los pasos existentes.

Segunda revisión del profesor guía (02-11-2026; Fase 2 v8, secciones 6.2 y 12), también en ``Opciones`` y apagadas por defecto.
Solo cambian D y E, así que en 2024 (sellado, solo hasta C) no tienen efecto:
  - ``tipoalta_vivos``: tipos de alta que dejan de ser censura y cuentan como egreso vivo (NO_EVENTO) en el modelo de mortalidad.
    Por defecto de la revisión, la hospitalización domiciliaria. No entran a E: salen en el paso EX11, porque su estadía en el
    hospital depende del programa de hospitalización domiciliaria de cada hospital.
  - ``deduplicar_q15``: las filas repetidas por clave natural (regla Q15 de la Fase 2: paciente, hospital, ingreso, alta, diagnóstico
    principal, nacimiento y sexo) salen de D en el paso EX10; se conserva la primera fila de cada clave.
  - ``excluir_entrantes_enlazados``: los traslados entrantes que se enlazan con un traslado saliente previo del mismo paciente, a
    otro hospital y dentro de ``ventana_enlace_traslado_dias``, salen de D en el paso EX12 (propuesta de la revisión para la
    decisión 16). El enlace se busca solo dentro del régimen de identificadores del año.
Los pasos nuevos van al final de la tabla. Con ellos: C = EX06 + EX10 + EX12 + D y D = EX07 + EX08 + EX09 + EX11 + E.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from . import definiciones as dfn
from . import elixhauser, esquema
from .config import ANIO_SELLADO, Umbrales, cargar_umbrales

PASOS = [
    ("TOTAL", "total", "Egresos leídos", "no"),
    ("EX01", "exclusion", "GRD no agrupable: sin código, centinela, formato inválido o MDC 99", "no"),
    ("EX02", "exclusion", "Recién nacidos (MDC 15)", "no"),
    ("EX03", "exclusion", "Obstétricos sin comorbilidades (MDC 14 sin categorías de Elixhauser)", "no"),
    ("A", "cohorte", "Variante A: cohorte dura", "no"),
    ("EX04", "exclusion", "Menores de 18 años o con edad desconocida", "no"),
    ("B", "cohorte", "Variante B: adultos", "no"),
    ("EX05", "exclusion", "Actividad distinta de hospitalización", "no"),
    ("C", "cohorte", "Variante C: adultos hospitalizados", "no"),
    ("EX06", "exclusion", "Censurados por tipo de alta o sin dato de desenlace", "si"),
    ("D", "cohorte", "Variante D: cohorte del modelo de mortalidad", "si"),
    ("EX07", "exclusion", "Fallecidos (salen del modelo de estadía)", "si"),
    ("EX08", "exclusion", "Sobrevivientes con estadía de 0 días o desconocida", "si"),
    ("E", "cohorte", "Variante E: cohorte del modelo de estadía", "si"),
    ("EX09", "exclusion", "Alta voluntaria y fuga, fuera de la cohorte de estadía (enmienda 02)", "si"),
    ("EX10", "exclusion", "Filas repetidas por clave natural (Q15), fuera de D (segunda revisión del profesor)", "si"),
    ("EX11", "exclusion", "Egresos vivos agregados por la revisión (hospitalización domiciliaria), fuera de E", "si"),
    ("EX12", "exclusion", "Traslados entrantes enlazados con un traslado saliente previo, fuera de D (segunda revisión del profesor)", "si"),
]
TIPOALTA_VIVOS_REVISION = ("HOSPITALIZACIÓN DOMICILIARIA",)
CLAVE_Q15 = ("ID_PACIENTE_NUM", "COD_HOSPITAL", "FECHA_INGRESO", "FECHAALTA", "DIAGNOSTICO1", "FECHA_NACIMIENTO", "SEXO")
DESC_EX03_MDC14_COMPLETO = "Obstétricos: MDC 14 completo (enmienda 02)"


@dataclass(frozen=True)
class Opciones:
    """Reglas de la enmienda 02 y de la segunda revisión del profesor. Apagadas por defecto."""
    mdc14_completo: bool = False
    excluir_tipoalta_e: tuple[str, ...] = ()
    tipoalta_vivos: tuple[str, ...] = ()
    deduplicar_q15: bool = False
    excluir_entrantes_enlazados: bool = False

    @property
    def activas(self) -> bool:
        return (self.mdc14_completo or bool(self.excluir_tipoalta_e) or bool(self.tipoalta_vivos) or self.deduplicar_q15
                or self.excluir_entrantes_enlazados)

    @property
    def necesita_etiqueta(self) -> bool:
        return bool(self.excluir_tipoalta_e) or bool(self.tipoalta_vivos)


def tipoalta_fuera_de_e(roles: pl.DataFrame) -> tuple[str, ...]:
    """Tipos de alta que la configuración deja solo para la sensibilidad de estadía (``rol_estancia = INCLUIDO_SENSIBILIDAD``)."""
    return tuple(sorted(roles.filter(pl.col("rol_estancia") == "INCLUIDO_SENSIBILIDAD")["tipoalta"].to_list()))


def opciones_enmienda02(roles: pl.DataFrame) -> Opciones:
    return Opciones(mdc14_completo=True, excluir_tipoalta_e=tipoalta_fuera_de_e(roles))


def opciones_revision(hd_vivo: bool = True, deduplicar: bool = True, entrantes: bool = True, mdc14: bool = True) -> Opciones:
    """Las cuatro reglas que propone la segunda revisión del profesor (02-11-2026), para el análisis de sensibilidad del indicador."""
    return Opciones(mdc14_completo=mdc14, tipoalta_vivos=TIPOALTA_VIVOS_REVISION if hd_vivo else (), deduplicar_q15=deduplicar,
                    excluir_entrantes_enlazados=entrantes)


def opciones_config(u: Umbrales | None = None) -> Opciones:
    """Reglas de la cohorte oficial según la sección [cohorte] de umbrales_fase3.toml (sin la sección, todas apagadas)."""
    u = u or cargar_umbrales()
    c = u.datos.get("cohorte", {})
    return Opciones(mdc14_completo=bool(c.get("mdc14_completo", False)), excluir_tipoalta_e=tuple(c.get("excluir_tipoalta_e", ())),
                    tipoalta_vivos=tuple(c.get("tipoalta_vivos", ())), deduplicar_q15=bool(c.get("deduplicar_q15", False)),
                    excluir_entrantes_enlazados=bool(c.get("excluir_entrantes_enlazados", False)))


DESC_EX06_CON_Q15 = "Censurados, sin dato de desenlace o filas repetidas (Q15)"


def cascada_publicable(casc: pl.DataFrame) -> pl.DataFrame:
    """Cascada por año para publicar: EX10 (filas repetidas, a veces de 1 a 9 por año) se suma a EX06, para que ningún conteo pequeño se
    pueda despejar restando. El período 2019–2023 se publica aparte con EX10 separado."""
    if "EX10" not in casc["paso"].to_list():
        return casc
    ex10 = casc.filter(pl.col("paso") == "EX10").select("periodo", pl.col("n").alias("_ex10"))
    out = casc.filter(pl.col("paso") != "EX10").join(ex10, on="periodo", how="left").with_columns(
        pl.when(pl.col("paso") == "EX06").then(pl.col("n") + pl.col("_ex10").fill_null(0)).otherwise(pl.col("n")).alias("n"),
        pl.when(pl.col("paso") == "EX06").then(pl.lit(DESC_EX06_CON_Q15)).otherwise(pl.col("descripcion")).alias("descripcion"))
    tot = out.filter(pl.col("paso") == "TOTAL").select("periodo", pl.col("n").alias("_t"))
    return out.join(tot, on="periodo").with_columns((pl.col("n") / pl.col("_t")).alias("prop_del_total")).drop("_ex10", "_t").select(casc.columns)


def marcar_q15(plata: pl.DataFrame, desenlace: pl.DataFrame) -> pl.Series:
    """Verdadero en las filas repetidas por clave natural (Q15), salvo la primera de cada clave. Necesita la fecha de alta: solo 2019 a 2023."""
    k = plata.select(*[c for c in CLAVE_Q15 if c != "FECHAALTA"]).with_columns(desenlace["FECHAALTA"])
    return (k["ID_PACIENTE_NUM"].is_not_null() & ~k.select(pl.struct(list(CLAVE_Q15)).is_first_distinct()).to_series()).alias("DUP_Q15")


def regimen_de(anio: int, u: Umbrales) -> list[int]:
    for g in u.datos["regimenes_id"]["anios"]:
        if anio in g:
            return [int(a) for a in g]
    raise ValueError(f"año sin régimen de identificadores: {anio}")


def entrantes_enlazados(anio: int, dir_plata: Path | str, u: Umbrales) -> pl.Series:
    """Para cada fila del año: verdadero si es un traslado entrante (TIPO_PROCEDENCIA con «OTROS HOSPITALES») de un paciente que tuvo un
    traslado saliente (derivación a otro hospital del servicio o de la red) en otro hospital, con alta entre 0 y W días antes del ingreso.

    El saliente se busca en el mismo año y en el anterior si es del mismo régimen de identificadores. Como en la Fase 2, el saliente debe ser
    de un adulto hospitalizado. Solo para años con desenlace (2019 a 2023): el saliente necesita el tipo de alta."""
    if anio == ANIO_SELLADO:
        raise ValueError("el enlace usa el tipo de alta: no se calcula en el año sellado")
    dir_plata = Path(dir_plata)
    w = int(u["ventana_enlace_traslado_dias"])
    reg = regimen_de(anio, u)
    c = pl.col
    sal = []
    for a in (anio - 1, anio):
        if a not in reg or not (dir_plata / f"desenlace_{a}.parquet").is_file():
            continue
        p = pl.read_parquet(dir_plata / f"plata_{a}.parquet", columns=["ID_EPISODIO", "ID_PACIENTE_NUM", "COD_HOSPITAL", "EDAD_ANIOS", "TIPO_ACTIVIDAD"])
        dd = pl.read_parquet(dir_plata / f"desenlace_{a}.parquet", columns=["ES_TRASLADO_SALIENTE", "FECHAALTA"])
        sal.append(p.hstack(dd).filter((c("ES_TRASLADO_SALIENTE") == 1) & c("ID_PACIENTE_NUM").is_not_null() & c("FECHAALTA").is_not_null()
                                       & (c("EDAD_ANIOS") >= int(u["edad_adulto"])) & (c("TIPO_ACTIVIDAD") == dfn.ACTIVIDAD_HOSP))
                   .select("ID_PACIENTE_NUM", c("COD_HOSPITAL").alias("hosp_s"), "FECHAALTA"))
    p = pl.read_parquet(dir_plata / f"plata_{anio}.parquet", columns=["ID_EPISODIO", "ID_PACIENTE_NUM", "COD_HOSPITAL", "FECHA_INGRESO", "ES_TRASLADO_ENTRANTE"])
    ent = p.with_row_index("_i").filter((c("ES_TRASLADO_ENTRANTE") == 1) & c("ID_PACIENTE_NUM").is_not_null() & c("FECHA_INGRESO").is_not_null())
    marca = np.zeros(p.height, dtype=bool)
    if sal and ent.height:
        pares = (ent.select("_i", "ID_PACIENTE_NUM", "COD_HOSPITAL", "FECHA_INGRESO").join(pl.concat(sal), on="ID_PACIENTE_NUM")
                 .with_columns((c("FECHA_INGRESO") - c("FECHAALTA")).dt.total_days().alias("delta"))
                 .filter((c("delta") >= 0) & (c("delta") <= w) & (c("COD_HOSPITAL") != c("hosp_s"))))
        marca[pares["_i"].unique().to_numpy()] = True
    return pl.Series("ENTRANTE_ENLAZADO", marca)


def marcar(base: pl.DataFrame, edad_adulto: int = 18, opciones: Opciones = Opciones()) -> pl.DataFrame:
    """Indicadores de exclusión y de cohorte. ``base`` trae MDC, GRD_ESTADO, EDAD_ANIOS, TIPO_ACTIVIDAD, N_COMORB_MDC14 y, si hay desenlace, ROL y ESTANCIA_DIAS.
    Con ``opciones.excluir_tipoalta_e`` también hace falta ETIQUETA_TIPOALTA."""
    c = pl.col
    f = lambda e: e.fill_null(False)  # noqa: E731
    con_res = "ROL" in base.columns
    b = base.with_columns((c("GRD_ESTADO") != "agrupable").alias("EX01"))
    obst = (c("MDC") == "14") if opciones.mdc14_completo else ((c("MDC") == "14") & (c("N_COMORB_MDC14") == 0))
    b = b.with_columns(
        f(~c("EX01") & (c("MDC") == "15")).alias("EX02"),
        f(~c("EX01") & obst).alias("EX03"),
    )
    b = b.with_columns((~(c("EX01") | c("EX02") | c("EX03"))).alias("A"))
    b = b.with_columns(f(c("A") & (c("EDAD_ANIOS") >= edad_adulto)).alias("B"))
    b = b.with_columns(f(c("B") & (c("TIPO_ACTIVIDAD") == dfn.ACTIVIDAD_HOSP)).alias("C"))
    if con_res:
        if opciones.necesita_etiqueta and "ETIQUETA_TIPOALTA" not in b.columns:
            raise ValueError("excluir_tipoalta_e y tipoalta_vivos necesitan la columna ETIQUETA_TIPOALTA")
        rol = c("ROL")
        if opciones.tipoalta_vivos:
            vivo = f(c("ETIQUETA_TIPOALTA").is_in(list(opciones.tipoalta_vivos)))
            rol = pl.when(vivo).then(pl.lit("NO_EVENTO")).otherwise(c("ROL"))
        d0 = c("C") & rol.is_in(["EVENTO", "NO_EVENTO"])
        if not (opciones.deduplicar_q15 or opciones.excluir_entrantes_enlazados or opciones.tipoalta_vivos):
            b = b.with_columns(d0.alias("D"))
        else:
            for col, activa in (("DUP_Q15", opciones.deduplicar_q15), ("ENTRANTE_ENLAZADO", opciones.excluir_entrantes_enlazados)):
                if activa and col not in b.columns:
                    raise ValueError(f"la opción necesita la columna {col}")
            b = b.with_columns(d0.alias("_D0"), rol.alias("_ROL"))
            resto = c("_D0")
            if opciones.deduplicar_q15:
                b = b.with_columns(f(resto & c("DUP_Q15")).alias("EX10"))
                resto = resto & ~c("EX10")
            if opciones.excluir_entrantes_enlazados:
                b = b.with_columns(f(resto & c("ENTRANTE_ENLAZADO")).alias("EX12"))
                resto = resto & ~c("EX12")
            b = b.with_columns(resto.alias("D"))
            rol = c("_ROL")
        e0 = f(c("D") & (rol == "NO_EVENTO") & (c("ESTANCIA_DIAS") > 0))
        if opciones.tipoalta_vivos:
            b = b.with_columns(f(e0 & c("ETIQUETA_TIPOALTA").is_in(list(opciones.tipoalta_vivos))).alias("EX11"))
            e0 = e0 & ~c("EX11")
        if opciones.excluir_tipoalta_e:
            b = b.with_columns(f(e0 & c("ETIQUETA_TIPOALTA").is_in(list(opciones.excluir_tipoalta_e))).alias("EX09"))
            e0 = e0 & ~c("EX09")
        b = b.with_columns(e0.alias("E"))
        if "_ROL" in b.columns:
            b = b.with_columns(c("_ROL").alias("ROL_MODELO")).drop("_D0", "_ROL")
    return b


def cascada_anio(b: pl.DataFrame, anio: int, opciones: Opciones = Opciones()) -> pl.DataFrame:
    """Tabla de la cascada de un año, con el mismo formato de la Fase 2 (periodo, orden, paso, tipo, descripcion, n, ...)."""
    c = pl.col
    con_res = "D" in b.columns
    n = b.height
    s = b.select(
        c("EX01").sum().alias("EX01"), c("EX02").sum().alias("EX02"), c("EX03").sum().alias("EX03"), c("A").sum().alias("A"),
        c("B").sum().alias("B"), c("C").sum().alias("C"),
        *( [c("D").sum().alias("D"), (c("D") & (c("ROL") == "EVENTO")).sum().alias("EX07"), c("E").sum().alias("E")] if con_res else []),
    ).row(0, named=True)
    v = {"TOTAL": n, **{k: int(x) for k, x in s.items()}}
    v["EX04"], v["EX05"] = v["A"] - v["B"], v["B"] - v["C"]
    if con_res:
        for k in ("EX09", "EX10", "EX11", "EX12"):
            if k in b.columns:
                v[k] = int(b[k].sum())
        v["EX06"] = v["C"] - v["D"] - v.get("EX10", 0) - v.get("EX12", 0)
        v["EX08"] = v["D"] - v["EX07"] - v["E"] - v.get("EX09", 0) - v.get("EX11", 0)
    filas = []
    for orden, (paso, tipo, desc, usa) in enumerate(PASOS):
        if paso not in v:
            continue
        if paso == "EX03" and opciones.mdc14_completo:
            desc = DESC_EX03_MDC14_COMPLETO
        filas.append({"periodo": str(anio), "orden": orden, "paso": paso, "tipo": tipo, "descripcion": desc, "n": v[paso],
                      "prop_del_total": v[paso] / n if n else 0.0, "usa_desenlace": usa,
                      "alcance": "resultado" if usa == "si" else "estructura"})
    return pl.DataFrame(filas)


def cuadra(df: pl.DataFrame) -> bool:
    """Entrada = salida + excluidos en cada tramo de la cascada."""
    v = dict(zip(df["paso"], df["n"]))
    ok = v["TOTAL"] == v["EX01"] + v["EX02"] + v["EX03"] + v["A"] and v["A"] == v["EX04"] + v["B"] and v["B"] == v["EX05"] + v["C"]
    if "D" in v:
        ok = (ok and v["C"] == v["EX06"] + v.get("EX10", 0) + v.get("EX12", 0) + v["D"]
              and v["D"] == v["EX07"] + v["EX08"] + v.get("EX09", 0) + v.get("EX11", 0) + v["E"])
    return bool(ok)


def comparar_con_referencia(cascada: pl.DataFrame, ruta_ref: Path | str) -> pl.DataFrame:
    """Diferencia entre la cascada calculada y la medida en la Fase 2, por periodo y paso (la columna ``diferencia`` debe ser 0)."""
    ref = pl.read_csv(ruta_ref, infer_schema=False).select(
        pl.col("periodo"), pl.col("paso"), pl.col("n").cast(pl.Int64).alias("n_fase2")
    )
    return cascada.select(pl.col("periodo").cast(pl.String), "paso", pl.col("n").cast(pl.Int64).alias("n_fase3")).join(
        ref, on=["periodo", "paso"], how="left"
    ).with_columns((pl.col("n_fase3") - pl.col("n_fase2")).alias("diferencia"))


def construir_cohorte(anio: int, dir_plata: Path | str, dir_cohortes: Path | str, u: Umbrales | None = None, e: elixhauser.Elixhauser | None = None,
                      opciones: Opciones = Opciones()) -> pl.DataFrame:
    """Marca las cohortes de un año y escribe ``cohorte_<año>.parquet`` (ID_EPISODIO, indicadores y N_COMORB_MDC14). Devuelve la cascada del año.
    Con ``opciones`` activas (enmienda 02) también escribe EX09 si corresponde; ``dir_cohortes`` debe ser otra carpeta."""
    u = u or cargar_umbrales()
    e = e or elixhauser.cargar()
    dir_plata, dir_cohortes = Path(dir_plata), Path(dir_cohortes)
    dir_cohortes.mkdir(parents=True, exist_ok=True)
    cols = ["ID_EPISODIO", "MDC", "GRD_ESTADO", "EDAD_ANIOS", "TIPO_ACTIVIDAD"]
    base = pl.read_parquet(dir_plata / f"plata_{anio}.parquet", columns=cols)
    # la comorbilidad solo se calcula para MDC 14 agrupable: se leen los diagnósticos de esas filas únicamente
    m = (pl.col("MDC") == "14") & (pl.col("GRD_ESTADO") == "agrupable")
    sub_ids = base.filter(m)["ID_EPISODIO"]
    dx = pl.read_parquet(dir_plata / f"plata_{anio}.parquet", columns=["ID_EPISODIO", *esquema.DIAG_SEC]).filter(pl.col("ID_EPISODIO").is_in(sub_ids.implode()))
    comorb = np.full(base.height, -1, dtype=np.int32)
    if dx.height:
        flags = elixhauser.calcular_flags(dx, esquema.DIAG_SEC, e)["N_COMORB_ELIX"].to_numpy()
        pos = base.with_row_index("i").filter(m)["i"].to_numpy()
        comorb[pos] = flags
    base = base.with_columns(pl.when(pl.Series(comorb) >= 0).then(pl.Series(comorb)).alias("N_COMORB_MDC14"))
    if anio != ANIO_SELLADO:
        cols_des = ["ID_EPISODIO", "ROL", "ESTANCIA_DIAS"] + (["ETIQUETA_TIPOALTA"] if opciones.necesita_etiqueta else [])
        des = pl.read_parquet(dir_plata / f"desenlace_{anio}.parquet", columns=cols_des + (["FECHAALTA"] if opciones.deduplicar_q15 else []))
        if des["ID_EPISODIO"].to_list() != base["ID_EPISODIO"].to_list():
            raise ValueError("Plata y desenlace deben tener las mismas filas en el mismo orden")
        base = base.with_columns(*[des[k] for k in cols_des[1:]])
        if opciones.deduplicar_q15:
            clave = pl.read_parquet(dir_plata / f"plata_{anio}.parquet", columns=[k for k in CLAVE_Q15 if k != "FECHAALTA"])
            base = base.with_columns(marcar_q15(clave, des))
        if opciones.excluir_entrantes_enlazados:
            base = base.with_columns(entrantes_enlazados(anio, dir_plata, u))
    marcado = marcar(base, int(u["edad_adulto"]), opciones)
    casc = cascada_anio(marcado, anio, opciones)
    keep = ["ID_EPISODIO", "N_COMORB_MDC14", "EX01", "EX02", "EX03", "A", "B", "C"] + (["D", "E"] if "D" in marcado.columns else [])
    keep += [k for k in ("EX09", "EX10", "EX11", "EX12", "ROL_MODELO") if k in marcado.columns]
    marcado.select(keep).write_parquet(dir_cohortes / f"cohorte_{anio}.parquet", compression="zstd")
    return casc
