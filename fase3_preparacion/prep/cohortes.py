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
]
DESC_EX03_MDC14_COMPLETO = "Obstétricos: MDC 14 completo (enmienda 02)"


@dataclass(frozen=True)
class Opciones:
    """Reglas de la enmienda 02 (borrador). Apagadas por defecto."""
    mdc14_completo: bool = False
    excluir_tipoalta_e: tuple[str, ...] = ()

    @property
    def activas(self) -> bool:
        return self.mdc14_completo or bool(self.excluir_tipoalta_e)


def tipoalta_fuera_de_e(roles: pl.DataFrame) -> tuple[str, ...]:
    """Tipos de alta que la configuración deja solo para la sensibilidad de estadía (``rol_estancia = INCLUIDO_SENSIBILIDAD``)."""
    return tuple(sorted(roles.filter(pl.col("rol_estancia") == "INCLUIDO_SENSIBILIDAD")["tipoalta"].to_list()))


def opciones_enmienda02(roles: pl.DataFrame) -> Opciones:
    return Opciones(mdc14_completo=True, excluir_tipoalta_e=tipoalta_fuera_de_e(roles))


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
        b = b.with_columns((c("C") & c("ROL").is_in(["EVENTO", "NO_EVENTO"])).alias("D"))
        if opciones.excluir_tipoalta_e:
            if "ETIQUETA_TIPOALTA" not in b.columns:
                raise ValueError("excluir_tipoalta_e necesita la columna ETIQUETA_TIPOALTA")
            b = b.with_columns(f(c("D") & (c("ROL") == "NO_EVENTO") & (c("ESTANCIA_DIAS") > 0)).alias("_E0"))
            b = b.with_columns(f(c("_E0") & c("ETIQUETA_TIPOALTA").is_in(list(opciones.excluir_tipoalta_e))).alias("EX09"))
            b = b.with_columns((c("_E0") & ~c("EX09")).alias("E")).drop("_E0")
        else:
            b = b.with_columns(f(c("D") & (c("ROL") == "NO_EVENTO") & (c("ESTANCIA_DIAS") > 0)).alias("E"))
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
        v["EX06"] = v["C"] - v["D"]
        if "EX09" in b.columns:
            v["EX09"] = int(b["EX09"].sum())
        v["EX08"] = v["D"] - v["EX07"] - v["E"] - v.get("EX09", 0)
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
        ok = ok and v["C"] == v["EX06"] + v["D"] and v["D"] == v["EX07"] + v["EX08"] + v.get("EX09", 0) + v["E"]
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
        cols_des = ["ID_EPISODIO", "ROL", "ESTANCIA_DIAS"] + (["ETIQUETA_TIPOALTA"] if opciones.excluir_tipoalta_e else [])
        des = pl.read_parquet(dir_plata / f"desenlace_{anio}.parquet", columns=cols_des)
        if des["ID_EPISODIO"].to_list() != base["ID_EPISODIO"].to_list():
            raise ValueError("Plata y desenlace deben tener las mismas filas en el mismo orden")
        base = base.with_columns(*[des[k] for k in cols_des[1:]])
    marcado = marcar(base, int(u["edad_adulto"]), opciones)
    casc = cascada_anio(marcado, anio, opciones)
    keep = ["ID_EPISODIO", "N_COMORB_MDC14", "EX01", "EX02", "EX03", "A", "B", "C"] + (["D", "E"] if "D" in marcado.columns else [])
    keep += ["EX09"] if "EX09" in marcado.columns else []
    marcado.select(keep).write_parquet(dir_cohortes / f"cohorte_{anio}.parquet", compression="zstd")
    return casc
