"""Ayudas de prueba de la Tarea 12: constructores a mano para el ledger, las decisiones y el cierre (no leen datos)."""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from eda import config
from eda.contexto import ESQUEMA_METRICAS, Contexto
from eda.ledger import Reglas

COLUMNAS_CIFRAS = ("id", "documento", "ubicacion", "descripcion", "valor", "unidad", "decimales", "tipo", "alcance", "requiere_resultado", "clave_eda", "nota")


# ------------------------------------------------------------------------------------------------------------------------------------ ledger
def cifra_fila(
    id: str = "C900", *, valor: float | int | str, tipo: str = "proporcion", unidad: str = "porcentaje", decimales: int = 2, alcance: str = "2023",
    requiere_resultado: str = "no", clave: str = "x.y", documento: str = "Fase 2", nota: str = "", descripcion: str = "cifra de prueba", ubicacion: str = "1.1",
) -> dict[str, str]:
    """Una fila de ``cifras_documentadas.csv`` (todo texto) con valores por defecto de prueba."""
    return {
        "id": id, "documento": documento, "ubicacion": ubicacion, "descripcion": descripcion, "valor": str(valor), "unidad": unidad, "decimales": str(decimales),
        "tipo": tipo, "alcance": alcance, "requiere_resultado": requiere_resultado, "clave_eda": clave, "nota": nota,
    }


def cifras_df(*filas: Mapping[str, str]) -> pl.DataFrame:
    """Tabla de cifras documentadas (todas las columnas como texto) con las filas dadas."""
    return pl.DataFrame([{c: f.get(c, "") for c in COLUMNAS_CIFRAS} for f in filas], schema={c: pl.String for c in COLUMNAS_CIFRAS})


def metrica(valor: float | None, *, ic: tuple[float, float] | None = None, alcance: str = "", tabla: str = "t.csv", nota: str = "", nivel: str = "extracto") -> dict[str, Any]:
    """Una métrica indexada como la entrega ``ledger.indexar_metricas``."""
    return {
        "valor": None if valor is None else float(valor), "ic_bajo": None if ic is None else float(ic[0]), "ic_alto": None if ic is None else float(ic[1]),
        "tabla": tabla, "nivel": nivel, "alcance": alcance, "nota": nota,
    }


def metricas_df(*filas: Mapping[str, Any]) -> pl.DataFrame:
    """Tabla de métricas con el esquema de ``Contexto.metricas()``; cada fila lleva al menos ``clave`` y ``valor``."""
    base = {"ic_bajo": None, "ic_alto": None, "tarea": "T00", "tabla": "t.csv", "nivel": "extracto", "alcance": "", "nota": ""}
    return pl.DataFrame([{**base, **f} for f in filas], schema=ESQUEMA_METRICAS)


def reglas(**cambios: Any) -> Reglas:
    """``Reglas`` del ledger con los umbrales reales del protocolo; ``cambios`` reemplaza campos (por ejemplo ``nivel="completo"`` o ``z_95=1.0``)."""
    base = {
        "z_umbral": 3.0, "dif_relevante_pp": 0.5, "z_95": 1.959964, "celda_minima": 10, "anio_expuesto": 2024, "anios_resultado": (2019, 2020, 2021, 2022, 2023),
        "nivel": "extracto",
    }
    base.update(cambios)
    return Reglas(**base)


FILAS_POR_ANIO = {2019: 50_000, 2020: 50_000, 2021: 50_000, 2022: 50_000, 2023: 50_000, 2024: 50_000}


# ------------------------------------------------------------------------------------------------------------------------------ decisiones
class FuenteEvidenciaFalsa:
    """Implementa el protocolo ``decisiones.FuenteEvidencia`` con DataFrames hechos a mano.

    ``tabla`` devuelve ``None`` cuando la tabla no existe (el evaluador responde ``no_evaluable``). Los umbrales por defecto son los reales de ``config/umbrales.toml``.
    """

    def __init__(
        self, tablas: Mapping[str, pl.DataFrame] | None = None, umbrales: Mapping[str, Any] | None = None, metricas: Mapping[str, Mapping[str, Any]] | None = None,
        etiqueta: str = "extracto", cifras: Mapping[str, float] | None = None,
    ) -> None:
        self._tablas = dict(tablas or {})
        self._umbrales = {**umbrales_reales(), **(umbrales or {})}
        self._metricas = dict(metricas or {})
        self._cifras = dict(cifras or {})
        self.etiqueta = etiqueta

    def tabla(self, nombre: str, columnas_texto: Sequence[str] = ()) -> pl.DataFrame | None:
        return self._tablas.get(nombre)

    def umbral(self, nombre: str) -> Any:
        return self._umbrales[nombre]

    def metrica(self, clave: str) -> Mapping[str, Any] | None:
        return self._metricas.get(clave)

    def cifra_documentada(self, id_cifra: str) -> float | None:
        return self._cifras.get(id_cifra)


def umbrales_reales() -> dict[str, Any]:
    """Todos los umbrales de ``config/umbrales.toml`` por nombre, más las claves de la partición temporal (``anios_resultado``, ``anio_expuesto``...)."""
    u = config.cargar_umbrales()
    return {**{n: u[n] for n in u.nombres()}, **dict(u.particion)}


def df(*filas: Mapping[str, Any]) -> pl.DataFrame:
    """DataFrame desde filas (diccionarios); infiere el tipo con todas las filas para tolerar celdas nulas al principio."""
    return pl.DataFrame(list(filas), infer_schema_length=None)


def tablas_base() -> dict[str, pl.DataFrame]:
    """Tablas publicadas hechas a mano para las 26 hipótesis y los 15 supuestos, con valores parecidos a los del extracto real.

    Resultados esperados con ``metricas_base()`` y ``CIFRAS_BASE``: en un extracto, H01a matiza, H02a/b/c matizan, H03 respalda, H07 respalda, H08 matiza, H11 matiza,
    H14 respalda, H15b matiza y H17a/b matizan; las hipótesis que requieren los datos completos (H01b, H01c, H06a, H06b, H13 y H16a) quedan no evaluables. En modo
    completo, H01b y H01c respaldan, H13 contradice (la tasa de enlace no alcanza el mínimo) y H06a, H06b y H16a describen. Cada prueba cambia las celdas que le importan.
    """
    quiebres = [("no", ""), ("alza", ""), ("baja", "")]
    return {
        "t24_resumen.csv": df({"n_quiebres": 1, "quiebres": "2020 a 2021", "regimenes": "2019 a 2020; 2021 a 2024"}),
        "t23_historial_factibilidad.csv": df(
            {"anio": 2019, "regimen": "2019-2020", "factibilidad": 0.0, "cumple_minimo": "no"}, {"anio": 2020, "regimen": "2019-2020", "factibilidad": 0.971, "cumple_minimo": "si"},
            {"anio": 2021, "regimen": "2021-2023", "factibilidad": 0.0, "cumple_minimo": "no"}, {"anio": 2022, "regimen": "2021-2023", "factibilidad": 0.984, "cumple_minimo": "si"},
            {"anio": 2023, "regimen": "2021-2023", "factibilidad": 0.998, "cumple_minimo": "si"},
        ),
        "matriz_decision_variables.csv": df(
            {"variable": "N_EGRESOS_12M", "medicion_completa": "si", "incremento_auc": 0.02, "asociacion_hospital": 0.05},
            {"variable": "DIAS_DESDE_EGRESO_PREVIO", "medicion_completa": "si", "incremento_auc": 0.01, "asociacion_hospital": 0.10},
        ),
        "t23_proxy_hospital.csv": df(
            {"tipo_fila": "variable", "variable": "SERVICIOINGRESO", "asociacion_hospital": 0.256, "clase_proxy": "moderado", "n_niveles_totales": None, "n_niveles_en_un_hospital": None},
            {"tipo_fila": "variable", "variable": "ESPECIALIDAD_MEDICA", "asociacion_hospital": 0.167, "clase_proxy": "moderado", "n_niveles_totales": None, "n_niveles_en_un_hospital": None},
            {"tipo_fila": "cardinalidad_especialidad", "variable": "ESPECIALIDAD_MEDICA", "asociacion_hospital": None, "clase_proxy": None, "n_niveles_totales": 75, "n_niveles_en_un_hospital": 12},
            {"tipo_fila": "variable", "variable": "PREVISION", "asociacion_hospital": 0.094, "clase_proxy": "bajo", "n_niveles_totales": None, "n_niveles_en_un_hospital": None},
            {"tipo_fila": "variable", "variable": "NACIONALIDAD", "asociacion_hospital": 0.073, "clase_proxy": "bajo", "n_niveles_totales": None, "n_niveles_en_un_hospital": None},
            {"tipo_fila": "variable", "variable": "ES_EXTRANJERO", "asociacion_hospital": 0.265, "clase_proxy": "moderado", "n_niveles_totales": None, "n_niveles_en_un_hospital": None},
            {"tipo_fila": "variable", "variable": "PROVINCIA", "asociacion_hospital": 0.725, "clase_proxy": "alto", "n_niveles_totales": None, "n_niveles_en_un_hospital": None},
            {"tipo_fila": "variable", "variable": "COMUNA", "asociacion_hospital": 0.653, "clase_proxy": "alto", "n_niveles_totales": None, "n_niveles_en_un_hospital": None},
        ),
        "t23_fuga.csv": df(
            {"tipo_fila": "variable", "variable": "IR_29301_SEVERIDAD", "supera_a_basales": "si", "supera_umbral_fuga": "si", "prohibida_siempre": "no", "auc_ranking": 0.842, "auc_confirmacion": 0.858,
             "prop_igualdad": None, "ic_bajo": None, "ic_alto": None, "coincide_mayoria": None},
            {"tipo_fila": "variable", "variable": "IR_29301_MORTALIDAD", "supera_a_basales": "si", "supera_umbral_fuga": "si", "prohibida_siempre": "si", "auc_ranking": 0.860, "auc_confirmacion": 0.871,
             "prop_igualdad": None, "ic_bajo": None, "ic_alto": None, "coincide_mayoria": None},
            {"tipo_fila": "variable", "variable": "SERVICIOINGRESO", "supera_a_basales": "no", "supera_umbral_fuga": "no", "prohibida_siempre": "no", "auc_ranking": 0.797, "auc_confirmacion": 0.796,
             "prop_igualdad": None, "ic_bajo": None, "ic_alto": None, "coincide_mayoria": None},
            {"tipo_fila": "concordancia", "variable": "SERVICIO_INGRESO_IGUAL_ALTA", "supera_a_basales": None, "supera_umbral_fuga": None, "prohibida_siempre": None, "auc_ranking": None,
             "auc_confirmacion": None, "prop_igualdad": 0.992, "ic_bajo": 0.9914, "ic_alto": 0.9926, "coincide_mayoria": "si", "n_sin_traslado": 92445, "n_iguales_sin_traslado": 91706},
            {"tipo_fila": "concordancia_con_traslado", "variable": "SERVICIO_INGRESO_IGUAL_ALTA", "supera_a_basales": None, "supera_umbral_fuga": None, "prohibida_siempre": None,
             "auc_ranking": None, "auc_confirmacion": None, "prop_igualdad": 0.152, "ic_bajo": 0.148, "ic_alto": 0.157, "coincide_mayoria": "no", "n_con_traslado": 22696,
             "n_iguales_con_traslado": 3454},
        ),
        "t23_resumen_hospitales.csv": df(
            {"autorref_prop_exacta": 0.3064, "autorref_supera_maximo": "si", "enlace_anio": 2023, "enlace_ventana_dias": 30, "enlace_tasa": 0.0283, "enlace_cumple_minimo": "no"}
        ),
        "t22_deriva_categorias.csv": df({"columna": "ETNIA", "anio": 2019, "categoria": "AYMARA"}, {"columna": "ETNIA", "anio": 2019, "categoria": "MAPUCHE"},
                                        {"columna": "ETNIA", "anio": 2020, "categoria": "MAPUCHE"}, {"columna": "PREVISION", "anio": 2019, "categoria": "FONASA A"}),
        # ``prevalencia`` es sobre todos los egresos de la variante y ``prevalencia_conocida`` sobre los de desenlace conocido; ``dif_prevalencia_pp`` de B es B menos A en esta última
        "t23_cascada_variantes.csv": df(
            {"periodo": "2019-2023", "variante": "A", "prevalencia": 0.0341, "prevalencia_conocida": 0.0364, "dif_prevalencia_pp": None},
            {"periodo": "2019-2023", "variante": "B", "prevalencia": 0.0402, "prevalencia_conocida": 0.0431, "dif_prevalencia_pp": 0.67},
            {"periodo": "2019", "variante": "A", "prevalencia": 0.0276, "prevalencia_conocida": 0.0301, "dif_prevalencia_pp": None},
        ),
        "t23_hospitales_resumen.csv": df(
            {"anio": 2022, "n_hospitales": 65, "n_pediatricos": 3, "n_hospitales_con_cohorte": 64, "prop_celdas_bajo_minimo": 0.031},
            {"anio": 2023, "n_hospitales": 68, "n_pediatricos": 3, "n_hospitales_con_cohorte": 67, "prop_celdas_bajo_minimo": 0.030},
            {"anio": 2024, "n_hospitales": 72, "n_pediatricos": 3, "n_hospitales_con_cohorte": None, "prop_celdas_bajo_minimo": None},
        ),
        "t23_volumen_umbrales.csv": df(
            {"anio": 2022, "egresos_min": 500, "muertes_esperadas_min": 10, "n_hospitales": 64, "n_cumplen_ambos": 22},
            {"anio": 2023, "egresos_min": 500, "muertes_esperadas_min": 10, "n_hospitales": 67, "n_cumplen_ambos": 23},
            {"anio": 2023, "egresos_min": 200, "muertes_esperadas_min": 5, "n_hospitales": 67, "n_cumplen_ambos": 41},
        ),
        "t23_cola_estancia.csv": df(
            {"base": "E", "regla": "p99", "corte_dias": 51, "cumple_def": "si", "cumple_conf": "si", "propuesta": "si", "prop_afectados_def": 0.0097},
            {"base": "E", "regla": "p99,5", "corte_dias": 66, "cumple_def": "si", "cumple_conf": "si", "propuesta": "no", "prop_afectados_def": 0.0049},
        ),
        "t23_resumen_seleccion.csv": df(
            {"elix_top5": "ELIX_24; ELIX_04; ELIX_25; ELIX_30; ELIX_27", "elix_n_coinciden_fase1": 3, "elix_n_coinciden_repo": 2, "h17a_n_clave": 47, "h17a_n_psi_alto": 4}
        ),
        "t23_estancia_dispersion.csv": df({"periodo": "2019-2023", "base": "E", "indice_dispersion": 21.07, "media": 7.31}, {"periodo": "2019", "base": "E", "indice_dispersion": 42.25, "media": 7.18}),
        "t23_resumen.csv": df({"n_cero_total": 43354, "prop_cero_ambulatoria": 0.757, "prop_cero_fallecidos": 0.0134, "prop_cero_censurados": 0.0273, "n_meses_evaluados": 48,
                               "n_meses_alza": 11, "n_meses_baja": 11, "intento_2024": "rechazado"}),
        "t23_serie_mensual.csv": df({"mes_id": "2020-01", "quiebre": "no"}, {"mes_id": "2020-02", "quiebre": "alza"}, {"mes_id": "2020-03", "quiebre": "baja"}, {"mes_id": "2019-01", "quiebre": None}),
        "t23_resultado_estratos.csv": df(
            {"periodo": "2019-2023", "variable": "actividad_grupo", "categoria": "ambulatoria", "razon_vs_resto": 0.109, "bajo_umbral": "si"},
            {"periodo": "2019-2023", "variable": "actividad_grupo", "categoria": "hospitalización", "razon_vs_resto": 9.146, "bajo_umbral": None},
            {"periodo": "2019-2023", "variable": "estancia_cero", "categoria": "0 días", "razon_vs_resto": 0.320, "bajo_umbral": "no"},
            {"periodo": "2019-2023", "variable": "estancia_cero", "categoria": "1 día o más", "razon_vs_resto": 3.124, "bajo_umbral": None},
        ),
        "t23_enlaces_traslado.csv": df(
            {"periodo": "2019-2023", "direccion": "entrante", "ventana_dias": 30, "n_elegibles": 14939, "tasa_enlace": 0.0071},
            {"periodo": "2019-2023", "direccion": "saliente", "ventana_dias": 30, "n_elegibles": 5017, "tasa_enlace": 0.0259},
            {"periodo": "2019-2023", "direccion": "entrante", "ventana_dias": 7, "n_elegibles": 14939, "tasa_enlace": 0.003},
        ),
        "t23_resultado_entrantes.csv": df(
            {"periodo": "2019-2023", "entrante": "si", "mortalidad": 0.0753, "rr": 1.856, "rr_bajo": 1.738, "rr_alto": 1.982},
            {"periodo": "2019-2023", "entrante": "no", "mortalidad": 0.0406, "rr": None, "rr_bajo": None, "rr_alto": None},
        ),
        # --- supuestos
        "t23_elix_vs_3c.csv": df(*({"elix_id": f"ELIX_{i:02d}", "supera_umbral": "si" if i <= 11 else "no"} for i in range(1, 32))),
        "t24_tipoalta_etiquetas.csv": df({"etiqueta": "DOMICILIO", "metodo": "exacta", "n": 224254}, {"etiqueta": "FALLECIDO", "metodo": "exacta", "n": 7748},
                                         {"etiqueta": "OTRA", "metodo": "por_prefijo", "n": 120}, {"etiqueta": "RARA", "metodo": "sin_rol", "n": 5}),
        "t22_cardinalidades.csv": df(
            *({"columna": c, "periodo": p, "n_observado": o, "chao1": ch, "documentado": d}
              for c, p, o, ch, d in (("ESPECIALIDAD_MEDICA", "2019-2024", 90, 90.33, 158), ("PREVISION", "2019-2024", 15, 15.0, 19), ("TIPO_PROCEDENCIA", "2019-2024", 19, 19.0, 25),
                                     ("TIPOALTA", "2019-2023", 11, 11.0, 18), ("DIAGNOSTICO1", "2019-2024", 6178, 7500.28, 9500), ("DIAGNOSTICO1", "2019", 3783, 5321.85, 9500)))
        ),
        "t21_deriva_formato.csv": df(
            {"anio": 2022, "columna": "FECHA_INGRESO", "clase": "ISO", "n_clase": 49999}, {"anio": 2022, "columna": "FECHA_INGRESO", "clase": "DD-MM-AAAA", "n_clase": 0},
            {"anio": 2023, "columna": "FECHA_INGRESO", "clase": "ISO", "n_clase": 0}, {"anio": 2023, "columna": "FECHA_INGRESO", "clase": "DD-MM-AAAA", "n_clase": 49999},
            {"anio": 2023, "columna": "CIP_ENCRIPTADO", "clase": "SOLO_DIGITOS", "n_clase": 49000},
        ),
        "t24_resumen_catalogos.csv": df({"n_codigos_hospital_distintos": 72, "n_codigos_hospital_fuera_catalogo": 1}),
        # la métrica de la Tarea 5 usa el grupo "todos los IDs": los otros grupos son subconjuntos y no entran en la base de S01
        "t24_id_consistencia.csv": df(
            {"regimen": "2019-2024", "grupo_id": "todos los IDs", "n_ids_con_dos_o_mas_episodios": 8975, "n_ids_nacimiento_inconsistente": 163,
             "n_ids_con_sexo_en_dos_o_mas_episodios": 8970, "n_ids_sexo_inconsistente": 65},
            {"regimen": "2019-2024", "grupo_id": "IDs de largo 9", "n_ids_con_dos_o_mas_episodios": 100, "n_ids_nacimiento_inconsistente": 12,
             "n_ids_con_sexo_en_dos_o_mas_episodios": 100, "n_ids_sexo_inconsistente": 3},
        ),
        "t23_cascada.csv": df(*(
            {"periodo": per, "paso": paso, "n": n} for per, pasos in (("2019", (49999, 17, 1142, 3809, 45031, 8039, 36992, 8953, 28039, 1351, 26688, 1059, 974, 24655)),)
            for paso, n in zip(("TOTAL", "EX01", "EX02", "EX03", "A", "EX04", "B", "EX05", "C", "EX06", "D", "EX07", "EX08", "E"), pasos)
        )),
        "t22_bloque_rn.csv": df({"columna": "RN2ESTADO", "n_informados": 67123, "prop_modal": 0.993}, {"columna": "RN3ESTADO", "n_informados": 66764, "prop_modal": 0.99994},
                                {"columna": "RN4ESTADO", "n_informados": 66759, "prop_modal": 1.0}),
        "t22_conciliacion.csv": df(
            {"columna": "A", "estado": "COINCIDE", "n_filas_med": 1000, "n_sin_med": 100, "n_centinela_med": 50, "pct_sin_doc": 10.0, "pct_sin_med": 10.0},     # sigue el conteo sin centinelas
            {"columna": "B", "estado": "DISCREPANCIA", "n_filas_med": 1000, "n_sin_med": 100, "n_centinela_med": 50, "pct_sin_doc": 15.0, "pct_sin_med": 10.0},   # sigue el conteo con centinelas
            {"columna": "C", "estado": "COINCIDE", "n_filas_med": 1000, "n_sin_med": 100, "n_centinela_med": 0, "pct_sin_doc": 10.0, "pct_sin_med": 10.0},      # sin centinelas: no cuenta
            {"columna": "D", "estado": "COINCIDE", "n_filas_med": 1000, "n_sin_med": 100, "n_centinela_med": 80, "pct_sin_doc": 10.0, "pct_sin_med": 10.0},
        ),
        "t24_duplicados.csv": df({"anio": 2019, "n_filas_duplicadas_clave": 12, "n_solapes": None, "n_duplicados_exactos_por_id_y_fechas": 16},
                                 {"anio": 2020, "n_filas_duplicadas_clave": None, "n_solapes": 34, "n_duplicados_exactos_por_id_y_fechas": None}),
    }


def metricas_base(nivel: str = "extracto") -> dict[str, dict[str, Any]]:
    """Métricas hechas a mano para H11 y para los supuestos S01, S03, S07, S08, S11 y S14."""
    return {
        "calidad.valores_rut.total": metrica(0, nivel=nivel),
        "id.consistencia.nacimiento_inconsistente_prop": metrica(0.018162, nivel=nivel),
        "id.consistencia.sexo_inconsistente_prop": metrica(0.007242, nivel=nivel),
        "calidad.cie10_principal_valido_pct": metrica(99.999332, nivel=nivel),
        "estructura.n_anios_id_beneficiario": metrica(1, nivel=nivel),
        "estructura.primer_anio_id_beneficiario": metrica(2024, nivel=nivel),
        "hospitales.n_distintos.maximo": metrica(72, nivel=nivel),
        "codificacion.aumento_pct_2019_2024": metrica(40.44, ic=(38.66, 42.24), alcance="2019-2024", nivel=nivel),
        "calidad.alta_antes_ingreso.total": metrica(0, nivel=nivel),
        "calidad.edad_negativa.total": metrica(None, nota="[suprimido: menos de diez casos]", nivel=nivel),
        "calidad.edad_mayor_110.total": metrica(None, nota="[suprimido: menos de diez casos]", nivel=nivel),
    }


CIFRAS_BASE = {"C077": 65.0, "C078": 68.0}


def fuente_base(etiqueta: str = "extracto", **cambios: Any) -> FuenteEvidenciaFalsa:
    """``FuenteEvidenciaFalsa`` con las tablas, métricas y cifras de ``tablas_base``; ``cambios`` reemplaza tablas (``nombre_csv=DataFrame``) con la clave ``tablas`` o métricas
    con la clave ``metricas`` (diccionarios que se agregan)."""
    tablas = {**tablas_base(), **cambios.get("tablas", {})}
    metricas = {**metricas_base(etiqueta), **cambios.get("metricas", {})}
    return FuenteEvidenciaFalsa(tablas, metricas=metricas, etiqueta=etiqueta, cifras=CIFRAS_BASE)


# ------------------------------------------------------------------------------------------------------------------------------- contextos
def copiar_tablas(ctx: Contexto, destino: Path | str, *, con_figuras: bool = False, etiqueta: str | None = None) -> Contexto:
    """Contexto nuevo (en ``destino``) con una copia de las tablas, el índice de tablas y las métricas de ``ctx``.

    Sirve para que una prueba escriba tablas sin tocar el contexto de solo lectura ``pipeline_sintetico``. Con ``con_figuras`` copia también los PNG y el índice de figuras.
    Con ``etiqueta="completo"`` el contexto nuevo es de modo completo (la fuente sintética se reetiqueta; las tablas copiadas no cambian).
    """
    etiqueta = etiqueta or ctx.etiqueta
    fuente_nueva = ctx.fuente if etiqueta == ctx.fuente.etiqueta else dataclasses.replace(ctx.fuente, etiqueta=etiqueta)
    nuevo = Contexto.crear(etiqueta, fuente=fuente_nueva, dir_salida_base=Path(destino))
    for ruta in ctx.dir_tablas.glob("*.csv"):
        shutil.copy2(ruta, nuevo.dir_tablas / ruta.name)
    nuevo._tablas = {k: dict(v) for k, v in ctx._tablas.items()}
    nuevo._metricas = [dict(m) for m in ctx._metricas]
    if con_figuras:
        for ruta in ctx.dir_figuras.glob("*.png"):
            shutil.copy2(ruta, nuevo.dir_figuras / ruta.name)
        nuevo.figuras._entradas = dict(ctx.figuras._entradas)
    return nuevo
