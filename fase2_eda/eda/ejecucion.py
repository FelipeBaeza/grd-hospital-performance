"""Utilidades de ejecución compartidas por run_fase2_eda.py y las pruebas: manifiesto de la corrida."""

from __future__ import annotations

from pathlib import Path

from . import manifiesto, rutas
from .contexto import Contexto

ARCHIVOS_DOCS = (
    "fase1_comprension_dominio/decisiones_pendientes.csv",
    "fase1_comprension_dominio/reglas_variables.csv",
    "fase1_comprension_dominio/tipoalta_roles.csv",
    "fase1_comprension_dominio/matriz_trazabilidad.csv",
    "fase1_comprension_dominio/config/fase1_spec.json",
    "fase1_comprension_dominio/Fase_1__Comprensión_del_dominio_-_Categorias_de_Selección.csv",
    "fase1_comprension_dominio/Fase_1__Comprensión_del_dominio_-_Decisiones_Metodológicas.csv",
    "fase1_comprension_dominio/Fase_1__Comprensión_del_dominio_-_Reglas_del_Modelo.csv",
    "fase1_comprension_dominio/Fase_1__Comprensión_del_dominio_-_Requisitos_del_Modelo.csv",
    "fase2_comprension_datos/Fase_2__Diccionario_de_Base_de_Datos.csv",
    "fase2_comprension_datos/Fase_2__Problemas_de_Calidad_de_Datos.csv",
    "fase2_comprension_datos/Fase_2__Variables_Derivadas.csv",
)
CATALOGOS_RAIZ = (
    "CIE-10.csv",
    "CIE-9 .csv",
    "GRDs_T10901.xlsx",
    "Hospitales.csv",
    "IR-GRD.csv",
    "Severidad GRD.csv",
    "Mortalidad GRD.csv",
    "Tipo de Pabellon.csv",
)


def archivos_insumo(raiz_proyecto: Path | None = None) -> list[Path]:
    """Documentos de las Fases 1 y 2 y catálogos de la raíz que alimentan el análisis (si existen)."""
    raiz = raiz_proyecto or rutas.RAIZ_PROYECTO
    cand = [raiz / "docs" / a for a in ARCHIVOS_DOCS] + [raiz / c for c in CATALOGOS_RAIZ]
    cand += sorted(raiz.glob("Oficio_Respuesta_N*23364_T10901.pdf"))
    return [p for p in cand if p.is_file()]


def archivos_config() -> list[Path]:
    return manifiesto.archivos_de(rutas.DIR_CONFIG)


def archivos_codigo() -> list[Path]:
    codigo = manifiesto.archivos_de(rutas.RAIZ_EDA / "eda", "*.py")
    runner = rutas.RAIZ_EDA / "run_fase2_eda.py"
    if runner.is_file():
        codigo.append(runner)
    return codigo


def archivos_salida(ctx: Contexto) -> list[Path]:
    return [p for p in manifiesto.archivos_de(ctx.dir_salida) if p.name != "manifiesto.json"]


def resumen_accesos(ctx: Contexto) -> dict:
    reg = ctx.cortafuegos.registro()
    por_decision = {k: int(v) for k, v in reg.group_by("decision").len().iter_rows()} if reg.height else {}
    return {
        "anio_expuesto": ctx.cortafuegos.anio_expuesto,
        "por_decision": dict(sorted(por_decision.items())),
        "bloqueos": ctx.cortafuegos.n_bloqueos(),
        "lecturas_en_claro_2024_por_reglas_autorizadas": ctx.cortafuegos.n_lecturas_en_claro_expuestas(),
        "violaciones_de_politica": ctx.cortafuegos.violaciones(),
    }


def escribir_manifiesto_corrida(ctx: Contexto, extras: dict | None = None, raiz_proyecto: Path | None = None) -> Path:
    """Escribe ``manifiesto.json`` con huellas de insumos, configuración, código y salidas."""
    m = manifiesto.construir_manifiesto(
        fuente=ctx.etiqueta,
        escaneos=ctx.lector.escanear_todo(),
        semilla=ctx.semilla,
        archivos_config=archivos_config() + archivos_insumo(raiz_proyecto),
        archivos_codigo=archivos_codigo(),
        archivos_salida=archivos_salida(ctx),
        resumen_accesos=resumen_accesos(ctx),
        extras=extras,
    )
    return manifiesto.escribir_manifiesto(m, ctx.dir_salida / "manifiesto.json")
