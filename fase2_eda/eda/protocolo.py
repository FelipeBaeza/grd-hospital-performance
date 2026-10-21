"""Validación del protocolo congelado en la Tarea 1 y tabla de cobertura decisiones x análisis."""

from __future__ import annotations

import polars as pl

from . import config

TAREAS_ANALISIS = tuple(range(3, 13))
VALORES_DEPENDE = ("si", "parcial", "no")


def _num_decisiones(decisiones: pl.DataFrame) -> list[int]:
    return sorted(int(x) for x in decisiones["n"].to_list())


def _afectadas(texto: str) -> set[int]:
    return {int(p.strip()) for p in texto.replace(",", ";").split(";") if p.strip().isdigit()}


def validar_hipotesis(hipotesis: pl.DataFrame, decisiones: pl.DataFrame) -> list[str]:
    """Problemas del protocolo de hipótesis (lista vacía si todo está bien)."""
    problemas: list[str] = []
    obligatorias = {"id", "decision", "pregunta", "analisis", "tarea", "criterio_decision", "depende_de_datos"}
    faltan = obligatorias - set(hipotesis.columns)
    if faltan:
        return [f"faltan columnas en hipotesis.csv: {sorted(faltan)}"]

    ids = hipotesis["id"].to_list()
    if len(ids) != len(set(ids)):
        problemas.append("hay ids de hipótesis repetidos")

    for fila in hipotesis.iter_rows(named=True):
        hid = fila["id"]
        dep = fila["depende_de_datos"]
        if dep not in VALORES_DEPENDE:
            problemas.append(f"{hid}: depende_de_datos debe ser si, parcial o no (es {dep!r})")
        for campo in ("pregunta", "analisis", "criterio_decision"):
            if not fila[campo].strip():
                problemas.append(f"{hid}: campo vacío {campo}")
        tarea = fila["tarea"].strip()
        if dep in ("si", "parcial"):
            if not tarea.isdigit() or int(tarea) not in TAREAS_ANALISIS:
                problemas.append(f"{hid}: debe apuntar a una tarea entre 3 y 12 (tiene {tarea!r})")
        elif tarea and (not tarea.isdigit() or int(tarea) not in TAREAS_ANALISIS):
            problemas.append(f"{hid}: tarea fuera de 3 a 12 ({tarea!r})")

    nums = set(_num_decisiones(decisiones))
    primarias: dict[int, list[str]] = {}
    for fila in hipotesis.iter_rows(named=True):
        d = fila["decision"].strip()
        if d.isdigit():
            primarias.setdefault(int(d), []).append(fila["depende_de_datos"])
            if int(d) not in nums:
                problemas.append(f"{fila['id']}: la decisión {d} no existe en decisiones_pendientes.csv")
    for n in sorted(nums):
        if n not in primarias:
            problemas.append(f"la decisión {n} no tiene hipótesis ni marca 'no depende de datos'")
    return problemas


def tabla_cobertura(hipotesis: pl.DataFrame, decisiones: pl.DataFrame) -> pl.DataFrame:
    """Matriz decisiones x tareas de análisis con el número de hipótesis que las cubren."""
    filas = []
    for d in decisiones.iter_rows(named=True):
        n = int(d["n"])
        cuenta = {f"t{t:02d}": 0 for t in TAREAS_ANALISIS}
        deps: list[str] = []
        ids: list[str] = []
        for h in hipotesis.iter_rows(named=True):
            primaria = h["decision"].strip() == str(n)
            afectada = n in _afectadas(h.get("decision_afectada", ""))
            if not (primaria or afectada):
                continue
            ids.append(h["id"])
            if primaria:
                deps.append(h["depende_de_datos"])
            t = h["tarea"].strip()
            if t.isdigit() and int(t) in TAREAS_ANALISIS:
                cuenta[f"t{int(t):02d}"] += 1
        if "si" in deps:
            dep = "si"
        elif "parcial" in deps:
            dep = "parcial"
        else:
            dep = "no"
        filas.append(
            {
                "decision": n,
                "tema": d["tema"],
                "depende_de_datos": dep,
                "n_hipotesis": len(ids),
                "hipotesis": "; ".join(ids),
                **cuenta,
            }
        )
    return pl.DataFrame(filas).sort("decision")


def validar_plan_figuras(plan: pl.DataFrame) -> list[str]:
    problemas: list[str] = []
    for col in ("id_figura", "tarea", "proposito", "tabla_csv"):
        if col not in plan.columns:
            return [f"falta la columna {col} en plan_figuras.csv"]
    ids = plan["id_figura"].to_list()
    if len(ids) != len(set(ids)):
        problemas.append("hay figuras planificadas repetidas")
    for fila in plan.iter_rows(named=True):
        if not fila["tarea"].isdigit():
            problemas.append(f"{fila['id_figura']}: tarea inválida")
        if not fila["proposito"].strip():
            problemas.append(f"{fila['id_figura']}: sin propósito")
        if not fila["tabla_csv"].strip():
            problemas.append(f"{fila['id_figura']}: sin CSV de respaldo")
    return problemas


def validar_definicion_terminado(df: pl.DataFrame) -> list[str]:
    problemas: list[str] = []
    for col in ("id", "criterio", "como_se_verifica", "verificador"):
        if col not in df.columns:
            return [f"falta la columna {col} en definicion_terminado.csv"]
    for fila in df.iter_rows(named=True):
        for col in ("criterio", "como_se_verifica", "verificador"):
            if not fila[col].strip():
                problemas.append(f"{fila['id']}: campo vacío {col}")
    return problemas


def validar_todo() -> dict[str, list[str]]:
    """Valida los cuatro archivos de protocolo con las rutas por defecto."""
    u = config.cargar_umbrales()
    return {
        "umbrales": config.validar_umbrales(u),
        "hipotesis": validar_hipotesis(config.cargar_hipotesis(), config.cargar_decisiones_fase1()),
        "plan_figuras": validar_plan_figuras(config.cargar_plan_figuras()),
        "definicion_terminado": validar_definicion_terminado(config.cargar_definicion_terminado()),
    }
