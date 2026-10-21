"""Regla de selección y de parsimonia, escrita en reglas_fase4.toml antes de entrenar: gana el más simple que rinda lo mismo."""
from __future__ import annotations

import math

from .config import Reglas


def _mejora(tarea: str, incumbente: float, retador: float, reglas: Reglas) -> tuple[float, float, bool]:
    s = reglas.seleccion
    if tarea == "mortalidad":
        mejora = retador - incumbente                              # PR AUC: mayor es mejor
        return mejora, s["mortalidad_mejora_minima_abs"], mejora >= s["mortalidad_mejora_minima_abs"]
    mejora = (incumbente - retador) / incumbente                   # devianza: menor es mejor
    return mejora, s["estadia_mejora_minima_rel"], mejora >= s["estadia_mejora_minima_rel"]


def calibracion_no_peor(tarea: str, inc_2023: dict, ret_2023: dict, reglas: Reglas) -> tuple[bool, dict]:
    s = reglas.seleccion
    d_oe_inc, d_oe_ret = abs(math.log(inc_2023["oe_global"])), abs(math.log(ret_2023["oe_global"]))
    ok = d_oe_ret <= d_oe_inc + s["tolerancia_ln_oe_2023"]
    detalle = {"ln_oe_incumbente": d_oe_inc, "ln_oe_retador": d_oe_ret}
    if tarea == "mortalidad":
        d_p_inc, d_p_ret = abs(1 - inc_2023["pendiente"]), abs(1 - ret_2023["pendiente"])
        ok = ok and d_p_ret <= d_p_inc + s["tolerancia_pendiente_2023"]
        detalle.update({"dist_pendiente_incumbente": d_p_inc, "dist_pendiente_retador": d_p_ret})
    return ok, detalle


def aplicar_regla(tarea: str, candidatos: dict[str, dict], reglas: Reglas) -> dict:
    """`candidatos[nombre] = {"score": float, "metricas_2023": dict}`. Recorre el orden de simple a complejo; el retador reemplaza al incumbente solo si
    mejora lo suficiente en la métrica de selección y su calibración en 2023 (antes de recalibrar) no es peor."""
    orden = [n for n in reglas.seleccion["orden"] if n in candidatos]
    ganador = orden[0]
    pasos = []
    for retador in orden[1:]:
        mejora, minimo, suficiente = _mejora(tarea, candidatos[ganador]["score"], candidatos[retador]["score"], reglas)
        cal_ok, det = calibracion_no_peor(tarea, candidatos[ganador]["metricas_2023"], candidatos[retador]["metricas_2023"], reglas)
        reemplaza = suficiente and cal_ok
        pasos.append({"incumbente": ganador, "retador": retador, "mejora": mejora, "mejora_minima": minimo, "mejora_suficiente": suficiente,
                      "calibracion_no_peor": cal_ok, **det, "reemplaza": reemplaza})
        if reemplaza:
            ganador = retador
    return {"tarea": tarea, "ganador": ganador, "pasos": pasos}
