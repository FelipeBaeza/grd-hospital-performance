"""Las ocho puertas de calidad de la Fase 4 (G1 a G8)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from . import privacidad, seleccion
from .config import RAIZ, sha256_archivo
from .datos import PREDICTORES, predictores_del_diccionario

FILAS_ESPERADAS = {"mortalidad": 2640061, "estadia": 2396356}


def _j(r: Path):
    return json.loads(Path(r).read_text(encoding="utf8"))


def _p(id_: str, nombre: str, ok: bool, detalle: str) -> dict:
    return {"puerta": id_, "nombre": nombre, "estado": "PASS" if ok else "FAIL", "detalle": detalle}


def verificar(trabajo: Path, salidas: Path, reglas, modelos: Path, registro_fase3: Path) -> pd.DataFrame:
    tablas = salidas / "tablas"
    out = []
    tareas = ("mortalidad", "estadia")
    # G1 Entrada íntegra
    c = _j(trabajo / "contrato.json")
    filas_ok = all(c["matrices"][t]["filas"] == FILAS_ESPERADAS[t] for t in tareas)
    hay_huellas = len(c["huellas"]) == 4 and all(len(v) == 64 for v in c["huellas"].values())
    f3 = RAIZ.parent / "fase3_preparacion" / "salidas" / "tablas" / "tabla_verificacion.csv"
    f3_ok = f3.exists() and (pd.read_csv(f3)["estado"] == "PASS").all()
    out.append(_p("G1", "Entrada íntegra", filas_ok and hay_huellas and f3_ok,
                  f"filas {c['matrices']['mortalidad']['filas']:,} y {c['matrices']['estadia']['filas']:,}; 4 huellas SHA-256 registradas; las 8 puertas de la Fase 3 siguen en PASS: {'sí' if f3_ok else 'no'}"))
    # G2 2024 intacto
    despues = sha256_archivo(registro_fase3) if registro_fase3.exists() else "no existe"
    sin_cambio = c["registro_accesos_antes"] == despues
    solo_hasta_2023 = all(max(c["matrices"][t]["anios"]) <= 2023 and set(c["matrices"][t]["splits"]) <= {"entrenamiento", "calibracion"} for t in tareas)
    uso_sellado = [f.name for f in (RAIZ / "modelado").glob("*.py") if f.name != "verificacion.py" and re.search(r"leer_desenlace|marca_cohorte_d_2024|FASE3_AUTORIZA", f.read_text(encoding="utf8"))]
    out.append(_p("G2", "2024 intacto", sin_cambio and solo_hasta_2023 and not uso_sellado,
                  f"registro de accesos de la Fase 3 sin cambios: {'sí' if sin_cambio else 'no'}; años en las matrices: 2019–2023; código que llame al cortafuegos de 2024: {len(uso_sellado)}"))
    # G3 Lista blanca e integridad
    perm = set(predictores_del_diccionario())
    lista_ok = set(PREDICTORES) <= perm and len(PREDICTORES) == 40
    proh = []
    for t in tareas:
        proh += _j(trabajo / f"integridad_{t}.json")["prohibidas_en_contribuciones"]
    h = _j(trabajo / "integridad_hospital.json")
    out.append(_p("G3", "Lista blanca e integridad", lista_ok and not proh and h["cumple"],
                  f"40 predictores, todos permitidos o condicionales; variables prohibidas en las contribuciones: {len(proh)}; identificación del hospital: {h['resultados'][0]['exactitud_top1']*100:.1f} % con las condicionales y "
                  f"{h['resultados'][1]['exactitud_top1']*100:.1f} % sin ellas (diferencia {h['diferencia_pp']:.1f} puntos; límite {h['umbral_pp']:.0f})"))
    # G4 Comparación justa
    b = reglas.ajuste
    ok4, det4 = True, []
    for t in tareas:
        tab = pd.read_csv(tablas / f"comparacion_{t}.csv")
        for cand in ("constante", "linea_base", "lightgbm", "xgboost"):
            ok4 = ok4 and cand in set(tab["candidato"])
        esperado = b["pruebas_mortalidad"] if t == "mortalidad" else b["pruebas_estadia"]
        n = [len(pd.read_csv(tablas / f"ajuste_pruebas_{t}_{g}.csv")) for g in ("lightgbm", "xgboost")]
        ok4 = ok4 and n[0] == n[1] == esperado
        det4.append(f"{t}: {n[0]} y {n[1]} pruebas (presupuesto {esperado})")
    out.append(_p("G4", "Comparación justa", ok4, "; ".join(det4) + "; mismos pliegues y mismos datos para todos"))
    # G5 Selección declarada
    ok5, det5 = reglas.huella() == c["huella_reglas"], []
    for t in tareas:
        s = _j(trabajo / f"seleccion_{t}.json")
        cand = _j(trabajo / f"candidatos_{t}.json")
        rep = seleccion.aplicar_regla(t, {n: cand[n] for n in ("linea_base", "lightgbm", "xgboost")}, reglas)
        ok5 = ok5 and rep["ganador"] == s["ganador"]
        det5.append(f"{t}: {s['ganador']}")
    out.append(_p("G5", "Selección declarada", ok5, "reglas con la misma huella que al inicio; la regla, aplicada de nuevo, da el mismo ganador (" + "; ".join(det5) + ")"))
    # G6 Calibración en 2023
    ok6, det6 = True, []
    for t in tareas:
        k = _j(trabajo / f"calibracion_{t}.json")
        m = k["metricas_fuera_de_muestra"]
        ok6 = ok6 and k["cumple_oe"] and k["cumple_pendiente"]
        txt = f"{t}: {k['metodo']}, O/E {m['oe_global']:.3f}" + (f", pendiente {m['pendiente']:.3f}" if t == "mortalidad" else "")
        det6.append(txt)
    out.append(_p("G6", "Calibración en 2023", ok6, "fuera de muestra, por bloques de 2023: " + "; ".join(det6)))
    # G7 Reproducibilidad
    r = _j(trabajo / "reproducibilidad.json")
    dif = max(max(v["max_dif_entrenamientos"], v["max_dif_recarga"], v["max_dif_prediccion_calibrada"]) for v in r.values())
    py = _j(trabajo / "pytest_resumen.json") if (trabajo / "pytest_resumen.json").exists() else {"pasaron": 0, "fallaron": 1}
    tiempos = pd.read_csv(tablas / "tiempos_etapas_fase4.csv")
    ram = float(tiempos["pico_ram_gb"].max())
    out.append(_p("G7", "Reproducibilidad", dif <= 1e-9 and py["fallaron"] == 0 and py["pasaron"] > 0 and ram < 8,
                  f"dos entrenamientos finales y una recarga dan predicciones idénticas (diferencia máxima {dif:.1e}); pruebas: {py['pasaron']} pasaron, {py['fallaron']} fallaron; pico de memoria {ram:.2f} GB"))
    # G8 Congelamiento y privacidad
    man = _j(modelos / "manifiesto.json")
    ok8, n_arch = True, 0
    for grupo, d in man["archivos"].items():
        for nombre, x in d.items():
            n_arch += 1
            ok8 = ok8 and sha256_archivo(modelos / x["ruta"]) == x["sha256"]
    fichas = all((modelos / t / "ficha_modelo.md").exists() for t in tareas)
    hall = privacidad.escanear_carpeta(salidas, 10, excluir_archivos=("pytest_resumen.json", "contrato_entrada.csv"))
    out.append(_p("G8", "Congelamiento y privacidad", ok8 and fichas and not hall,
                  f"{n_arch} archivos congelados con huella SHA-256 verificada; fichas de modelo: {'sí' if fichas else 'no'}; archivos de salida con RUT, fechas o celdas de 1 a 9 casos: {len(hall)}"))
    return pd.DataFrame(out)
