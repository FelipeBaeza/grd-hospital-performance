"""Puertas de la preparación de la Fase 5 (antes de abrir 2024). Las ocho puertas de cierre de la fase (G1–G8) se verifican al terminar la evaluación."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import apertura, huellas, privacidad
from .config import DIR_MODELOS_F4, DIR_PLANES_B, DIR_SALIDAS, RAIZ, REGISTRO_FASE3, sha256_archivo


def _p(id_, nombre, ok, detalle):
    return {"puerta": id_, "nombre": nombre, "estado": "PASS" if ok else "FAIL", "detalle": detalle}


def verificar_preparacion(prot, trabajo: Path, con_pytest: bool = True) -> pd.DataFrame:
    out = []
    pre = json.loads((RAIZ / "documentos" / "registro_preapertura.json").read_text(encoding="utf8"))
    # P1 modelos intactos
    r = huellas.verificar_manifiesto(DIR_MODELOS_F4)
    rb = huellas.verificar_manifiesto(DIR_PLANES_B)
    out.append(_p("P1", "Modelos y planes B intactos", all(v[0] for v in r.values()) and all(v[0] for v in rb.values()), f"{len(r)} archivos de la Fase 4 y {len(rb)} de los planes B con huella SHA-256 igual a su manifiesto"))
    # P2 protocolo registrado
    out.append(_p("P2", "Protocolo pre-registrado", pre["huella_protocolo"] == prot.huella(), f"huella del protocolo registrada antes de la apertura: {pre['huella_protocolo'][:16]}…"))
    # P3 2024 sin abrir
    n = apertura.lecturas_concedidas(REGISTRO_FASE3)
    igual = pre["huellas_congeladas"]["registro_accesos_fase3"] == (sha256_archivo(REGISTRO_FASE3) if REGISTRO_FASE3.exists() else "no existe")
    out.append(_p("P3", "2024 sin abrir", n == 0 and igual, f"lecturas concedidas del desenlace de 2024 en el registro: {n}; registro sin cambios desde el pre-registro: {'sí' if igual else 'no'}"))
    # P4 cohortes reproducen 2023
    e = json.loads((DIR_SALIDAS / "ensayo_2023" / "resumen_ensayo.json").read_text(encoding="utf8"))["cuadra_con_fase3"]
    ok4 = e["mortalidad"][0] == e["mortalidad"][1] and e["estadia"][0] == e["estadia"][1] and e["mismos_ids_mortalidad"] and e["mismos_ids_estadia"]
    out.append(_p("P4", "Cohortes D y E reproducen 2023", ok4, f"mortalidad {e['mortalidad'][0]:,} y estadía {e['estadia'][0]:,} egresos, con los mismos identificadores que las matrices de la Fase 3"))
    # P5 embudo validado
    v = json.loads((DIR_SALIDAS / "tablas" / "validacion_embudo.json").read_text(encoding="utf8"))
    ok5 = 0.02 <= v["fuera_95"] <= 0.09 and v["fuera_998"] <= 0.01
    out.append(_p("P5", "Embudo validado con hospitales simulados", ok5, f"sin diferencia real quedan fuera del 95 %: {v['fuera_95']:.3f} (esperado cerca de 0,05) y del 99,8 %: {v['fuera_998']:.4f} (esperado cerca de 0,002)"))
    # P6 ensayo completo
    tablas = ["criterios_mortalidad", "criterios_estadia", "oe_hospital_mortalidad", "oe_hospital_estadia", "indice_exhaustividad_hospital", "subgrupos_mortalidad", "subgrupos_estadia", "tasa_traslado_hospital"]
    falt = [t for t in tablas if not (DIR_SALIDAS / "ensayo_2023" / "tablas" / f"{t}.csv").exists()]
    out.append(_p("P6", "Ensayo general con 2023", not falt, "todas las tablas de la evaluación se generaron con 2023" if not falt else f"faltan: {falt}"))
    # P7 privacidad
    h = privacidad.escanear_carpeta(DIR_SALIDAS)
    r7 = {"rut": sum(x["rut"] for x in h), "fechas": sum(x["fechas"] for x in h), "celdas_pequenas": sum(x["celdas_pequenas"] for x in h)}
    out.append(_p("P7", "Privacidad de las salidas", r7["rut"] == 0 and r7["fechas"] == 0 and r7["celdas_pequenas"] == 0, f"RUT: {r7['rut']}; fechas completas: {r7['fechas']}; celdas de 1 a 9 casos: {r7['celdas_pequenas']}"))
    # P8 pruebas
    pas, fal = 0, 0
    if con_pytest:
        rr = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=-p no:cacheprovider", "--tb=line"], cwd=RAIZ, capture_output=True, text=True)
        ult = rr.stdout.strip().splitlines()[-1] if rr.stdout.strip() else ""
        pas = int(re.search(r"(\d+) passed", ult).group(1)) if re.search(r"(\d+) passed", ult) else 0
        fal = int(re.search(r"(\d+) failed", ult).group(1)) if re.search(r"(\d+) failed", ult) else (0 if rr.returncode == 0 else 1)
        (trabajo / "pytest_fase5.json").write_text(json.dumps({"pasaron": pas, "fallaron": fal}), encoding="utf8")
    else:
        d = json.loads((trabajo / "pytest_fase5.json").read_text(encoding="utf8"))
        pas, fal = d["pasaron"], d["fallaron"]
    out.append(_p("P8", "Pruebas automáticas", pas > 0 and fal == 0, f"{pas} pasaron, {fal} fallaron"))
    # estado de la apertura
    aut = (RAIZ / prot.apertura["archivo_autorizacion"]).exists()
    out.append({"puerta": "A", "nombre": "Autorización de apertura", "estado": "PENDIENTE" if not aut else "RECIBIDA", "detalle": "falta la autorización escrita del profesor guía" if not aut else "archivo de autorización presente"})
    df = pd.DataFrame(out)
    df.to_csv(DIR_SALIDAS / "tablas" / "tabla_verificacion_preparacion_fase5.csv", index=False)
    return df


def veredicto_global(destino: Path) -> dict:
    """Resume los 7 criterios de la Fase 1 con el veredicto de cada uno (lee las tablas ya escritas)."""
    t = destino / "tablas"
    cm = pd.read_csv(t / "criterios_mortalidad.csv")
    ce = pd.read_csv(t / "criterios_estadia.csv")
    c = pd.concat([cm, ce])
    c = c[c["veredicto"] != "informativa"]
    return {"n": int(len(c)), "cumple": int((c["veredicto"] == "cumple").sum()), "incertidumbre": int((c["veredicto"] == "cumple con incertidumbre").sum()), "no_cumple": int((c["veredicto"] == "no cumple").sum())}


def verificar_cierre(prot, trabajo: Path, con_pytest: bool = True) -> pd.DataFrame:
    """Las ocho puertas de cierre G1–G8. Miden la calidad del proceso, no si los modelos cumplen los criterios (eso es el veredicto)."""
    out = []
    destino = DIR_SALIDAS / "evaluacion_2024"
    pre = json.loads((RAIZ / "documentos" / "registro_preapertura.json").read_text(encoding="utf8"))
    # G1 modelos intactos
    r = huellas.verificar_manifiesto(DIR_MODELOS_F4)
    rb = huellas.verificar_manifiesto(DIR_PLANES_B)
    out.append(_p("G1", "Modelos intactos", all(v[0] for v in r.values()) and all(v[0] for v in rb.values()), f"{len(r)} archivos de la Fase 4 y {len(rb)} de los planes B siguen con la huella del manifiesto"))
    # G2 apertura única y autorizada
    n = apertura.lecturas_concedidas(REGISTRO_FASE3)
    aut = RAIZ / prot.apertura["archivo_autorizacion"]
    try:
        apertura.comprobar_autorizacion(aut, prot.apertura["frase"])
        okaut = True
    except apertura.AperturaNoAutorizada:
        okaut = False
    out.append(_p("G2", "Apertura única y autorizada", n == 1 and okaut, f"lecturas concedidas de 2024 en el registro: {n}; archivo de autorización completo: {'sí' if okaut else 'no'}"))
    # G3 protocolo pre-registrado y sin cambios
    reg = pd.read_csv(REGISTRO_FASE3)
    lect = reg[(reg["evento"] == "lectura_desenlace") & (reg["concedido"] == "si")]
    t_lect = pd.to_datetime(lect["momento_utc"]).min().timestamp() if len(lect) else float("inf")
    okp = pre["huella_protocolo"] == prot.huella() and pre["momento_epoch"] < t_lect
    out.append(_p("G3", "Protocolo pre-registrado", okp, "la huella del protocolo es la registrada y el registro es anterior a la apertura" if okp else "el protocolo cambió o se registró después de la apertura"))
    # G4 sin reajuste con 2024
    bad = [f.name for f in (RAIZ / "evaluacion").glob("*.py") if f.name not in ("planes_b.py", "verificacion.py") and re.search(r"\.fit\(|ajustar_final|entrenar\(", f.read_text(encoding="utf8"))]
    out.append(_p("G4", "Sin reajuste con 2024", not bad and all(v[0] for v in r.values()), "el código de la fase no entrena ni recalibra y los modelos no cambiaron" if not bad else f"se encontraron ajustes en: {bad}"))
    # G5 criterios completos
    cm, ce = pd.read_csv(destino / "tablas" / "criterios_mortalidad.csv"), pd.read_csv(destino / "tablas" / "criterios_estadia.csv")
    vv = veredicto_global(destino)
    emb = all((destino / "figuras" / f"embudo_{x}.png").exists() for x in ("mortalidad", "estadia"))
    out.append(_p("G5", "Criterios completos", vv["n"] == 7 and emb and cm["valor"].notna().all() and ce["valor"].notna().all(), f"{vv['n']} criterios con valor, intervalo y veredicto; gráficos de embudo de mortalidad y estadía generados"))
    # G6 indicador válido
    res = json.loads((destino / "resumen_evaluacion_2024.json").read_text(encoding="utf8"))
    om, oe_ = pd.read_csv(destino / "tablas" / "oe_hospital_mortalidad.csv"), pd.read_csv(destino / "tablas" / "oe_hospital_estadia.csv")
    ok6 = res["hospitales_mortalidad"]["n_elegibles"] >= 30 and len(om) == res["hospitales_mortalidad"]["n_elegibles"] and len(oe_) == res["hospitales_estadia"]["n_elegibles"] and om.loc[om["observado"].astype(str) != "<10", "oe"].notna().all()
    out.append(_p("G6", "Indicador válido", ok6, f"O/E calculado en {len(om)} hospitales de mortalidad y {len(oe_)} de estadía que cumplen el volumen mínimo (el O/E se omite solo donde hay menos de 10 eventos)"))
    # G7 privacidad y sin ranking
    h = privacidad.escanear_carpeta(destino)
    sin_rank = all(pd.read_csv(f)["cod_hospital"].is_monotonic_increasing for f in (destino / "tablas").glob("*hospital*.csv"))
    no_rank_col = not any(("rank" in c.lower() or "posicion" in c.lower()) for f in (destino / "tablas").glob("*.csv") for c in pd.read_csv(f, nrows=1).columns)
    out.append(_p("G7", "Privacidad y sin ranking", not h and sin_rank and no_rank_col, f"RUT, fechas y celdas de 1 a 9 casos: {len(h)} hallazgos; tablas por hospital ordenadas por código y sin columnas de posición"))
    # G8 reproducibilidad
    if con_pytest:
        rr = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=-p no:cacheprovider", "--tb=line"], cwd=RAIZ, capture_output=True, text=True)
        ult = rr.stdout.strip().splitlines()[-1] if rr.stdout.strip() else ""
        pas = int(re.search(r"(\d+) passed", ult).group(1)) if re.search(r"(\d+) passed", ult) else 0
        fal = int(re.search(r"(\d+) failed", ult).group(1)) if re.search(r"(\d+) failed", ult) else (0 if rr.returncode == 0 else 1)
    else:
        pas, fal = 0, 1
    hd = trabajo / "evaluacion_2024" / "huella_desenlace_2024.json"
    out.append(_p("G8", "Reproducibilidad", pas > 0 and fal == 0 and hd.exists() and res.get("bootstrap") == prot.incertidumbre["bootstrap"], f"{pas} pruebas pasaron, {fal} fallaron; semilla {prot.semilla}; {res.get('bootstrap')} réplicas; huella del desenlace abierto registrada"))
    df = pd.DataFrame(out)
    df.to_csv(DIR_SALIDAS / "tablas" / "tabla_verificacion_cierre_fase5.csv", index=False)
    return df
