"""Orquestador de la Fase 3: corre las etapas en orden, escribe las tablas de salida y verifica las ocho puertas."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

from . import bronce, catalogo, cohortes, definiciones as dfn, elixhauser, oro, plata, verificacion
from .config import ANIO_SELLADO, ANIOS, ANIOS_RESULTADO, DIR_CONFIG, DIR_SALIDAS, RAIZ, RAIZ_PROYECTO, cargar_umbrales

ETAPAS = ("bronce", "plata", "cohortes", "oro", "catalogo", "verificacion")


def pico_ram_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # kB -> GB en Linux


def localizar_archivo(data_root: Path, anio: int) -> Path:
    candidatos = [p for p in data_root.rglob("*.txt") if str(anio) in p.name and p.name.upper().startswith("GRD_PUBLICO")]
    if not candidatos:
        raise FileNotFoundError(f"no se encontró el archivo de {anio} en {data_root}")
    return sorted(candidatos)[0]


def escribir(df: pl.DataFrame, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(ruta)


def texto_suprimido(df: pl.DataFrame, columnas: list[str], minimo: int) -> pl.DataFrame:
    """Reemplaza por «<minimo» los conteos de 1 a minimo-1 (regla de privacidad DT04 de la Fase 2); el resto se guarda como texto."""
    return df.with_columns([pl.when((pl.col(c) > 0) & (pl.col(c) < minimo)).then(pl.lit(f"<{minimo}")).otherwise(pl.col(c).cast(pl.String)).alias(c) for c in columnas])


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def descripcion_oro() -> pl.DataFrame:
    filas = [
        ("ID_EPISODIO", "Identificador del egreso: año, línea del archivo y huella del contenido", "clave", "no es predictor", "Bronce"),
        ("ANIO_EGRESO", "Año del archivo de egresos", "clave", "no es predictor", "nombre del archivo"),
        ("SPLIT", "Conjunto: entrenamiento (2019–2022), calibración (2023) o prueba (2024); se decide solo por el año", "clave", "no es predictor", "ANIO_EGRESO"),
        ("COD_HOSPITAL", "Código del hospital; sirve para agrupar resultados, nunca como predictor (R2)", "clave", "no es predictor", "COD_HOSPITAL"),
        ("EDAD_ANIOS", "Edad al ingreso, en años cumplidos (vacía si es imposible)", "numérica", "permitida", "FECHA_NACIMIENTO y FECHA_INGRESO"),
        ("SEXO", "Sexo registrado (MUJER, HOMBRE o SIN_DATO)", "categórica", "permitida", "SEXO"),
        ("ES_FIN_SEMANA", "1 si el ingreso fue sábado o domingo", "binaria", "permitida (señal baja)", "FECHA_INGRESO"),
        ("TIPO_INGRESO", "Forma de ingreso (urgencia o programado)", "categórica", "condicional", "TIPO_INGRESO"),
        ("TIPO_PROCEDENCIA", "De dónde viene el paciente", "categórica", "condicional", "TIPO_PROCEDENCIA"),
        ("PROCEDENCIA_AGR", "Procedencia agrupada: URGENCIA, PROGRAMADO o DERIVADO_OTRO_HOSPITAL (pendiente de revisión clínica)", "categórica", "condicional", "TIPO_PROCEDENCIA"),
        ("NIVEL_CUIDADO_INGRESO", "1 si ingresó a una unidad de cuidados especiales (UCI, UTI o intermedios); vacía si falta el servicio", "binaria", "condicional", "SERVICIOINGRESO"),
        ("CIE10_3C", "Diagnóstico principal en 3 caracteres (los poco frecuentes: RARO)", "categórica", "permitida con reserva", "DIAGNOSTICO1"),
        ("GRUPO_CLINICO", "Capítulo CIE-10 del diagnóstico principal", "categórica", "permitida con reserva", "DIAGNOSTICO1 y catálogo CIE-10"),
    ]
    nombres = elixhauser.cargar().nombres_es
    for i in elixhauser.IDS:
        filas.append((i, f"Elixhauser: {nombres[i]} (solo diagnósticos secundarios)", "binaria", "permitida con reserva", "DIAGNOSTICO2 a DIAGNOSTICO35"))
    filas += [
        ("EN_COHORTE_D", "El egreso pertenece a la cohorte de mortalidad (vacía en 2024)", "marca", "no es predictor", "cohortes"),
        ("EN_COHORTE_E", "El egreso pertenece a la cohorte de estadía (vacía en 2024)", "marca", "no es predictor", "cohortes"),
        ("MORTALIDAD_BINARIA", "1 si el paciente falleció; archivo de desenlace aparte", "desenlace", "desenlace", "TIPOALTA"),
        ("ESTANCIA_DIAS", "Días de hospitalización (fecha de alta menos fecha de ingreso); archivo de desenlace aparte", "desenlace", "desenlace", "FECHAALTA y FECHA_INGRESO"),
        ("ESTANCIA_DIAS_TOPE", "Estadía recortada en el percentil 99 de 2019–2022; archivo de desenlace aparte", "desenlace", "desenlace", "ESTANCIA_DIAS"),
        ("N_COMORB_ELIX", "Número de categorías de Elixhauser; tabla de línea base aparte", "numérica", "solo línea base", "ELIX_01 a ELIX_31"),
        ("PUNTAJE_VAN_WALRAVEN", "Puntaje de van Walraven; tabla de línea base aparte", "numérica", "solo línea base", "ELIX_01 a ELIX_31"),
        ("N_DX_SECUNDARIOS", "Cantidad de diagnósticos secundarios (exhaustividad); tabla de línea base aparte", "numérica", "no es predictor", "DIAGNOSTICO2 a DIAGNOSTICO35"),
    ]
    return pl.DataFrame(filas, schema=["columna", "descripcion", "tipo", "estado", "origen"], orient="row")


def run(data_root: Path, capas: Path, salidas: Path, etapas: list[str], anios: list[int], proyecto: Path, lote: int | None = None) -> dict:
    u = cargar_umbrales()
    tablas = salidas / "tablas"
    tablas.mkdir(parents=True, exist_ok=True)
    roles = pl.read_csv(DIR_CONFIG / "tipoalta_roles.csv")
    d_b, d_p, d_s, d_c, d_o = (capas / n for n in ("bronce", "plata", "sellado", "cohortes", "oro"))
    tiempos = []

    def paso(nombre):
        t0 = time.time()
        return lambda: tiempos.append({"etapa": nombre, "segundos": round(time.time() - t0, 1), "pico_ram_gb": round(pico_ram_gb(), 2)})

    if "bronce" in etapas:
        fin = paso("bronce")
        reps = []
        for a in anios:
            print(f"[bronce] {a}", flush=True)
            reps.append(bronce.construir_bronce(a, localizar_archivo(data_root, a), d_b, u, lote))
        fin()
    if "plata" in etapas:
        fin = paso("plata")
        for a in anios:
            print(f"[plata] {a}", flush=True)
            plata.construir_plata(a, d_b, d_p, d_s, roles, u, lote)
        fin()
    if "cohortes" in etapas:
        fin = paso("cohortes")
        casc = []
        for a in anios:
            print(f"[cohortes] {a}", flush=True)
            casc.append(cohortes.construir_cohorte(a, d_p, d_c, u))
        fin()
    if "oro" in etapas:
        fin = paso("oro")
        info = oro.construir_oro(anios, d_p, d_c, d_o, proyecto / "CIE-10.csv", u)
        print("[oro] matrices", flush=True)
        m = oro.construir_matrices(anios, d_p, d_o, u)
        fin()
        escribir(texto_suprimido(info["tabla_vocabularios"], ["n_entrenamiento"], int(u["celda_minima"])), tablas / "oro_vocabularios.csv")
        (d_o / "parametros_oro.json").write_text(json.dumps({**m, "vocabularios": info["vocabularios"]}, indent=1), encoding="utf8")
    if "catalogo" in etapas:
        fin = paso("catalogo")
        cat, concil = catalogo.construir_catalogo(anios, d_p, d_c, d_o, proyecto / "Hospitales.csv", int(u["celda_minima"]))
        escribir(cat, tablas / "catalogo_hospitales_anio.csv")
        escribir(concil, tablas / "catalogo_conciliacion.csv")
        (tablas / "catalogo_hospitales.sql").write_text(catalogo.script_sql(cat), encoding="utf8")
        fin()
    escribir(pl.DataFrame(tiempos), tablas / "tiempos_etapas.csv") if tiempos else None
    return {"tiempos": tiempos}


def tablas_resumen(capas: Path, salidas: Path, anios: list[int]) -> None:
    """Resúmenes de Bronce, Plata, cascada y Oro a partir de los reportes de cada capa."""
    u = cargar_umbrales()
    tablas = salidas / "tablas"
    d_b, d_p, d_c, d_o = (capas / n for n in ("bronce", "plata", "cohortes", "oro"))
    rb = [json.loads((d_b / f"reporte_bronce_{a}.json").read_text(encoding="utf8")) for a in anios]
    escribir(pl.DataFrame([{k: r[k] for k in ("anio", "archivo", "sha256_origen", "bytes_origen", "lineas_datos", "filas_leidas", "filas_rechazadas", "cuadra",
                                              "columnas_origen", "columnas_bronce", "filas_repetidas_exactas")} for r in rb]).rename({"filas_rechazadas": "lineas_defectuosas_rechazadas"}), tablas / "bronce_resumen.csv")
    celda = int(u["celda_minima"])
    rut = [{"anio": r["anio"], "columna": c, "n": (str(n) if n >= celda else f"<{celda}")} for r in rb for c, n in r["hallazgos_rut"].items()]
    escribir(pl.DataFrame(rut, schema={"anio": pl.Int64, "columna": pl.String, "n": pl.String}), tablas / "bronce_hallazgos_rut.csv")
    rp = [json.loads((d_p / f"reporte_plata_{a}.json").read_text(encoding="utf8")) for a in anios]
    escribir(pl.DataFrame([{"anio": r["anio"], "filas_bronce": r["filas_bronce"], "filas_plata": r["filas_plata"], "cuadra": r["cuadra"], "columnas_plata": r["columnas_plata"],
                            "episodios_mismo_paciente_mismas_fechas": r["episodios_mismo_paciente_mismas_fechas"]} for r in rp]), tablas / "plata_resumen.csv")
    escribir(pl.DataFrame([{"anio": r["anio"], "regla": k, "n": v} for r in rp for k, v in r["anulados"].items()]).pipe(texto_suprimido, ["n"], celda), tablas / "plata_anulados.csv")
    escribir(pl.DataFrame([{"anio": r["anio"], "columna": k, "n_faltantes": v, "prop": (v / r["filas_plata"]) if (v == 0 or v >= celda) else None} for r in rp for k, v in r["faltantes"].items()]).pipe(texto_suprimido, ["n_faltantes"], celda), tablas / "plata_faltantes.csv")
    # cascada
    casc = pl.concat([cohortes_cascada_desde_archivo(d_c, d_p, a) for a in anios])
    escribir(casc, tablas / "cascada_fase3.csv")
    comp = cohortes.comparar_con_referencia(casc, DIR_CONFIG / "cifras_referencia_cascada.csv")
    escribir(comp, tablas / "comparacion_cascada_fase2.csv")
    cuadran = {str(a): cohortes.cuadra(casc.filter(pl.col("periodo") == str(a))) for a in anios}
    (tablas / "cascada_cuadra.json").write_text(json.dumps({"por_anio": cuadran, "todas_cuadran": all(cuadran.values())}, indent=1), encoding="utf8")
    # períodos
    per = []
    for nombre, lista, pasos in (("2019-2024 (estructura)", list(anios), ["TOTAL", "EX01", "EX02", "EX03", "A", "EX04", "B", "EX05", "C"]),
                                 ("2019-2023 (desenlace)", [a for a in anios if a in ANIOS_RESULTADO], ["TOTAL", "C", "EX06", "D", "EX07", "EX08", "E"]),
                                 ("2019-2022 (entrenamiento)", [a for a in anios if a <= 2022], ["D", "E"]), ("2023 (calibración)", [2023] if 2023 in anios else [], ["D", "E"])):
        sub = casc.filter(pl.col("periodo").is_in([str(a) for a in lista]) & pl.col("paso").is_in(pasos)).group_by("paso", "descripcion", "orden").agg(pl.col("n").sum()).sort("orden")
        per += [{"periodo": nombre, "paso": r["paso"], "descripcion": r["descripcion"], "n": r["n"]} for r in sub.iter_rows(named=True)]
    escribir(pl.DataFrame(per), tablas / "cascada_periodos.csv")


def cohortes_cascada_desde_archivo(d_c: Path, d_p: Path, a: int) -> pl.DataFrame:
    """Reconstruye la cascada de un año desde el archivo de cohorte (así se puede regenerar sin recalcular)."""
    c = pl.read_parquet(d_c / f"cohorte_{a}.parquet")
    if a != ANIO_SELLADO:
        des = pl.read_parquet(d_p / f"desenlace_{a}.parquet", columns=["ROL"])
        c = c.with_columns(des["ROL"])
    return cohortes.cascada_anio(c, a)


def tablas_oro(capas: Path, salidas: Path, anios: list[int]) -> dict:
    u = cargar_umbrales()
    tablas = salidas / "tablas"
    d_o, d_p = capas / "oro", capas / "plata"
    escribir(descripcion_oro(), tablas / "diccionario_oro.csv")
    # prevalencia de Elixhauser por año (cohorte C) y mortalidad por conjunto
    prev = []
    for a in anios:
        p = pl.read_parquet(d_o / f"oro_predictores_{a}.parquet", columns=["ANIO_EGRESO", *oro.ELIX])
        row = p.select([pl.col(i).mean().alias(i) for i in oro.ELIX]).row(0)
        prev += [{"anio": a, "elix_id": i, "prevalencia": v} for i, v in zip(oro.ELIX, row)]
    escribir(pl.DataFrame(prev), tablas / "oro_elix_prevalencia_anio.csv")
    mm = pl.read_parquet(d_o / "matriz_mortalidad.parquet", columns=["ID_EPISODIO", "SPLIT", "ANIO_EGRESO"]).join(pl.read_parquet(d_o / "desenlace_mortalidad.parquet"), on="ID_EPISODIO")
    escribir(mm.group_by("SPLIT").agg(pl.len().alias("n_episodios"), pl.col("MORTALIDAD_BINARIA").sum().alias("n_fallecidos"), pl.col("MORTALIDAD_BINARIA").mean().alias("tasa_mortalidad")).sort("SPLIT"), tablas / "oro_mortalidad_conjuntos.csv")
    escribir(mm.group_by("ANIO_EGRESO").agg(pl.len().alias("n_episodios"), pl.col("MORTALIDAD_BINARIA").sum().alias("n_fallecidos"), pl.col("MORTALIDAD_BINARIA").mean().alias("tasa_mortalidad")).sort("ANIO_EGRESO"), tablas / "oro_mortalidad_anio.csv")
    ee = pl.read_parquet(d_o / "matriz_estadia.parquet", columns=["ID_EPISODIO", "SPLIT"]).join(pl.read_parquet(d_o / "desenlace_estadia.parquet"), on="ID_EPISODIO")
    escribir(ee.group_by("SPLIT").agg(pl.len().alias("n_episodios"), pl.col("ESTANCIA_DIAS").mean().alias("estancia_media"), pl.col("ESTANCIA_DIAS_TOPE").mean().alias("estancia_media_con_tope"),
                                      (pl.col("ESTANCIA_DIAS") > pl.col("ESTANCIA_DIAS_TOPE")).sum().alias("n_recortados")).sort("SPLIT"), tablas / "oro_estadia_conjuntos.csv")
    par = json.loads((d_o / "parametros_oro.json").read_text(encoding="utf8"))
    tope = int(par["tope_estancia_dias"])
    # tope recalculado de forma independiente (solo 2019-2022) y vocabularios con y sin 2023-2024
    dias = []
    for a in anios:
        if a in u.particion["anios_entrenamiento"]:
            c = pl.read_parquet(capas / "cohortes" / f"cohorte_{a}.parquet", columns=["E"])
            des = pl.read_parquet(d_p / f"desenlace_{a}.parquet", columns=["ESTANCIA_DIAS"])
            dias.append(des.filter(c["E"])["ESTANCIA_DIAS"].to_numpy())
    import numpy as np
    d_ent = np.concatenate(dias)
    tope_indep = oro.percentil_estancia(d_ent, float(u["percentil_estancia"]))
    afect = int((d_ent > tope).sum())
    escribir(pl.DataFrame([{"percentil": u["percentil_estancia"], "tope_dias": tope, "n_sobrevivientes_2019_2022": len(d_ent), "n_recortados": afect, "prop_recortados": afect / len(d_ent),
                            "maximo_dias": int(d_ent.max())}]), tablas / "oro_estancia_percentil99.csv")
    voc_ok = True
    # vocabularios: se recalculan desde los archivos finales solo con 2019–2022 y se comparan con la tabla guardada
    tv = pl.read_csv(tablas / "oro_vocabularios.csv", infer_schema=False)
    conservados = {v: set(tv.filter((pl.col("variable") == v) & (pl.col("conservado") == "true"))["nivel"].to_list()) for v in oro.CATEGORICAS}
    for a in anios:
        if a in u.particion["anios_entrenamiento"]:
            p = pl.read_parquet(d_o / f"oro_predictores_{a}.parquet", columns=list(oro.CATEGORICAS))
            for v in oro.CATEGORICAS:
                vistos = set(p[v].unique().to_list()) - {"RARO"}
                voc_ok = voc_ok and vistos <= conservados[v]
    (d_o / "verificacion_vocabularios.json").write_text(json.dumps({"identicos": bool(voc_ok), "tope_registrado": tope, "tope_con_entrenamiento": tope_indep}, indent=1), encoding="utf8")
    return {"tope": tope}


def determinismo(capas_a: Path, capas_b: Path) -> dict:
    iguales, total = 0, 0
    detalle = []
    for pa in sorted(capas_a.rglob("*.parquet")):
        pb = capas_b / pa.relative_to(capas_a)
        total += 1
        ok = pb.exists() and sha256(pa) == sha256(pb)
        iguales += int(ok)
        detalle.append({"archivo": str(pa.relative_to(capas_a)), "igual": ok})
    return {"archivos": total, "iguales": iguales, "detalle": detalle}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fase 3: preparación de los datos")
    ap.add_argument("--data-root", required=True, help="carpeta con los archivos GRD_PUBLICO_<año>.txt")
    ap.add_argument("--capas", help="carpeta de las capas (por defecto <data-root>/_fase3_capas)")
    ap.add_argument("--etapas", default=",".join(ETAPAS))
    ap.add_argument("--anios", default=",".join(map(str, ANIOS)))
    ap.add_argument("--sin-pytest", action="store_true")
    ap.add_argument("--sin-determinismo", action="store_true")
    ap.add_argument("--proyecto", default=str(RAIZ_PROYECTO), help="carpeta con CIE-10.csv y Hospitales.csv")
    a = ap.parse_args(argv)
    data_root = Path(a.data_root)
    capas = Path(a.capas) if a.capas else data_root / "_fase3_capas"
    etapas = [e.strip() for e in a.etapas.split(",") if e.strip()]
    anios = [int(x) for x in a.anios.split(",")]
    t0 = time.time()
    run(data_root, capas, DIR_SALIDAS, [e for e in etapas if e != "verificacion"], anios, Path(a.proyecto))
    if "verificacion" in etapas:
        tablas_resumen(capas, DIR_SALIDAS, anios)
        tablas_oro(capas, DIR_SALIDAS, anios)
        tablas = DIR_SALIDAS / "tablas"
        if not a.sin_determinismo:
            capas2 = capas.with_name(capas.name + "_2")
            if capas2.exists():
                shutil.rmtree(capas2)
            print("[determinismo] segunda corrida completa", flush=True)
            import os
            os.environ["FASE3_REGISTRO_ACCESOS"] = str(capas2 / "registro_accesos_2.csv")
            run(data_root, capas2, capas2 / "_salidas", [e for e in ETAPAS if e != "verificacion"], anios, Path(a.proyecto))
            tablas_oro_rep = determinismo(capas, capas2)
            del os.environ["FASE3_REGISTRO_ACCESOS"]
            shutil.rmtree(capas2)
            tablas_oro_rep["pico_ram_gb"] = round(pico_ram_gb(), 2)
            (tablas / "determinismo.json").write_text(json.dumps(tablas_oro_rep, indent=1), encoding="utf8")
        if not a.sin_pytest:
            r = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=-p no:cacheprovider", "--tb=line"], cwd=RAIZ, capture_output=True, text=True)
            ult = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""   # «53 passed in 2.1s»
            import re
            pas = int((re.search(r"(\d+) passed", ult) or [0, 0])[1]) if re.search(r"(\d+) passed", ult) else 0
            fal = int((re.search(r"(\d+) failed", ult) or [0, 0])[1]) if re.search(r"(\d+) failed", ult) else (0 if r.returncode == 0 else 1)
            (tablas / "pytest_resumen.json").write_text(json.dumps({"pasaron": pas, "fallaron": fal, "linea_final": ult}, indent=1, ensure_ascii=False), encoding="utf8")
        ver = verificacion.verificar(capas, DIR_SALIDAS, anios)
        escribir(ver, tablas / "tabla_verificacion.csv")
        print(ver)
    print(f"listo en {time.time() - t0:.0f} s; pico de RAM {pico_ram_gb():.2f} GB", flush=True)
    return 0
