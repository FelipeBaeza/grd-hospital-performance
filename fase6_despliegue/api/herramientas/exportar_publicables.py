"""F0. Arma el paquete de datos publicables a partir de los resultados agregados de la Fase 5 y de los modelos congelados de la Fase 4.

Uso:
  python herramientas/exportar_publicables.py --fase5 <.../fase5_evaluacion/salidas/evaluacion_2024> --modelos <.../fase4_modelado/modelos> \
      --capas <carpeta con oro/> --cie10 <CIE-10.csv> --elix <elixhauser_categorias.csv> --salida ../datos/publicables

Solo lee tablas agregadas. Para el catálogo de la demo lee las columnas CIE10_3C, GRUPO_CLINICO, TIPO_PROCEDENCIA y PROCEDENCIA_AGR de la capa Oro
y guarda únicamente las correspondencias entre categorías (nunca filas de pacientes). Todo lo que se escribe pasa por el escáner de privacidad.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import privacidad  # noqa: E402

ROMANOS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX", "XXI", "XXII"]
EXCLUIR_PROC = {"(sin dato)", "RARO"}
EXCLUIR_INGRESO = {"(sin dato)", "RARO"}


def sha(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tablas_fase5(f5: Path, out: Path) -> dict:
    t = f5 / "tablas"
    res = json.loads((f5 / "resumen_evaluacion_2024.json").read_text(encoding="utf8"))
    # criterios
    partes = []
    for tarea in ("mortalidad", "estadia"):
        d = pd.read_csv(t / f"criterios_{tarea}.csv")
        d.insert(0, "tarea", tarea)
        d["es_criterio"] = d["veredicto"] != "informativa"
        d["orden"] = range(len(d))
        partes.append(d)
    pd.concat(partes).rename(columns={"criterio": "criterio"}).to_csv(out / "criterio_global.csv", index=False)
    # indicador por hospital
    partes = []
    for tarea in ("mortalidad", "estadia"):
        d = pd.read_csv(t / f"oe_hospital_{tarea}.csv", dtype={"cod_hospital": str, "observado": str})
        d.insert(0, "tarea", tarea)
        partes.append(d.rename(columns={"observado": "observado_pub"}))
    ind = pd.concat(partes)
    ind.to_csv(out / "indicador_hospital.csv", index=False)
    # contexto
    ex = pd.read_csv(t / "indice_exhaustividad_hospital.csv", dtype={"cod_hospital": str}).drop(columns=["n_egresos"])
    tr = pd.read_csv(t / "tasa_traslado_hospital.csv", dtype={"cod_hospital": str})
    en = pd.read_csv(t / "traslado_entrante_hospital.csv", dtype={"cod_hospital": str})
    ctx = ex.merge(tr, on="cod_hospital", how="outer").merge(en, on="cod_hospital", how="outer").sort_values("cod_hospital")
    ctx = ctx[ctx["cod_hospital"].isin(ind["cod_hospital"])]
    ctx.to_csv(out / "contexto_hospital.csv", index=False)
    # subgrupos y calibración
    sub = []
    cal = []
    for tarea in ("mortalidad", "estadia"):
        d = pd.read_csv(t / f"subgrupos_{tarea}.csv", dtype={"grupo": str, "observado": str})
        d.insert(0, "tarea", tarea)
        sub.append(d.rename(columns={"observado": "observado_pub"}))
        c = pd.read_csv(t / f"calibracion_deciles_{tarea}.csv")
        c.insert(0, "tarea", tarea)
        cal.append(c)
    pd.concat(sub).to_csv(out / "subgrupo.csv", index=False)
    pd.concat(cal).to_csv(out / "calibracion_decil.csv", index=False)
    return res


def recalibracion(modelos: Path, manifiesto: dict, out: Path) -> None:
    """Pasa los parámetros de los recalibradores (Platt y factor único) a JSON, para que la API no deba cargar objetos serializados."""
    sys.path.insert(0, str(Path(modelos).resolve().parents[0]))
    r = {}
    for tarea, tipo in (("mortalidad", "platt"), ("estadia", "factor")):
        ruta = modelos / manifiesto["archivos"][tarea]["recalibrador"]["ruta"]
        if sha(ruta) != manifiesto["archivos"][tarea]["recalibrador"]["sha256"]:
            raise SystemExit(f"el recalibrador de {tarea} no coincide con el manifiesto")
        o = joblib.load(ruta)
        r[tarea] = ({"tipo": "platt", "a": float(o.a), "b": float(o.b)} if tipo == "platt" else {"tipo": "factor", "f": float(o.f)}) | {"fuente_sha256": manifiesto["archivos"][tarea]["recalibrador"]["sha256"]}
    (out / "recalibracion.json").write_text(json.dumps(r, indent=1), encoding="utf8")


def catalogo_demo(capas: Path, cie10: Path, elix: Path, modelos: Path, manifiesto: dict, out: Path) -> dict:
    import lightgbm as lgb
    import polars as pl
    b = lgb.Booster(model_file=str(modelos / manifiesto["archivos"]["mortalidad"]["modelo"]["ruta"]))
    nombres = ["SEXO", "TIPO_INGRESO", "TIPO_PROCEDENCIA", "PROCEDENCIA_AGR", "CIE10_3C", "GRUPO_CLINICO"]
    niveles = dict(zip(nombres, b.pandas_categorical))
    # correspondencias a partir de la capa Oro (solo pares de categorías)
    frames = [pl.read_parquet(capas / "oro" / f"oro_predictores_{a}.parquet", columns=["CIE10_3C", "GRUPO_CLINICO", "TIPO_PROCEDENCIA", "PROCEDENCIA_AGR"]) for a in range(2019, 2025)]
    d = pl.concat(frames)
    cie_g = d.group_by("CIE10_3C").agg(pl.col("GRUPO_CLINICO").n_unique().alias("n"), pl.col("GRUPO_CLINICO").mode().first().alias("g"))
    unicos = {r["CIE10_3C"]: r["g"] for r in cie_g.filter(pl.col("n") == 1).to_dicts()}
    pr = d.group_by("TIPO_PROCEDENCIA").agg(pl.col("PROCEDENCIA_AGR").n_unique().alias("n"), pl.col("PROCEDENCIA_AGR").mode().first().alias("a"))
    proc_agr = {r["TIPO_PROCEDENCIA"]: r["a"] for r in pr.filter(pl.col("n") == 1).to_dicts()}
    # nombres CIE-10 de 3 caracteres (catálogo público del MINSAL/DEIS que ya está en el proyecto)
    cie = pd.read_csv(cie10, dtype=str)
    cie3 = cie[cie["Código"].str.len() == 3].drop_duplicates("Código").set_index("Código")
    cap = cie.drop_duplicates("Código").set_index("Código")["Capítulo"]
    diag = []
    for c in niveles["CIE10_3C"]:
        if c == "RARO" or c not in unicos or unicos[c] in ("RARO", "sin capítulo") or c not in cie3.index:
            continue
        diag.append({"codigo": c, "nombre": str(cie3.loc[c, "Descripción"]), "grupo_clinico": unicos[c]})
    el = pd.read_csv(elix)
    comorb = [{"id": r["elix_id"], "nombre": r["nombre_es"]} for _, r in el.sort_values("orden").iterrows()]
    cat = {
        "sexo": [s for s in niveles["SEXO"] if s != "SIN_DATO"],
        "tipo_ingreso": [s for s in niveles["TIPO_INGRESO"] if s not in EXCLUIR_INGRESO],
        "tipo_procedencia": [{"valor": s, "procedencia_agr": proc_agr[s]} for s in niveles["TIPO_PROCEDENCIA"] if s not in EXCLUIR_PROC and s in proc_agr],
        "diagnosticos": sorted(diag, key=lambda x: x["codigo"]),
        "comorbilidades": comorb,
        "grupos_clinicos": {ROMANOS[i]: str(cap[cap.str.startswith(f"Cap.{i + 1:02d}")].iloc[0]) if (cap.str.startswith(f"Cap.{i + 1:02d}")).any() else "" for i in range(len(ROMANOS))},
    }
    (out / "demo_catalogo.json").write_text(json.dumps(cat, ensure_ascii=False, indent=1), encoding="utf8")
    return cat


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fase5", required=True)
    p.add_argument("--modelos", required=True)
    p.add_argument("--capas", required=True)
    p.add_argument("--cie10", required=True)
    p.add_argument("--elix", required=True)
    p.add_argument("--salida", required=True)
    a = p.parse_args()
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    modelos = Path(a.modelos)
    manifiesto = json.loads((modelos / "manifiesto.json").read_text(encoding="utf8"))
    res = tablas_fase5(Path(a.fase5), out)
    recalibracion(modelos, manifiesto, out)
    cat = catalogo_demo(Path(a.capas), Path(a.cie10), Path(a.elix), modelos, manifiesto, out)
    meta = {"anio_evaluado": 2024, "resumen_fase5": res, "huella_reglas_fase4": manifiesto["huella_reglas"], "version_modelos": manifiesto["versiones"],
            "archivos_modelo": {t: manifiesto["archivos"][t]["modelo"] for t in ("mortalidad", "estadia")}, "n_diagnosticos_demo": len(cat["diagnosticos"])}
    (out / "metadatos.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf8")
    hallazgos = privacidad.escanear_carpeta(out)
    if hallazgos:
        raise SystemExit(f"la puerta de privacidad detuvo la exportación: {hallazgos}")
    huellas = {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name != "huellas.json"}
    (out / "huellas.json").write_text(json.dumps(huellas, indent=1), encoding="utf8")
    print("exportado:", ", ".join(huellas), "| diagnósticos para la demo:", len(cat["diagnosticos"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
