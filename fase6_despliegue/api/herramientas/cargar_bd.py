"""F1. Crea el esquema y carga los datos publicables a PostgreSQL. Antes de cargar: huellas de los archivos y puerta de privacidad.
Después de cargar: compara cada tabla de la base con su archivo de origen y falla si algo no coincide."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, db, privacidad  # noqa: E402
from app.motor import sha256_archivo as sha  # noqa: E402


def _nulos(d: pd.DataFrame) -> list[dict]:
    d = d.astype(object).where(d.notna(), None)
    return d.to_dict("records")


def verificar_paquete(carpeta: Path) -> None:
    huellas = json.loads((carpeta / "huellas.json").read_text(encoding="utf8"))
    for nombre, h in huellas.items():
        if sha(carpeta / nombre) != h:
            raise SystemExit(f"{nombre} no coincide con huellas.json")
    hall = privacidad.escanear_carpeta(carpeta)
    if hall:
        raise SystemExit(f"la puerta de privacidad detuvo la carga: {hall}")


def cargar(carpeta: Path, url: str | None = None) -> dict:
    verificar_paquete(carpeta)
    from sqlalchemy import create_engine
    motor = create_engine(url) if url else db.motor()
    ddl = (Path(__file__).resolve().parents[1] / "app" / "esquema.sql").read_text(encoding="utf8")
    ind = pd.read_csv(carpeta / "indicador_hospital.csv", dtype={"cod_hospital": str, "observado_pub": str})
    crit = pd.read_csv(carpeta / "criterio_global.csv")
    ctx = pd.read_csv(carpeta / "contexto_hospital.csv", dtype={"cod_hospital": str})
    sub = pd.read_csv(carpeta / "subgrupo.csv", dtype={"grupo": str, "observado_pub": str})
    cal = pd.read_csv(carpeta / "calibracion_decil.csv")
    meta = json.loads((carpeta / "metadatos.json").read_text(encoding="utf8"))
    hosp = sorted(ind["cod_hospital"].unique())
    with motor.begin() as cx:
        for stmt in [s for s in ddl.split(";") if s.strip()]:
            cx.execute(text(stmt))
        cx.execute(text("INSERT INTO hospital VALUES (:c)"), [{"c": h} for h in hosp])
        cx.execute(text("INSERT INTO criterio_global VALUES (:tarea,:criterio,:valor,:ic_inf,:ic_sup,:umbral,:veredicto,:es_criterio,:orden)"), _nulos(crit[["tarea", "criterio", "valor", "ic_inf", "ic_sup", "umbral", "veredicto", "es_criterio", "orden"]]))
        cols = ["tarea", "cod_hospital", "n_egresos", "observado_pub", "esperado", "oe", "ic_inf", "ic_sup", "lim_inf_95", "lim_sup_95", "lim_inf_998", "lim_sup_998", "zona"]
        cx.execute(text(f"INSERT INTO indicador_hospital VALUES ({','.join(':' + c for c in cols)})"), _nulos(ind[cols]))
        cc = ["cod_hospital", "indice_vs_2023", "indice_relativo_al_anio", "tasa_traslado_saliente", "proporcion_traslado_entrante"]
        cx.execute(text(f"INSERT INTO contexto_hospital VALUES ({','.join(':' + c for c in cc)})"), _nulos(ctx[cc]))
        sc = ["tarea", "variable", "grupo", "n", "observado_pub", "esperado", "oe"]
        cx.execute(text(f"INSERT INTO subgrupo VALUES ({','.join(':' + c for c in sc)})"), _nulos(sub[sc]))
        kc = ["tarea", "decil", "n", "observado", "predicho", "razon_oe"]
        cx.execute(text(f"INSERT INTO calibracion_decil VALUES ({','.join(':' + c for c in kc)})"), _nulos(cal[kc]))
        cx.execute(text("INSERT INTO metadatos VALUES (:k, CAST(:v AS JSONB))"), [{"k": "paquete", "v": json.dumps(meta, ensure_ascii=False)}, {"k": "cargado_epoch", "v": json.dumps(int(time.time()))}])
    return verificar_carga(carpeta, motor)


def verificar_carga(carpeta: Path, motor) -> dict:
    """Cada tabla de la base debe ser idéntica a su archivo de origen (mismas filas y mismos valores)."""
    res = {}
    pares = {"indicador_hospital": ("indicador_hospital.csv", ["tarea", "cod_hospital"]), "criterio_global": ("criterio_global.csv", ["tarea", "criterio"]),
             "contexto_hospital": ("contexto_hospital.csv", ["cod_hospital"]), "subgrupo": ("subgrupo.csv", ["tarea", "variable", "grupo"]),
             "calibracion_decil": ("calibracion_decil.csv", ["tarea", "decil"])}
    for tabla, (archivo, clave) in pares.items():
        dtype = {"cod_hospital": str, "grupo": str, "observado_pub": str}
        csv = pd.read_csv(carpeta / archivo, dtype=dtype)
        bd = pd.read_sql(text(f"SELECT * FROM {tabla}"), motor)
        csv = csv[[c for c in bd.columns if c in csv.columns]].sort_values(clave).reset_index(drop=True)
        bd = bd[csv.columns].sort_values(clave).reset_index(drop=True)
        ok = len(csv) == len(bd)
        if ok:
            for c in csv.columns:
                a, b = csv[c], bd[c]
                if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                    ok &= bool(np.allclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True))
                else:
                    ok &= bool((a.fillna("∅").astype(str) == b.fillna("∅").astype(str)).all())
        res[tabla] = {"filas": int(len(bd)), "coincide": bool(ok)}
    if not all(v["coincide"] for v in res.values()):
        raise SystemExit(f"la base no coincide con los archivos de origen: {res}")
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datos", default=str(config.DIR_DATOS))
    p.add_argument("--esperar", type=int, default=0, help="segundos máximos para esperar a que la base responda")
    a = p.parse_args()
    t0 = time.time()
    while True:
        try:
            with db.motor().connect() as cx:
                cx.execute(text("SELECT 1"))
            break
        except Exception as e:  # noqa: BLE001
            if time.time() - t0 > a.esperar:
                raise SystemExit(f"no hay conexión con la base: {e}")
            time.sleep(1.5)
    print(json.dumps(cargar(Path(a.datos)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
