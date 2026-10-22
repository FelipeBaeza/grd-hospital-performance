"""Corre los análisis de la enmienda 01. Desde fase5_evaluacion/:

    python run_enmienda01.py --capas <capas> --trabajo4 <trabajo4> --trabajo5 <trabajo5> --interna <carpeta interna> \
        [--etapas modelo indicador publicar] [--exploracion <salida de la auditoría>]

- Solo corre si la enmienda está registrada y el código coincide con la huella registrada antes de la primera
  corrida (documentos/registro_corrida_enmienda_01.json).
- El diseño original lee el desenlace de 2024 ya abierto (trabajo5/evaluacion_2024), como secundario.
- Las tablas por hospital quedan en --interna, fuera del repositorio, porque no se publican.
- Lo publicable queda en salidas/enmienda_01/.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from enmienda01 import comun, indicador, modelo, publicar
from evaluacion.config import RAIZ, RAIZ_TESIS


def main() -> int:
    ap = argparse.ArgumentParser(description="Análisis de la enmienda 01 al protocolo de la Fase 5")
    for k in ("--capas", "--trabajo4", "--trabajo5", "--interna"):
        ap.add_argument(k, required=True)
    ap.add_argument("--etapas", nargs="+", default=["modelo", "indicador", "publicar"], choices=["modelo", "indicador", "publicar"])
    ap.add_argument("--exploracion", default=None, help="carpeta con los resultados exploratorios de la auditoría, para compararlos")
    a = ap.parse_args()
    try:
        guardia = comun.verificar_autorizacion()
    except comun.EnmiendaNoAutorizada as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    enm = comun.cargar_enmienda()
    capas, t4, t5, interna = Path(a.capas), Path(a.trabajo4), Path(a.trabajo5), Path(a.interna)
    interna.mkdir(parents=True, exist_ok=True)
    bitacora = open(interna / "bitacora.txt", "a", encoding="utf8")

    def log(msg: str) -> None:
        linea = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S} {msg}"
        print(linea, flush=True)
        bitacora.write(linea + "\n")
        bitacora.flush()

    inicio = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    log(f"enmienda 01: corrida autorizada ({guardia['registro_corrida'][:8]}…); etapas {a.etapas}")
    if "modelo" in a.etapas:
        log("lectura del desenlace de 2024 ya abierto (diseño original, secundario), autorizada por la enmienda 01")
        modelo.correr(capas, t4, t5, interna, enm, log=log)
    if "indicador" in a.etapas:
        res_ind = indicador.correr(capas, t5, interna, enm, RAIZ_TESIS, log=log)
    if "publicar" in a.etapas:
        res_mod = pd.read_csv(interna / "modelo_resultados.csv")
        if "indicador" not in a.etapas:
            res_ind = {k: pd.read_csv(interna / f"{k}_resultados.csv") for k in ("indicador", "marcas", "estratos", "validacion_p5b", "estabilidad")}
        from modelado.datos import huellas_entrada
        meta = {"enmienda": 1, **guardia, "inicio_utc": inicio, "fin_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
                "huellas_oro": huellas_entrada(capas), "lecturas_desenlace_2024": "una por corrida, del archivo ya abierto (diseño original)",
                "semilla_modelo": comun.SEMILLA_MODELO, "semilla_bootstrap": comun.SEMILLA_BOOTSTRAP}
        tablas = publicar.escribir(res_mod, res_ind, RAIZ / "salidas" / "enmienda_01", meta, Path(a.exploracion) if a.exploracion else None)
        log(f"publicado en salidas/enmienda_01/: {sorted(tablas)}")
    bitacora.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
