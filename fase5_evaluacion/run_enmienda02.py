"""Corre los análisis de la enmienda 02 (datos y cohortes). Desde fase5_evaluacion/:

    python run_enmienda02.py --capas <capas> --trabajo4 <trabajo4> --trabajo5 <trabajo5> \
        --interna01 <carpeta interna de la enmienda 01> --interna <carpeta interna de la enmienda 02>

- Solo corre si la enmienda 02 está registrada y el código coincide con documentos/registro_corrida_enmienda_02.json.
- La base de comparación es la de la enmienda 01, que se lee de su carpeta interna.
- E2.4a y E2.4b se corren solo si existe fase3_preparacion/config/tipo_establecimiento.csv y su huella se registró
  antes de correr; si no, quedan como pendientes en el resumen.
- Lo publicable queda en salidas/enmienda_02/; las tablas por hospital, en --interna.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

from enmienda02 import comun, filtros, indicador, modelo, publicar
from evaluacion.config import RAIZ, RAIZ_TESIS


def main() -> int:
    ap = argparse.ArgumentParser(description="Análisis de la enmienda 02 al protocolo de la Fase 5")
    for k in ("--capas", "--trabajo4", "--trabajo5", "--interna01", "--interna"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    try:
        guardia = comun.verificar_autorizacion()
        usar_deis = comun.catalogo_deis_autorizado(guardia)
    except comun.EnmiendaNoAutorizada as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    enm = comun.cargar_enmienda()
    capas, t4, t5, int01, interna = (Path(x) for x in (a.capas, a.trabajo4, a.trabajo5, a.interna01, a.interna))
    interna.mkdir(parents=True, exist_ok=True)
    bitacora = open(interna / "bitacora.txt", "a", encoding="utf8")

    def log(msg: str) -> None:
        linea = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S} {msg}"
        print(linea, flush=True)
        bitacora.write(linea + "\n")
        bitacora.flush()

    inicio = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    log(f"enmienda 02: corrida autorizada ({guardia['registro_corrida'][:8]}…); catálogo DEIS: {'sí' if usar_deis else 'no (E2.4 pendiente)'}")
    eps = {anio: filtros.episodios(capas, t5, anio) for anio in range(2019, 2025)}
    res_mod = modelo.correr(capas, t4, t5, int01, interna, enm, eps, log=log)
    res_ind = indicador.correr(capas, t5, int01, interna, enm, eps, RAIZ_TESIS, usar_deis, log=log)
    from modelado.datos import huellas_entrada
    meta = {"enmienda": 2, **{k: v for k, v in guardia.items() if k != "scripts_registrados"}, "inicio_utc": inicio,
            "fin_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(), "huellas_oro": huellas_entrada(capas),
            "catalogo_deis": usar_deis, "semilla_modelo": comun.SEMILLA_MODELO, "semilla_bootstrap": comun.SEMILLA_BOOTSTRAP}
    tablas = publicar.escribir(res_mod, res_ind, filtros.muertes_obstetricas(eps), RAIZ / "salidas" / "enmienda_02", meta)
    log(f"publicado en salidas/enmienda_02/: {sorted(tablas)}")
    bitacora.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
