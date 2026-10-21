"""F3: latencia p95 < 2 s con carga moderada (20 usuarios simultáneos)."""
import concurrent.futures as cf
import os
import time

import httpx
import numpy as np
import pytest

from conftest import API_URL

DURACION = float(os.environ.get("LATENCIA_SEGUNDOS", "30"))
RUTAS = ["/indicadores/mortalidad", "/indicadores/estadia", "/hospitales/101100", "/metodologia", "/criterios"]


def test_p95_menor_a_2_segundos(cliente, perfil):
    tiempos, fallos = [], 0

    def usuario(i):
        t_fin = time.monotonic() + DURACION
        out, f = [], 0
        with httpx.Client(base_url=API_URL, timeout=30) as c:
            k = i
            while time.monotonic() < t_fin:
                t0 = time.perf_counter()
                r = c.post("/demo/riesgo", json={**perfil, "edad": 20 + (k % 70)}) if k % 3 == 0 else c.get(RUTAS[k % len(RUTAS)])
                out.append(time.perf_counter() - t0)
                f += r.status_code >= 400 and r.status_code != 429
                k += 1
                time.sleep(0.05)
        return out, f

    with cf.ThreadPoolExecutor(20) as ex:
        for o, f in ex.map(usuario, range(20)):
            tiempos += o
            fallos += f
    p95 = float(np.percentile(tiempos, 95))
    print(f"\nsolicitudes={len(tiempos)} p50={np.percentile(tiempos, 50):.3f}s p95={p95:.3f}s max={max(tiempos):.3f}s fallos={fallos}")
    assert fallos == 0 and p95 < 2.0
