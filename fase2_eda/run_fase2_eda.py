"""Runner de la Fase 2 (comprensión de los datos, CRISP-DM): corre las Tareas 3 a 12, compone los siete informes, verifica la definición de terminado y escribe el manifiesto.

Uso: ``python run_fase2_eda.py [--fuente {extracto,completo}] [--data-root RUTA] [--archivo ANIO=RUTA ...] [--salida-base RUTA] [--sin-pytest] [--sin-determinismo] [--solo T3,T4,...]``.
La lógica vive en ``eda/orquestador.py``; este script solo orquesta la salida.
"""

import sys

from eda import orquestador

if __name__ == "__main__":
    sys.exit(orquestador.main())
