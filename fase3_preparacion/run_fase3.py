"""Runner de la Fase 3 (preparación de los datos, CRISP-DM). Uso: python run_fase3.py --data-root RUTA [--etapas bronce,plata,cohortes,oro,catalogo,verificacion] [--anios 2019,2020]"""
import sys

from prep import orquestador

if __name__ == "__main__":
    sys.exit(orquestador.main())
