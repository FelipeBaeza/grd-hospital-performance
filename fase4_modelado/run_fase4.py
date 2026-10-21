"""Punto de entrada de la Fase 4: python run_fase4.py --capas <capas de la Fase 3> --trabajo <carpeta de trabajo>"""
import sys

from modelado import orquestador

if __name__ == "__main__":
    sys.exit(orquestador.main())
