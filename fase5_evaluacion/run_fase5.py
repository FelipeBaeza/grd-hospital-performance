"""Punto de entrada de la Fase 5: python run_fase5.py --capas <capas> --trabajo <trabajo5> --trabajo-fase4 <trabajo4> [--etapas ...]
Sin autorización escrita del profesor guía, la etapa 'apertura' no se ejecuta."""
import sys

from evaluacion import orquestador

if __name__ == "__main__":
    sys.exit(orquestador.main())
