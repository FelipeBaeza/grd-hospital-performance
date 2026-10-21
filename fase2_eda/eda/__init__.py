"""Suite fase2_eda: EDA reproducible de la Fase 2 (comprensión de los datos, CRISP-DM).

Principios (ver README de los informes generados):
- Solo agregados: nunca se imprimen filas, identificadores ni fechas de nacimiento.
- Celdas de conteo con n entre 1 y 9 se suprimen (regla P2 de la Fase 1).
- 2024 está expuesto: solo estructura y calidad; el resultado se analiza con 2019 a 2023.
- Toda cifra de un informe proviene de un CSV de ``salidas/<fuente>/tablas``.
"""

__version__ = "1.0.0"
