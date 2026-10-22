# Guía de la Fase 4 (Modelado)

Reproducir (requiere las capas de la Fase 3 en `--capas`):

```
pip install -r config/requirements.txt
python run_fase4.py --capas <capas_fase3> --trabajo <carpeta_trabajo> --etapas contrato ajuste comparacion seleccion calibracion integridad exhaustividad congelar verificacion
python generar_figuras.py
pytest            # 48 pruebas
```

- Reglas congeladas: `config/reglas_fase4.toml` (huella SHA-256 en el manifiesto).
- Resultados: `salidas/tablas/`, figuras: `salidas/figuras/`, modelos congelados: `modelos/` (manifiesto con SHA-256).
- El desenlace de 2024 NO se lee en esta fase (se abre una sola vez en la Fase 5, con aprobación del profesor guía).
- Las etapas largas (ajuste) se reanudan desde el checkpoint Optuna (SQLite) si se interrumpen. Cada reanudación queda
  en `<trabajo>/optuna/reanudaciones.csv` (plan 4.8). Con `--semilla-por-prueba`, una corrida cortada y retomada da las
  mismas pruebas que una de corrido. Viene apagado porque la corrida congelada se hizo sin esa opción.
- Informe explicado: `docs/fase4_modelado/Informe_Fase4_v5.docx`.
