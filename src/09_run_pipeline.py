"""Paso 9: Orquestador Maestro de la Pipeline Completa (CLI).

Permite ejecutar de forma secuencial, reproducible y trazable cada uno de los 9 pasos del proyecto:
  00: Validador de esquemas de datos crudos
  01: Ingesta cruda a Parquet (Capa Bronce)
  02: Limpieza y tipado (Capa Plata)
  03: Generación de identificadores únicos
  04: Cascada de exclusiones clínicas CONSORT y comorbilidades Elixhauser
  05: Matriz analítica de features con garantía anti-fuga (Capa Gold)
  06: Entrenamiento y calibración de modelo de mortalidad (LightGBM)
  07: Entrenamiento de modelo de estancia hospitalaria (LightGBM Tweedie)
  08: Cálculo de benchmarks hospitalarios (HSMR e IEMC) y comparación vs. FONASA

Uso:
  python src/09_run_pipeline.py --all
  python src/09_run_pipeline.py --step 6
  python src/09_run_pipeline.py --from-step 4
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PASOS = [
    ("00", "src/00_schema_validator.py", "Validación de esquema y metadatos crudos"),
    ("01", "src/01_ingest_bronze.py", "Ingesta a Capa Bronce (Parquet)"),
    ("02", "src/02_clean_silver.py", "Limpieza, normalización y tipado (Capa Plata)"),
    ("03", "src/03_generate_id.py", "Generación determinista de ID_EPISODIO único"),
    ("04", "src/04_filter_cohort.py", "Cascada de exclusión CONSORT y comorbilidades Elixhauser"),
    ("05", "src/05_transform_gold.py", "Transformación a Capa Gold con garantía anti-fuga"),
    ("06", "src/06_train_mortality.py", "Entrenamiento del modelo de riesgo de mortalidad"),
    ("07", "src/07_train_los.py", "Entrenamiento del modelo de regresión de estancia"),
    ("08", "src/08_compute_benchmarks.py", "Cálculo de benchmarks hospitalarios (HSMR e IEMC)"),
]


def ejecutar_script(script_path: str, descripcion: str) -> bool:
    print("\n" + "=" * 80)
    print(f"EJECUTANDO: {descripcion}")
    print(f"Archivo:    {script_path}")
    print("=" * 80)

    t0 = time.time()
    cmd = [sys.executable, script_path]
    res = subprocess.run(cmd)

    tiempo = time.time() - t0
    if res.returncode == 0:
        print(f"-> PASO COMPLETADO EXITOSAMENTE en {tiempo:.2f}s")
        return True
    else:
        print(f"-> ERROR CRÍTICO EN EL PASO (Código de salida: {res.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Orquestador Maestro del Pipeline GRD")
    parser.add_argument("--all", action="store_true", help="Ejecuta todos los pasos desde 00 hasta 08")
    parser.add_argument("--step", type=str, help="Ejecuta un paso específico (ej. '04' o '4')")
    parser.add_argument("--from-step", type=str, help="Ejecuta desde un paso hasta el final (ej. '06')")

    args = parser.parse_args()

    if not args.all and not args.step and not args.from_step:
        parser.print_help()
        sys.exit(0)

    t_inicio = time.time()

    pasos_a_ejecutar = []
    if args.all:
        pasos_a_ejecutar = PASOS
    elif args.step:
        step_id = args.step.zfill(2)
        pasos_a_ejecutar = [p for p in PASOS if p[0] == step_id]
        if not pasos_a_ejecutar:
            print(f"Paso '{args.step}' no reconocido. Opciones: {[p[0] for p in PASOS]}")
            sys.exit(1)
    elif args.from_step:
        step_id = args.from_step.zfill(2)
        idx = next((i for i, p in enumerate(PASOS) if p[0] == step_id), None)
        if idx is None:
            print(f"Paso '{args.from_step}' no reconocido.")
            sys.exit(1)
        pasos_a_ejecutar = PASOS[idx:]

    print("=" * 80)
    print(f"INICIANDO EJECUCIÓN DEL PIPELINE ({len(pasos_a_ejecutar)} pasos programados)")
    print("=" * 80)

    exitosos = 0
    for paso_id, script_path, desc in pasos_a_ejecutar:
        ok = ejecutar_script(script_path, desc)
        if not ok:
            print(f"\nPipeline interrumpido debido a fallo en paso {paso_id}.")
            sys.exit(1)
        exitosos += 1

    t_total = time.time() - t_inicio
    print("\n" + "=" * 80)
    print(f"PIPELINE FINALIZADO CON ÉXITO: {exitosos}/{len(pasos_a_ejecutar)} pasos ejecutados en {t_total:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
