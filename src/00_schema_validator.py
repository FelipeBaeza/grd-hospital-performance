"""
00_schema_validator.py
----------------------
Paso 0: Validación de Esquema y Contrato de Columnas.

Valida que cada archivo plano de egresos GRD (.txt) contenga exactamente
las 129 columnas esperadas y aplica la regla de alias conocida
(ID_BENEFICIARIO -> CIP_ENCRIPTADO en registros de 2019).
"""

import sys
import glob
from pathlib import Path
import argparse
import yaml

# Alias conocidos entre extracciones
ALIAS_COLUMNAS = {
    "ID_BENEFICIARIO": "CIP_ENCRIPTADO"
}


def cargar_columnas_esperadas(ruta_contrato: Path) -> list[str]:
    """Carga la lista canónica de 129 columnas en orden desde el contrato YAML."""
    if not ruta_contrato.exists():
        raise FileNotFoundError(f"No se encontró el contrato de esquema en {ruta_contrato}")
    with open(ruta_contrato, "r", encoding="utf-8") as f:
        datos = yaml.safe_load(f)
    return [col["nombre"] for col in datos["columnas"]]


def validar_archivo_txt(
    ruta_txt: Path, columnas_esperadas: list[str], separador: str = "|"
) -> tuple[bool, str]:
    """
    Lee únicamente la cabecera (primera línea) del archivo TXT y valida las columnas.
    Retorna (es_valido, mensaje_resumen).
    """
    if not ruta_txt.exists():
        return False, f"El archivo no existe: {ruta_txt}"

    try:
        with open(ruta_txt, "r", encoding="utf-8-sig", errors="replace") as f:
            primera_linea = f.readline().strip()
    except Exception as e:
        return False, f"Error al abrir {ruta_txt.name}: {e}"

    if not primera_linea:
        return False, f"{ruta_txt.name} está vacío."

    columnas_observadas = [col.strip() for col in primera_linea.split(separador)]
    
    # Aplicar reglas de alias (ej. ID_BENEFICIARIO -> CIP_ENCRIPTADO)
    alias_aplicados = []
    columnas_normalizadas = []
    for c in columnas_observadas:
        if c in ALIAS_COLUMNAS:
            alias_aplicados.append(f"{c} -> {ALIAS_COLUMNAS[c]}")
            columnas_normalizadas.append(ALIAS_COLUMNAS[c])
        else:
            columnas_normalizadas.append(c)

    set_esperadas = set(columnas_esperadas)
    set_observadas = set(columnas_normalizadas)

    faltantes = [c for c in columnas_esperadas if c not in set_observadas]
    excedentes = [c for c in columnas_normalizadas if c not in set_esperadas]

    detalles = []
    if alias_aplicados:
        detalles.append(f"Alias: [{', '.join(alias_aplicados)}]")
    if faltantes:
        detalles.append(f"Faltan {len(faltantes)} cols: {faltantes[:3]}...")
    if excedentes:
        detalles.append(f"Exceden {len(excedentes)} cols: {excedentes[:3]}...")

    es_valido = (len(faltantes) == 0 and len(excedentes) == 0 and len(columnas_normalizadas) == len(columnas_esperadas))

    if es_valido:
        msg = f"✓ {ruta_txt.name}: 129 columnas verificadas con éxito."
        if alias_aplicados:
            msg += f" ({', '.join(alias_aplicados)})"
        return True, msg
    else:
        msg = f"✗ {ruta_txt.name}: Discrepancia de esquema ({len(columnas_normalizadas)} cols encontradas). {'; '.join(detalles)}"
        return False, msg


def main():
    parser = argparse.ArgumentParser(description="Paso 0: Validador de Esquema GRD")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directorio con los archivos .txt de FONASA",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config/contrato_esquema.yaml",
        help="Ruta al archivo contrato_esquema.yaml",
    )
    args = parser.parse_args()

    dir_datos = Path(args.data_dir)
    ruta_contrato = Path(args.config_path)

    # Si data/raw está vacío, sugerir o revisar ubicación de respaldo
    archivos = sorted(dir_datos.glob("*.txt"))
    if not archivos:
        # Fallback de conveniencia para entorno local
        dir_alternativo = Path("../Seminario/data seminario")
        if dir_alternativo.exists():
            print(f"[AVISO] No hay archivos en '{dir_datos}'. Usando directorio de muestra: '{dir_alternativo}'")
            archivos = sorted(dir_alternativo.glob("*.txt"))

    if not archivos:
        print(f"[ERROR] No se encontraron archivos .txt en '{dir_datos}'.")
        print("Coloca los archivos de FONASA en data/raw/ o indica la ruta con --data-dir")
        sys.exit(1)

    columnas_esperadas = cargar_columnas_esperadas(ruta_contrato)
    print("=" * 70)
    print(f"VALIDACIÓN DE ESQUEMA GRD (129 columnas requeridas)")
    print(f"Directorio de datos: {archivos[0].parent}")
    print(f"Archivos encontrados: {len(archivos)}")
    print("=" * 70)

    todos_validos = True
    for ruta in archivos:
        valido, msg = validar_archivo_txt(ruta, columnas_esperadas)
        print(msg)
        if not valido:
            todos_validos = False

    print("=" * 70)
    if todos_validos:
        print("✓ Todos los archivos cumplen estrictamente el contrato de 129 columnas.")
        sys.exit(0)
    else:
        print("✗ Se detectaron errores de esquema. Revisa los archivos indicados.")
        sys.exit(1)


if __name__ == "__main__":
    main()
