"""Módulo de comorbilidades Elixhauser (Quan et al., 2005) y Score Van Walraven.

Procesa diagnósticos secundarios (DIAGNOSTICO2 a DIAGNOSTICO35) para derivar:
- 31 variables binarias ELIX_01 a ELIX_31.
- N_COMORB_ELIX: Conteo total de comorbilidades activas.
- SCORE_VANWALRAVEN: Índice ponderado de mortalidad según Van Walraven (2009).
"""

from pathlib import Path
import polars as pl

# 31 Categorías canónicas de Elixhauser (Quan 2005)
CATEGORIAS_ELIXHAUSER = [f"ELIX_{i:02d}" for i in range(1, 32)]

# Pesos oficiales de Van Walraven (2009)
PESOS_VAN_WALRAVEN = {
    "ELIX_01": 7,   # Insuficiencia cardíaca congestiva
    "ELIX_02": 5,   # Arritmias cardíacas
    "ELIX_03": -1,  # Valvulopatía
    "ELIX_04": 4,   # Trastornos de la circulación pulmonar
    "ELIX_05": 2,   # Enfermedad vascular periférica
    "ELIX_06": 0,   # Hipertensión no complicada
    "ELIX_07": 0,   # Hipertensión complicada
    "ELIX_08": 7,   # Parálisis
    "ELIX_09": 6,   # Otros trastornos neurológicos
    "ELIX_10": 3,   # Enfermedad pulmonar crónica
    "ELIX_11": 0,   # Diabetes no complicada
    "ELIX_12": 0,   # Diabetes complicada
    "ELIX_13": 0,   # Hipotiroidismo
    "ELIX_14": 5,   # Insuficiencia renal
    "ELIX_15": 11,  # Enfermedad hepática
    "ELIX_16": 0,   # Úlcera péptica
    "ELIX_17": 0,   # VIH/SIDA
    "ELIX_18": 9,   # Linfoma
    "ELIX_19": 12,  # Cáncer metastásico
    "ELIX_20": 4,   # Tumor sólido sin metástasis
    "ELIX_21": 0,   # Artritis reumatoide / conectivopatías
    "ELIX_22": 3,   # Coagulopatía
    "ELIX_23": -4,  # Obesidad
    "ELIX_24": 6,   # Pérdida de peso
    "ELIX_25": 5,   # Trastornos hidroelectrolíticos
    "ELIX_26": -2,  # Anemia por hemorragia
    "ELIX_27": -2,  # Anemia por deficiencia
    "ELIX_28": 0,   # Abuso de alcohol
    "ELIX_29": -7,  # Abuso de drogas
    "ELIX_30": 0,   # Psicosis
    "ELIX_31": -3,  # Depresión
}


def cargar_reglas_mapeo(config_path: Path | str = "config/mapeo_elixhauser_quan2005.csv") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Carga y expande las reglas Quan (2005) a tablas de prefijos de longitud 3 y 4."""
    df_map = pl.read_csv(config_path, comment_prefix="#")
    rules_3 = []
    rules_4 = []

    for row in df_map.iter_rows(named=True):
        cat = row["categoria_id"]
        tipo = row["tipo_regla"]
        
        if tipo in ("EXACTO", "PREFIJO"):
            pfx = row["prefijo_cie10"].strip()
            if len(pfx) == 3:
                rules_3.append((pfx, cat))
            elif len(pfx) == 4:
                rules_4.append((pfx, cat))
        elif tipo == "RANGO":
            d = row["rango_desde"].strip()
            h = row["rango_hasta"].strip()
            pfx = d[0]
            start_num = int(d[1:])
            end_num = int(h[1:])
            num_len = len(d) - 1
            for n in range(start_num, end_num + 1):
                code = f"{pfx}{n:0{num_len}d}"
                if len(code) == 3:
                    rules_3.append((code, cat))
                elif len(code) == 4:
                    rules_4.append((code, cat))

    map_3 = pl.DataFrame(rules_3, orient="row", schema={"pfx3": pl.String, "categoria": pl.String}).unique()
    map_4 = pl.DataFrame(rules_4, orient="row", schema={"pfx4": pl.String, "categoria": pl.String}).unique()
    return map_3, map_4


def calcular_elixhauser(df: pl.DataFrame, config_path: Path | str = "config/mapeo_elixhauser_quan2005.csv") -> pl.DataFrame:
    """Calcula las 31 binarias Elixhauser, N_COMORB_ELIX y SCORE_VANWALRAVEN sobre un DataFrame con ID_EPISODIO."""
    map_3, map_4 = cargar_reglas_mapeo(config_path)

    # Identificar columnas de diagnósticos secundarios (DIAGNOSTICO2 .. DIAGNOSTICO35)
    # DIAGNOSTICO1 queda ESTRICTAMENTE EXCLUIDO para evitar circularidad con el motivo de ingreso
    dx_cols = [c for c in df.columns if c.startswith("DIAGNOSTICO") and c != "DIAGNOSTICO1"]
    if not dx_cols:
        raise ValueError("No se encontraron columnas de diagnósticos secundarios (DIAGNOSTICO2..35).")

    # Formato largo transitorio
    long = (
        df.select(["ID_EPISODIO"] + dx_cols)
        .unpivot(index="ID_EPISODIO", on=dx_cols, value_name="cie10")
        .filter(
            pl.col("cie10").is_not_null()
            & (pl.col("cie10").str.strip_chars() != "")
            & (~pl.col("cie10").str.to_uppercase().str.starts_with("Z"))
        )
        .with_columns([
            pl.col("cie10").str.to_uppercase().str.slice(0, 3).alias("pfx3"),
            pl.col("cie10").str.to_uppercase().str.slice(0, 4).alias("pfx4"),
        ])
    )

    # Coincidencias por prefijo 3 y 4
    m3 = long.join(map_3, on="pfx3", how="inner").select(["ID_EPISODIO", "categoria"])
    m4 = long.join(map_4, on="pfx4", how="inner").select(["ID_EPISODIO", "categoria"])
    aciertos = pl.concat([m3, m4]).unique()

    # Agrupar categorías presentes por episodio
    por_episodio = (
        aciertos.group_by("ID_EPISODIO")
        .agg(pl.col("categoria").alias("CATS"))
    )

    # Unir de vuelta con el DataFrame base y proyectar las 31 binarias
    cats_col = pl.col("CATS")
    proyecciones = [
        cats_col.list.contains(cat).fill_null(False).alias(cat)
        for cat in CATEGORIAS_ELIXHAUSER
    ]

    res = (
        df.select(["ID_EPISODIO"])
        .join(por_episodio, on="ID_EPISODIO", how="left")
        .with_columns(proyecciones)
        .drop("CATS")
    )

    # Conteo de comorbilidades activas
    res = res.with_columns(
        pl.sum_horizontal([pl.col(c).cast(pl.Int32) for c in CATEGORIAS_ELIXHAUSER]).alias("N_COMORB_ELIX")
    )

    # Cálculo de SCORE_VANWALRAVEN con jerarquías oficiales:
    # 1. ELIX_19 (Metástasis) domina a ELIX_20 (Tumor sólido)
    # 2. ELIX_12 (Diabetes complicada) domina a ELIX_11 (Diabetes simple)
    # Expresión ponderada
    vw_terms = []
    for cat in CATEGORIAS_ELIXHAUSER:
        w = PESOS_VAN_WALRAVEN.get(cat, 0)
        if w == 0:
            continue
        if cat == "ELIX_20":
            # Si ELIX_19 está activa, ELIX_20 se suprime del score
            cond = pl.col("ELIX_20") & (~pl.col("ELIX_19"))
            vw_terms.append(pl.when(cond).then(w).otherwise(0))
        else:
            vw_terms.append(pl.when(pl.col(cat)).then(w).otherwise(0))

    res = res.with_columns(
        pl.sum_horizontal(vw_terms).cast(pl.Int32).alias("SCORE_VANWALRAVEN")
    )

    return res
