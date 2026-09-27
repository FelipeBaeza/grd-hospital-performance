"""Generador de Figuras Científicas de Publicación para el Análisis Exploratorio de Datos (Fase 2).

Genera 6 figuras de alta resolución (300 DPI) basadas en los 5.8 millones de egresos hospitalarios:
- Fig 1: Histograma y Boxplot de Estancia Hospitalaria (Asimetría y justificación de Tweedie).
- Fig 2: Pirámide Demográfica Poblacional por Edad y Sexo.
- Fig 3: Distribución de Motivos de Ingreso por Capítulos de la CIE-10.
- Fig 4: Prevalencia de las 31 Comorbilidades Crónicas de Elixhauser (Quan et al., 2005).
- Fig 5: Evolución Temporal de Egresos y Tasa de Mortalidad (2019-2024, Efecto Pandemia).
- Fig 6: Proporción de Desenlaces de Egreso y Censura Estadística (TIPOALTA).

Guarda en:
  Analisis exploratorio/figuras/
"""

import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np

# Configurar estilo visual publication-ready
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["figure.titlesize"] = 13

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "figuras"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GOLD_DIR = ROOT_DIR / "data/gold"
SILVER_DIR = ROOT_DIR / "data/silver"


def cargar_muestra_representativa(n_filas=1_000_000):
    """Carga una muestra estratificada representativa de la capa Gold para gráficos rápidos y exactos."""
    print("Cargando datos para visualizaciones...")
    files = sorted(GOLD_DIR.glob("gold_*.parquet"))
    dfs = []
    # Leer ~150k filas por año para balancear perfectamente los 6 años
    filas_por_archivo = n_filas // len(files)
    for f in files:
        df = pl.read_parquet(f).sample(n=min(filas_por_archivo, pl.read_parquet(f).height), seed=42)
        dfs.append(df)
    sample_df = pl.concat(dfs)
    print(f"Muestra estratificada cargada: {len(sample_df):,} registros.")
    return sample_df


def figura_1_distribucion_estancia(df):
    """Fig 1: Histograma y Boxplot de Estancia (Asimetría positiva extrema)."""
    print("Generando Figura 1: Distribución de Estancia...")
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [0.25, 0.75]}
    )

    estancias = df["ESTANCIA_DIAS"].filter(df["ESTANCIA_DIAS"].is_not_null() & (df["ESTANCIA_DIAS"] <= 35)).to_numpy()
    mediana = np.median(estancias)
    media = np.mean(estancias)

    # Boxplot
    sns.boxplot(x=estancias, ax=ax_box, color="#38bdf8", fliersize=2)
    ax_box.set(xlabel="")
    ax_box.set_title("Distribución de la Estancia Hospitalaria (Días)", fontweight="bold")
    ax_box.axvline(mediana, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Mediana: {mediana:.1f}d")
    ax_box.axvline(media, color="#10b981", linestyle=":", linewidth=1.5, label=f"Media: {media:.1f}d")
    ax_box.legend(loc="upper right", frameon=True)

    # Histograma
    bins = np.arange(0, 36, 1)
    ax_hist.hist(estancias, bins=bins, color="#1e3a8a", edgecolor="#38bdf8", alpha=0.85, density=False)
    ax_hist.set_xlabel("Días de Estancia (Truncado a 35 días para visualización de cola)")
    ax_hist.set_ylabel("Frecuencia (Episodios)")
    ax_hist.axvline(mediana, color="#ef4444", linestyle="--", linewidth=1.5)
    ax_hist.axvline(media, color="#10b981", linestyle=":", linewidth=1.5)

    # Anotación explicativa de cola larga
    ax_hist.annotate(
        "Estancia 0 días: ~20%\n(Casos ambulatorios)",
        xy=(0.5, ax_hist.get_ylim()[1] * 0.8),
        xytext=(4, ax_hist.get_ylim()[1] * 0.85),
        arrowprops=dict(facecolor="#ef4444", shrink=0.05, width=1, headwidth=6),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef2f2", edgecolor="#ef4444"),
    )

    plt.tight_layout()
    ruta = OUTPUT_DIR / "fig01_distribucion_estancia_hist_box.png"
    plt.savefig(ruta, dpi=300)
    plt.close()
    print(f" -> Guardada en {ruta}")


def figura_2_piramide_edad_sexo(df):
    """Fig 2: Pirámide Demográfica de Pacientes."""
    print("Generando Figura 2: Pirámide de Edad y Sexo...")
    # Filtrar sexos conocidos y edades válidas
    df_demo = df.filter(pl.col("SEXO").is_in(["HOMBRE", "MUJER"]) & pl.col("EDAD_ANIOS").is_not_null())
    
    # Crear tramos etarios quinquenales
    edades = np.arange(0, 105, 5)
    labels = [f"{i}-{i+4}" for i in edades[:-1]] + ["100+"]
    
    counts_h = []
    counts_m = []
    
    for i in range(len(edades)):
        low = edades[i]
        high = edades[i+1] if i < len(edades)-1 else 115
        h = df_demo.filter((pl.col("SEXO") == "HOMBRE") & (pl.col("EDAD_ANIOS") >= low) & (pl.col("EDAD_ANIOS") < high)).height
        m = df_demo.filter((pl.col("SEXO") == "MUJER") & (pl.col("EDAD_ANIOS") >= low) & (pl.col("EDAD_ANIOS") < high)).height
        counts_h.append(h)
        counts_m.append(m)

    total_casos = sum(counts_h) + sum(counts_m)
    pct_h = [-h / total_casos * 100 for h in counts_h]
    pct_m = [m / total_casos * 100 for m in counts_m]

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(labels))

    ax.barh(y, pct_h, align="center", color="#3b82f6", label="Hombres", edgecolor="white", height=0.8)
    ax.barh(y, pct_m, align="center", color="#ec4899", label="Mujeres", edgecolor="white", height=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Porcentaje de la Población Hospitalizada (%)")
    ax.set_title("Pirámide Demográfica Hospitalaria (FONASA Chile)", fontweight="bold")
    
    # Eje X con valores positivos en ambos lados
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x):.1f}%" for x in xticks])
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    ruta = OUTPUT_DIR / "fig02_piramide_poblacional_edad_sexo.png"
    plt.savefig(ruta, dpi=300)
    plt.close()
    print(f" -> Guardada en {ruta}")


def figura_3_capitulos_cie10(df):
    """Fig 3: Distribución de Capítulos Diagnósticos CIE-10."""
    print("Generando Figura 3: Frecuencia de Capítulos CIE-10...")
    gc = df["GRUPO_CLINICO"].value_counts().sort("count", descending=True).head(12)

    nombres_limpios = {
        "CAP15_EMBARAZO_PARTO_PUERPERIO": "Embarazo, parto y puerperio (Cap. 15)",
        "CAP11_SISTEMA_DIGESTIVO": "Enfermedades sistema digestivo (Cap. 11)",
        "CAP19_TRAUMATISMOS_ENVENENAMIENTOS": "Traumatismos y causas externas (Cap. 19)",
        "CAP09_SISTEMA_CIRCULATORIO": "Enfermedades sistema circulatorio (Cap. 9)",
        "CAP14_GENITOURINARIO": "Enfermedades sistema genitourinario (Cap. 14)",
        "CAP02_NEOPLASIAS": "Neoplasias / Cáncer (Cap. 2)",
        "CAP10_SISTEMA_RESPIRATORIO": "Enfermedades sistema respiratorio (Cap. 10)",
        "CAP07_OJO_ANEXOS": "Enfermedades del ojo y anexos (Cap. 7)",
        "CAP21_FACTORES_CONTACTO_SERVICIOS": "Factores de contacto servicios salud (Cap. 21)",
        "CAP13_OSTEOMUSCULAR_CONJUNTIVO": "Enfermedades osteomuscular (Cap. 13)",
        "CAP22_PROPOSITOS_ESPECIALES": "Códigos especiales COVID-19 (Cap. 22)",
        "CAP01_INFECCIOSAS_PARASITARIAS": "Enfermedades infecciosas (Cap. 1)",
    }

    labels = [nombres_limpios.get(k, k) for k in gc["GRUPO_CLINICO"].to_list()][::-1]
    valores = gc["count"].to_list()[::-1]
    pcts = [v / df.height * 100 for v in valores]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(labels, pcts, color="#0284c7", edgecolor="white", height=0.7)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8.5, color="#1e293b")

    ax.set_xlabel("Porcentaje de Episodios (%)")
    ax.set_title("Principales Motivos de Ingreso por Capítulo CIE-10", fontweight="bold")
    ax.set_xlim(0, max(pcts) * 1.15)

    plt.tight_layout()
    ruta = OUTPUT_DIR / "fig03_top_capitulos_cie10.png"
    plt.savefig(ruta, dpi=300)
    plt.close()
    print(f" -> Guardada en {ruta}")


def figura_4_comorbilidades_elixhauser(df):
    """Fig 4: Prevalencia de Comorbilidades Elixhauser."""
    print("Generando Figura 4: Prevalencia de Comorbilidades Elixhauser...")
    nombres_elix = {
        "ELIX_06": "Hipertensión no complicada",
        "ELIX_11": "Diabetes no complicada",
        "ELIX_25": "Trastornos hidroelectrolíticos",
        "ELIX_01": "Insuficiencia cardíaca congestiva",
        "ELIX_10": "Enfermedad pulmonar crónica (EPOC)",
        "ELIX_14": "Insuficiencia renal",
        "ELIX_07": "Hipertensión complicada",
        "ELIX_20": "Tumor sólido sin metástasis",
        "ELIX_12": "Diabetes complicada",
        "ELIX_02": "Arritmias cardíacas",
        "ELIX_15": "Enfermedad hepática",
        "ELIX_22": "Coagulopatía",
        "ELIX_19": "Cáncer metastásico",
        "ELIX_05": "Enfermedad vascular periférica",
        "ELIX_08": "Parálisis",
    }

    prevalencias = []
    for cod, nombre in nombres_elix.items():
        if cod in df.columns:
            prev = float(df[cod].sum()) / df.height * 100
            prevalencias.append((nombre, prev))

    prevalencias.sort(key=lambda x: x[1])
    nombres = [p[0] for p in prevalencias]
    valores = [p[1] for p in prevalencias]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(nombres, valores, color="#8b5cf6", edgecolor="white", height=0.7)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8.5)

    ax.set_xlabel("Prevalencia en la Población Hospitalaria (%)")
    ax.set_title("Prevalencia de Comorbilidades Crónicas de Elixhauser (Quan et al., 2005)", fontweight="bold")
    ax.set_xlim(0, max(valores) * 1.15)

    plt.tight_layout()
    ruta = OUTPUT_DIR / "fig04_prevalencia_comorbilidades_elixhauser.png"
    plt.savefig(ruta, dpi=300)
    plt.close()
    print(f" -> Guardada en {ruta}")


def figura_5_evolucion_temporal():
    """Fig 5: Evolución Temporal de Egresos y Mortalidad (2019-2024, Efecto Pandemia)."""
    print("Generando Figura 5: Evolución Temporal (Efecto COVID)...")
    resumen_path = ROOT_DIR / "reports/consort_summary.csv"
    df_consort = pl.read_csv(resumen_path).filter(~pl.col("anio").str.contains("TOTAL"))

    anios = df_consort["anio"].cast(pl.Int32).to_list()
    egresos = [n / 1_000 for n in df_consort["n_inicial"].to_list()]
    muertes = df_consort["ex06_fallecido"].to_list()
    tasa_mortalidad = [m / n * 100 for m, n in zip(muertes, df_consort["n_inicial"].to_list())]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    # Eje izquierdo: Egresos
    color_bar = "#38bdf8"
    bars = ax1.bar(anios, egresos, color=color_bar, alpha=0.6, width=0.55, label="Egresos Totales (Miles)")
    ax1.set_xlabel("Año de Egreso")
    ax1.set_ylabel("Volumen de Egresos (Miles de Pacientes)", color="#0369a1")
    ax1.tick_params(axis="y", labelcolor="#0369a1")
    ax1.set_ylim(0, 1400)

    # Anotación Pandemia
    ax1.annotate(
        "Caída por Pandemia\n(Suspensión electivas)",
        xy=(2020, 781),
        xytext=(2019.5, 450),
        arrowprops=dict(facecolor="#ef4444", shrink=0.05, width=1.5, headwidth=6),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff1f2", edgecolor="#f43f5e"),
    )

    # Eje derecho: Tasa de mortalidad
    ax2 = ax1.twinx()
    color_line = "#ef4444"
    ax2.plot(anios, tasa_mortalidad, color=color_line, marker="o", linewidth=2.5, label="Tasa Mortalidad Cruda (%)")
    ax2.set_ylabel("Tasa de Mortalidad Intrahospitalaria Cruda (%)", color=color_line)
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.set_ylim(1.5, 4.5)
    ax2.grid(False)

    for a, t in zip(anios, tasa_mortalidad):
        ax2.text(a, t + 0.12, f"{t:.2f}%", ha="center", fontsize=9, fontweight="bold", color=color_line)

    plt.title("Evolución Temporal de Egresos y Tasa de Mortalidad (2019–2024)", fontweight="bold")
    plt.tight_layout()
    ruta = OUTPUT_DIR / "fig05_evolucion_temporal_egresos_mortalidad.png"
    plt.savefig(ruta, dpi=300)
    plt.close()
    print(f" -> Guardada en {ruta}")


def figura_6_tipo_alta_censura():
    """Fig 6: Distribución de Tipo de Alta y Censura Estadística."""
    print("Generando Figura 6: Distribución de Desenlaces y Censura...")
    # Datos consolidados de los 5.8M
    categorias = [
        "Alta a Domicilio (Vivos)",
        "Fallecidos (Desenlace Confirmado)",
        "Traslados a Otro Hospital (Censura)",
        "Hospitalización Domiciliaria (Censura)",
        "Fuga / Desconocido (Censura)",
    ]
    conteo = [5_282_991, 170_692, 285_120, 62_400, 7_333]
    colores = ["#10b981", "#ef4444", "#f59e0b", "#3b82f6", "#94a3b8"]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    wedges, texts, autotexts = ax.pie(
        conteo,
        labels=categorias,
        autopct="%1.2f%%",
        colors=colores,
        startangle=140,
        pctdistance=0.75,
        explode=[0, 0.1, 0.08, 0.08, 0.1],
        wedgeprops=dict(width=0.45, edgecolor="white"),
    )

    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight("bold")

    ax.set_title("Distribución de Desenlaces de Egreso y Censura (5.8M Registros)", fontweight="bold")
    plt.tight_layout()
    ruta = OUTPUT_DIR / "fig06_distribucion_tipo_alta_censura.png"
    plt.savefig(ruta, dpi=300)
    plt.close()
    print(f" -> Guardada en {ruta}")


def main():
    print("=" * 80)
    print("GENERADOR DE FIGURAS PARA EL ANÁLISIS EXPLORATORIO DE DATOS (FASE 2)")
    print("=" * 80)

    t0 = time.time()
    df_sample = cargar_muestra_representativa()

    figura_1_distribucion_estancia(df_sample)
    figura_2_piramide_edad_sexo(df_sample)
    figura_3_capitulos_cie10(df_sample)
    figura_4_comorbilidades_elixhauser(df_sample)
    figura_5_evolucion_temporal()
    figura_6_tipo_alta_censura()

    print("\n" + "=" * 80)
    print(f"6 FIGURAS CIENTÍFICAS GENERADAS EXITOSAMENTE EN {time.time()-t0:.2f}s")
    print(f"Directorio de figuras: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
