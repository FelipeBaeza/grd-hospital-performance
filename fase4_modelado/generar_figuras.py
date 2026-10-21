"""Gráficos del informe de la Fase 4, a partir de las tablas de salidas/tablas."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
NAVY, BLUE, LB, GR, OR, GN, RD = "#1F3864", "#2E5597", "#EAF1FB", "#7F7F7F", "#DD690A", "#2E7D32", "#9F3B38"
RAIZ = Path(__file__).resolve().parent
T = RAIZ / "salidas" / "tablas"
NOMBRES = {"constante": "Solo la media", "linea_base": "Línea base", "lineal": "Regresión lineal", "lightgbm": "LightGBM", "xgboost": "XGBoost", "van_walraven_referencia": "van Walraven + edad"}
COLORES = {"constante": "#BFBFBF", "linea_base": GR, "lineal": "#A6A6A6", "lightgbm": BLUE, "xgboost": OR, "van_walraven_referencia": "#D9D9D9"}


def limpiar(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE")
    ax.set_axisbelow(True)


def nombre_linea_base(tarea):
    return "Regresión logística" if tarea == "mortalidad" else "Regresión Poisson"


def comparacion(F):
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.9))
    for ax, tarea, col, titulo, mejor in ((axs[0], "mortalidad", "media_pliegues", "Mortalidad: PR AUC en los pliegues (mayor es mejor)", "max"),
                                          (axs[1], "estadia", "media_pliegues", "Estadía: devianza de Poisson en los pliegues (menor es mejor)", "min")):
        t = pd.read_csv(T / f"comparacion_{tarea}.csv")
        t = t[t["candidato"].isin(["constante", "linea_base", "lightgbm", "xgboost"])]
        etiquetas = [nombre_linea_base(tarea) if c == "linea_base" else NOMBRES[c] for c in t["candidato"]]
        ax.bar(etiquetas, t[col], color=[COLORES[c] for c in t["candidato"]], width=0.6)
        for i, v in enumerate(t[col]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=NAVY)
        ax.set_title(titulo, fontsize=9.5, color=NAVY, loc="left"); ax.tick_params(axis="x", labelsize=8); limpiar(ax)
        if mejor == "max":
            ax.set_ylim(0, t[col].max() * 1.15)
        else:
            ax.set_ylim(0, t[col].max() * 1.12)
    fig.tight_layout(); fig.savefig(F / "f4_comparacion.png", dpi=170); plt.close(fig)


def ajuste(F):
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.5))
    for ax, tarea, ylab, mejor in ((axs[0], "mortalidad", "mejor PR AUC acumulado", "max"), (axs[1], "estadia", "mejor devianza acumulada", "min")):
        for tipo in ("lightgbm", "xgboost"):
            ruta = T / f"ajuste_pruebas_{tarea}_{tipo}.csv"
            if not ruta.exists():
                continue
            d = pd.read_csv(ruta).sort_values("number")
            v = d["value"].fillna(np.nan)
            acum = v.cummax() if mejor == "max" else v.cummin()
            ax.plot(d["number"] + 1, acum, color=COLORES[tipo], lw=2, label=NOMBRES[tipo])
            ax.scatter(d["number"] + 1, v, color=COLORES[tipo], s=12, alpha=0.35)
        ax.set_xlabel("prueba de hiperparámetros", fontsize=9); ax.set_ylabel(ylab, fontsize=9); limpiar(ax); ax.legend(frameon=False, fontsize=8.5)
        ax.set_title("Mortalidad" if tarea == "mortalidad" else "Estadía", fontsize=10, color=NAVY, loc="left")
    fig.tight_layout(); fig.savefig(F / "f4_ajuste.png", dpi=170); plt.close(fig)


def calibracion(F):
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.1))
    for ax, tarea in zip(axs, ("mortalidad", "estadia")):
        t = pd.read_csv(T / f"calibracion_deciles_{tarea}.csv")
        mx = max(t["predicho"].max(), t["observado"].max()) * 1.05
        ax.plot([0, mx], [0, mx], color="#BFBFBF", lw=1.2, ls="--", label="calibración perfecta")
        for version, color, et in (("sin_calibrar", OR, "antes de recalibrar"), ("recalibrada_fuera_de_muestra", GN, "recalibrada")):
            d = t[t["version"] == version]
            ax.plot(d["predicho"], d["observado"], marker="o", color=color, lw=1.8, label=et)
        ax.set_xlabel("predicho (probabilidad)" if tarea == "mortalidad" else "predicho (días)", fontsize=9)
        ax.set_ylabel("observado", fontsize=9); limpiar(ax); ax.legend(frameon=False, fontsize=8.5)
        ax.set_title(("Mortalidad" if tarea == "mortalidad" else "Estadía") + " en 2023, por decil de predicción", fontsize=9.5, color=NAVY, loc="left")
        if tarea == "mortalidad":
            ax.set_xscale("symlog", linthresh=0.01); ax.set_yscale("symlog", linthresh=0.01)
    fig.tight_layout(); fig.savefig(F / "f4_calibracion.png", dpi=170); plt.close(fig)


def contribuciones(F):
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, tarea in zip(axs, ("mortalidad", "estadia")):
        t = pd.read_csv(T / f"contribuciones_{tarea}.csv").head(10)[::-1]
        ax.barh(t["variable"], t["proporcion"] * 100, color=BLUE if tarea == "mortalidad" else OR, height=0.6)
        for i, v in enumerate(t["proporcion"] * 100):
            ax.text(v + 0.3, i, f"{v:.1f} %", va="center", fontsize=8, color=NAVY)
        ax.set_title(("Mortalidad" if tarea == "mortalidad" else "Estadía") + ": peso de cada variable", fontsize=9.5, color=NAVY, loc="left")
        ax.tick_params(axis="y", labelsize=8); limpiar(ax); ax.grid(axis="x", color="#EEEEEE"); ax.grid(axis="y", visible=False)
        ax.set_xlabel("% de la contribución total", fontsize=8.5)
    fig.tight_layout(); fig.savefig(F / "f4_contribuciones.png", dpi=170); plt.close(fig)


def coherencia(F):
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.6))
    for ax, nombre, titulo in ((axs[0], "edad", "por tramo de edad"), (axs[1], "comorbilidad", "por número de condiciones de Elixhauser")):
        t = pd.read_csv(T / f"coherencia_mortalidad_{nombre}.csv")
        x = np.arange(len(t))
        ax.bar(x - 0.19, t["observado"] * 100, 0.38, color=GR, label="observado"); ax.bar(x + 0.19, t["predicho"] * 100, 0.38, color=BLUE, label="predicho")
        ax.set_xticks(x); ax.set_xticklabels(t["grupo"].astype(str), fontsize=8.5); ax.set_ylabel("mortalidad (%)", fontsize=9)
        ax.set_title("Mortalidad en 2023 " + titulo, fontsize=9.5, color=NAVY, loc="left"); limpiar(ax); ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(F / "f4_coherencia.png", dpi=170); plt.close(fig)


def hospital(F):
    t = pd.read_csv(T / "integridad_identificacion_hospital.csv")
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    etiquetas = ["Con las 40 variables", "Sin las 4 condicionales"]
    vals = [t.loc[t["con_condicionales"], "exactitud_top1"].iloc[0] * 100, t.loc[~t["con_condicionales"], "exactitud_top1"].iloc[0] * 100]
    ax.bar(etiquetas, vals, color=[BLUE, GN], width=0.5)
    may = t["clase_mayoritaria"].iloc[0] * 100
    ax.axhline(may, color=RD, ls="--", lw=1.3); ax.text(1.28, may + 0.4, f"adivinar siempre el hospital más grande: {may:.1f} %", ha="right", fontsize=8.5, color=RD)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, f"{v:.1f} %", ha="center", fontsize=9, color=NAVY)
    ax.set_ylabel("exactitud al adivinar el hospital (%)", fontsize=9); ax.set_ylim(0, max(vals + [may]) * 1.25); limpiar(ax)
    fig.tight_layout(); fig.savefig(F / "f4_hospital.png", dpi=170); plt.close(fig)


def tiempos(F):
    t = pd.read_csv(T / "tiempos_etapas_fase4.csv")
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    ax.bar(t["etapa"], t["segundos"] / 60, color=BLUE, width=0.55)
    for i, v in enumerate(t["segundos"] / 60):
        ax.text(i, v, f"{v:.0f} min" if v >= 1 else f"{v*60:.0f} s", ha="center", va="bottom", fontsize=8.5, color=NAVY)
    ax.set_ylabel("minutos", fontsize=9); ax.tick_params(axis="x", labelsize=8); limpiar(ax)
    ax2 = ax.twinx(); ax2.plot(t["etapa"], t["pico_ram_gb"], color=OR, marker="o", lw=2); ax2.set_ylim(0, 8); ax2.set_ylabel("pico de memoria (GB)", fontsize=9, color=OR); ax2.spines["top"].set_visible(False)
    ax2.axhline(7, color=RD, ls="--", lw=1)
    fig.tight_layout(); fig.savefig(F / "f4_tiempos.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--salida", default=str(RAIZ / "salidas" / "figuras")); a = ap.parse_args()
    F = Path(a.salida); F.mkdir(parents=True, exist_ok=True)
    for fn in (comparacion, ajuste, calibracion, contribuciones, coherencia, hospital, tiempos):
        fn(F)
    print("figuras listas en", F)
