"""Figuras de la evaluación (gráfico de embudo por tarea)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import embudo

NAVY, GR, OR, RD, BLUE = "#1F3864", "#7F7F7F", "#DD690A", "#9F3B38", "#2E5597"


def grafico_embudo(tabla: pd.DataFrame, tau2: float, disp: float, titulo: str, etiqueta_x: str, ruta: Path, aviso: str = "") -> None:
    n_ocultos = int(tabla["oe"].isna().sum())
    tabla = tabla[tabla["oe"].notna()].reset_index(drop=True)      # los hospitales con menos de 10 eventos no se grafican (privacidad)
    esp = tabla["esperado"].to_numpy(float)
    x = np.logspace(np.log10(esp.min() * 0.8), np.log10(esp.max() * 1.2), 200)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for z, ls in ((1.96, "--"), (3.09, ":")):
        lo, hi = embudo.limites(x, tau2, z, disp)
        ax.plot(x, lo, color=OR, ls=ls, lw=1.8)
        ax.plot(x, hi, color=OR, ls=ls, lw=1.8)
        lo0, hi0 = embudo.limites(x, 0.0, z, disp)
        ax.plot(x, lo0, color=GR, ls=ls, lw=1.0)
        ax.plot(x, hi0, color=GR, ls=ls, lw=1.0)
    dentro = tabla["zona"] == "dentro de lo esperado"
    ax.scatter(esp[dentro], tabla["oe"][dentro], s=34, color=BLUE, zorder=3, label="dentro de lo esperado")
    ax.scatter(esp[~dentro], tabla["oe"][~dentro], s=46, color=RD, zorder=4, label="fuera del límite del 95 %")
    ax.axhline(1, color=NAVY, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(etiqueta_x, fontsize=9)
    ax.set_ylabel("O/E (observado / esperado)", fontsize=9)
    ax.set_title(titulo, fontsize=10, color=NAVY, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    if n_ocultos:
        ax.text(0.99, 0.03, f"{n_ocultos} hospital(es) con menos de 10 eventos no se grafican", transform=ax.transAxes, fontsize=8, color=GR, style="italic", ha="right")
    if aviso:
        ax.text(0.01, 0.03, aviso, transform=ax.transAxes, fontsize=8.5, color=RD, style="italic")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=160)
    plt.close(fig)
