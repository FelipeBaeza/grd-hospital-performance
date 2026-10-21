"""Gráficos del informe de la Fase 3. Usa solo las tablas de salidas/tablas y, para la distribución de estadía, la matriz de estadía de la capa Oro (agregada a conteos por día)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
NAVY, BLUE, LB, GR, OR, GN, RD = "#1F3864", "#2E5597", "#EAF1FB", "#7F7F7F", "#DD690A", "#2E7D32", "#9F3B38"
RAIZ = Path(__file__).resolve().parent
T = RAIZ / "salidas" / "tablas"


def limpiar(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE")
    ax.set_axisbelow(True)


def fmt(n):
    return f"{n:,.0f}".replace(",", ".")


def flujo(F):
    fig, ax = plt.subplots(figsize=(10, 3.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, t, sub, fc=LB, ec=BLUE, tc=NAVY):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h - 3.2, t, ha="center", va="center", fontsize=10.5, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center", fontsize=8, color="#404040", linespacing=1.25)

    def arr(x1, y1, x2, y2, c=BLUE):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color=c, lw=1.6))
    box(1, 22, 15, 15, "Cruda", "6 archivos .txt\n5.808.536 líneas\n(solo lectura)", fc="#F2F2F2", ec=GR, tc="#404040")
    box(21, 22, 17, 15, "Bronce", "Parquet por año\nlo mismo que el .txt,\nsin cambiar nada")
    box(43, 22, 17, 15, "Plata", "tipos, fechas y\ndiagnósticos limpios;\nvalores imposibles anulados")
    box(65, 22, 15, 15, "Cohortes", "cascada A a E\nreporte tipo CONSORT")
    box(85, 22, 14, 15, "Oro", "variables finales\nmatrices de los\ndos modelos", fc="#FDF0E4", ec=OR, tc="#8A4208")
    for a, b in [(16.4, 20.6), (38.4, 42.6), (60.4, 64.6), (80.4, 84.6)]:
        arr(a, 29.5, b, 29.5)
    box(50, 1, 22, 14, "Entrena y calibra", "2019–2022 entrena\n2023 calibra", fc="#E8F3E8", ec=GN, tc="#1E5A22")
    box(76, 1, 23, 14, "2024 sellado", "variables sí; el desenlace\nvive en una carpeta cerrada", fc="#F8E9E8", ec=RD, tc="#7A2B29")
    arr(88, 21.4, 68, 15.8, GN); arr(93, 21.4, 91, 15.8, RD)
    box(1, 1, 40, 14, "Catálogo de hospitales", "un conteo por hospital y año\nningún registro de paciente", fc="#F2F2F2", ec=GR, tc="#404040")
    arr(50, 22, 30, 15.8, GR)
    fig.tight_layout(); fig.savefig(F / "f3_flujo_capas.png", dpi=170); plt.close(fig)


def cascada(F):
    c = pl.read_csv(T / "cascada_fase3.csv").filter(pl.col("periodo").cast(pl.String).str.contains(r"^20(19|2[0-3])$"))
    g = c.group_by(["orden", "paso", "tipo", "descripcion"]).agg(pl.col("n").sum()).sort("orden")
    filas = g.to_dicts()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = len(filas)
    etiquetas = {"TOTAL": "Egresos leídos", "A": "A · cohorte dura", "B": "B · adultos", "C": "C · adultos hospitalizados", "D": "D · modelo de mortalidad", "E": "E · modelo de estadía"}
    for i, r in enumerate(filas):
        yy = y - 1 - i
        if r["tipo"] in ("total", "cohorte"):
            ax.barh(yy, r["n"], color=NAVY if r["tipo"] == "cohorte" else GR, height=0.62)
            ax.text(r["n"] + 30000, yy, fmt(r["n"]), va="center", fontsize=9, color=NAVY, fontweight="bold")
            ax.text(-30000, yy, etiquetas[r["paso"]], va="center", ha="right", fontsize=9.5, color=NAVY, fontweight="bold")
        else:
            ax.barh(yy, r["n"], color=OR, height=0.42, alpha=0.9)
            ax.text(r["n"] + 30000, yy, "−" + fmt(r["n"]), va="center", fontsize=8.5, color="#8A4208")
            nombre = {"EX01": "GRD no agrupable", "EX02": "Recién nacidos", "EX03": "Obstétricos sin comorbilidad", "EX04": "Menores de 18 o edad desconocida",
                      "EX05": "No es hospitalización", "EX06": "Censurados / sin desenlace", "EX07": "Fallecidos (salen de estadía)", "EX08": "Estadía 0 o desconocida"}[r["paso"]]
            ax.text(-30000, yy, nombre, va="center", ha="right", fontsize=8.5, color="#404040")
    ax.set_yticks([]); ax.set_xlim(0, 5.3e6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v/1e6:.0f} M"))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Cascada de cohortes, 2019–2023 (5.119.0 mil egresos leídos)".replace("5.119.0 mil", fmt(g["n"][0])), fontsize=11, color=NAVY, loc="left")
    fig.subplots_adjust(left=0.28); fig.savefig(F / "f3_cascada.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def tiempos(F):
    t = pl.read_csv(T / "tiempos_etapas.csv")
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.bar(t["etapa"], t["segundos"], color=BLUE, width=0.55)
    for x, v in zip(t["etapa"], t["segundos"]):
        ax.text(x, v + 2, f"{v:.0f} s", ha="center", fontsize=9, color=NAVY)
    ax.set_ylabel("segundos", fontsize=9); limpiar(ax)
    ax2 = ax.twinx(); ax2.plot(t["etapa"], t["pico_ram_gb"], color=OR, marker="o", lw=2); ax2.set_ylim(0, 8); ax2.set_ylabel("pico de memoria RAM (GB)", fontsize=9, color=OR)
    ax2.axhline(7, color=RD, ls="--", lw=1); ax2.text(len(t) - 0.55, 7.15, "equipo de prueba: 7 GB", ha="right", fontsize=8, color=RD)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout(); fig.savefig(F / "f3_tiempos_memoria.png", dpi=170); plt.close(fig)


def filas_por_capa(F):
    r = pl.read_csv(T / "plata_resumen.csv").sort("anio")
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    x = np.arange(r.height); ax.bar(x - 0.2, r["filas_bronce"], 0.38, color=GR, label="Bronce"); ax.bar(x + 0.2, r["filas_plata"], 0.38, color=BLUE, label="Plata")
    for i, v in enumerate(r["filas_plata"]):
        ax.text(i, v + 15000, fmt(v), ha="center", fontsize=8.5, color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(r["anio"]); ax.set_ylim(0, r["filas_plata"].max() * 1.15); limpiar(ax); ax.legend(frameon=False, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v/1e3:.0f} mil"))
    fig.tight_layout(); fig.savefig(F / "f3_filas_bronce_plata.png", dpi=170); plt.close(fig)


def faltantes(F):
    f = pl.read_csv(T / "plata_faltantes.csv", infer_schema_length=None).with_columns(pl.col("n_faltantes").cast(pl.String))
    f = f.filter(pl.col("columna").is_in(["SEXO", "TIPO_INGRESO", "TIPO_PROCEDENCIA", "EDAD_ANIOS", "SERVICIOINGRESO", "DIAGNOSTICO1", "COD_HOSPITAL", "TIPO_ACTIVIDAD", "CIP_ENCRIPTADO", "GRD_ESTADO"]))
    if f.height == 0:
        return
    p = f.group_by("columna").agg(pl.col("prop").max()).sort("prop", descending=True).head(9)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.barh(p["columna"][::-1], (p["prop"].fill_null(0) * 100)[::-1], color=BLUE, height=0.55)
    for i, v in enumerate((p["prop"].fill_null(0) * 100)[::-1]):
        ax.text(v + 0.05, i, f"{v:.1f} %", va="center", fontsize=8.5, color=NAVY)
    ax.set_xlabel("% máximo de faltantes en un año (nunca se imputó)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(F / "f3_faltantes.png", dpi=170); plt.close(fig)


def mortalidad(F):
    m = pl.read_csv(T / "oro_mortalidad_anio.csv").sort("ANIO_EGRESO")
    col = [BLUE if a <= 2022 else GN for a in m["ANIO_EGRESO"]]
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.bar(m["ANIO_EGRESO"].cast(pl.String), m["tasa_mortalidad"] * 100, color=col, width=0.55)
    for i, (v, n) in enumerate(zip(m["tasa_mortalidad"] * 100, m["n_fallecidos"])):
        ax.text(i, v + 0.1, f"{v:.2f} %", ha="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.bar(["2024"], [0], color=RD); ax.text(5, 0.35, "sellado\n(no se abre)", ha="center", fontsize=9, color=RD)
    ax.set_ylabel("mortalidad hospitalaria (%)", fontsize=9); ax.set_ylim(0, 8); limpiar(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="entrenamiento"), Patch(color=GN, label="calibración")], frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(F / "f3_mortalidad.png", dpi=170); plt.close(fig)


def estadia(F, capas):
    d = pl.read_parquet(capas / "oro" / "desenlace_estadia.parquet")
    col = [c for c in d.columns if c.startswith("ESTANCIA")][0]
    sp = pl.read_parquet(capas / "oro" / "matriz_estadia.parquet", columns=["ID_EPISODIO", "SPLIT"]) if "SPLIT" not in d.columns else d
    d = d.join(sp.select("ID_EPISODIO", "SPLIT"), on="ID_EPISODIO") if "SPLIT" not in d.columns else d
    e = d.filter(pl.col("SPLIT") == "entrenamiento")[col].to_numpy()
    tope = int(pl.read_csv(T / "oro_estancia_percentil99.csv")["tope_dias"][0])
    cuentas = np.bincount(np.clip(e, 0, 120))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    dias = np.arange(len(cuentas))
    ax.bar(dias, cuentas, color=[BLUE if x <= tope else OR for x in dias], width=0.85)
    ax.set_yscale("log"); ax.axvline(tope + 0.5, color=RD, ls="--", lw=1.4)
    ax.text(tope + 2, cuentas[1:].max() / 3, f"tope = {tope} días\n(percentil 99)", color=RD, fontsize=9)
    ax.set_xlabel("estadía (días; 120 agrupa 120 o más)", fontsize=9); ax.set_ylabel("egresos (escala log)", fontsize=9); limpiar(ax)
    fig.tight_layout(); fig.savefig(F / "f3_estadia_tope.png", dpi=170); plt.close(fig)


def elixhauser(F):
    p = pl.read_csv(T / "oro_elix_prevalencia_anio.csv")
    cat = pl.read_csv(RAIZ / "config" / "elixhauser_categorias.csv")
    nombre = dict(zip(cat["elix_id"].to_list(), cat["nombre_es"].to_list()))
    top = p.filter(pl.col("anio") == 2023).sort("prevalencia", descending=True).head(10)["elix_id"].to_list()
    fig, ax = plt.subplots(figsize=(8.8, 4))
    for i, e in enumerate(top):
        s = p.filter(pl.col("elix_id") == e).sort("anio")
        ax.plot(s["anio"], s["prevalencia"] * 100, marker="o", lw=1.8, label=str(nombre.get(e, e))[:34])
    ax.set_ylabel("% de egresos con la condición", fontsize=9); limpiar(ax); ax.legend(fontsize=7.3, frameon=False, ncol=2, loc="upper left")
    ax.set_xticks(sorted(p["anio"].unique().to_list())); ax.set_ylim(0, 58)
    fig.tight_layout(); fig.savefig(F / "f3_elixhauser_anio.png", dpi=170); plt.close(fig)


def particion(F):
    fig, ax = plt.subplots(figsize=(9, 2.2)); ax.set_xlim(2018.5, 2024.5); ax.set_ylim(0, 3); ax.axis("off")
    for a0, a1, c, t, tc in [(2018.55, 2022.5, BLUE, "Entrenamiento 2019–2022", "white"), (2022.5, 2023.5, GN, "Calibración 2023", "white"), (2023.5, 2024.45, RD, "Prueba 2024\nSELLADO", "white")]:
        ax.add_patch(FancyBboxPatch((a0, 1.1), a1 - a0 - 0.04, 1.2, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec="none"))
        ax.text((a0 + a1) / 2, 1.7, t, ha="center", va="center", color=tc, fontsize=10, fontweight="bold")
    for a in range(2019, 2025):
        ax.text(a, 0.7, str(a), ha="center", fontsize=9, color=NAVY)
    ax.text(2020.5, 0.2, "el vocabulario y el tope de estadía se calculan solo aquí", ha="center", fontsize=8.5, color=BLUE, style="italic")
    fig.tight_layout(); fig.savefig(F / "f3_particion.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--capas", required=True); ap.add_argument("--salida", default=str(RAIZ / "salidas" / "figuras")); a = ap.parse_args()
    F = Path(a.salida); F.mkdir(parents=True, exist_ok=True)
    for fn in (flujo, cascada, tiempos, filas_por_capa, faltantes, mortalidad, particion, elixhauser):
        fn(F)
    estadia(F, Path(a.capas))
    print("figuras listas en", F)
