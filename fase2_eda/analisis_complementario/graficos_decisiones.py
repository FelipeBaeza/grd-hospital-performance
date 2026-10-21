import csv, numpy as np, matplotlib, sys, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "LiberationSans" in f:
        try: fm.fontManager.addfont(f)
        except Exception: pass
AN, OUT = sys.argv[1], sys.argv[2]
S1, S2, S3 = "#2e5597", "#eb6834", "#1baf7a"
INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": "#1f3864", "axes.titlelocation": "left", "legend.frameon": False, "savefig.dpi": 200})
def rd(f): return list(csv.DictReader(open(f"{AN}/{f}")))
miles = lambda x: f"{x:,.0f}".replace(",", ".")
# ---- g17
g = {r["grupo"]: r for r in rd("a_grupos_edad.csv")}
hp = rd("a_hospitales_prop_salen.csv")
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1, 1.35]})
claves = ["1_recien_nacido", "2_menor_18", "3_adulto"]
_n = ["Recién nacidos\n(MDC 15)", "Menores de 18\n(sin recién nacidos)", "Adultos\n(18 o más)"]
nombres = [f"{a}\n{miles(float(g[k]['egresos']))} egresos" for a, k in zip(_n, claves)]
m = [float(g[k]["mortalidad"]) * 100 for k in claves]
b = ax[0].bar(nombres, m, color=[S2, S2, S1], width=.6)
for rect, k, v in zip(b, claves, m):
    ax[0].text(rect.get_x() + rect.get_width() / 2, v + .08, f"{v:.2f}".replace(".", ",") + " %", ha="center", fontsize=9, color="#0b0b0b", fontweight="bold")
ax[0].set_ylabel("Mortalidad (%)"); ax[0].set_ylim(0, 5.4); ax[0].set_title("Mortalidad por grupo (2019–2023)"); ax[0].grid(axis="x", visible=False)
ax[0].tick_params(axis="x", labelsize=8.3)
p = [float(r["prop_salen"]) * 100 for r in hp]
cols = [S2 if v > 50 else S1 for v in p]
ax[1].bar(range(len(p)), p, color=cols, width=.8)
ax[1].axhline(10, color=MUTED, lw=.8, ls="--"); ax[1].text(len(p) - 1, 11.5, "10 %", ha="right", fontsize=8, color=MUTED)
ax[1].set_xlabel("Los 68 hospitales, de mayor a menor"); ax[1].set_ylabel("% de sus egresos que salen"); ax[1].set_xticks([])
ax[1].set_title("Cuánto de cada hospital sale al quedarse solo con adultos"); ax[1].grid(axis="x", visible=False)
ax[1].text(3.2, 88, "3 hospitales pediátricos", fontsize=8.5, color=S2, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/g17_menores_rn.png", bbox_inches="tight"); plt.close(fig)
# ---- g18
h = rd("b_hist_estancia.csv"); x = np.array([int(r["estancia"]) for r in h]); n = np.array([int(r["count"]) for r in h])
import json
rs = json.load(open(f"{AN}/b_resumen.json")); C = int(rs["corte_p99_2019_2022"])
j = rd("b_oe_estadia_hospital.csv")
a = np.array([float(r["oe_sin"]) for r in j]); c = np.array([float(r["oe_cortado"]) for r in j])
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))
ax[0].bar(x[x <= C], n[x <= C], color=S1, width=1); ax[0].bar(x[x > C], n[x > C], color=S2, width=1)
ax[0].set_yscale("log"); ax[0].axvline(C + .5, color="#1f3864", lw=1); ax[0].text(C + 3, n.max() * .35, f"corte: {C} días", fontsize=8.5, color="#1f3864", fontweight="bold")
ax[0].set_xlabel("Días de estadía (el último valor suma 120 o más)"); ax[0].set_ylabel("Egresos (escala logarítmica)")
ax[0].set_title(f"La cola: {rs['prop_egresos_cola']*100:.1f}".replace(".", ",") + f" % de los egresos, {rs['prop_dias_en_cola']*100:.1f}".replace(".", ",") + " % de los días", fontsize=10)
ax[1].plot([.45, 1.4], [.45, 1.4], color=MUTED, lw=.8, ls="--"); ax[1].scatter(a, c, s=22, color=S1, alpha=.85, edgecolor="white", lw=.4)
ax[1].axvspan(.95, 1.05, color=GRID, alpha=.6, lw=0); ax[1].axhspan(.95, 1.05, color=GRID, alpha=.6, lw=0)
ax[1].set_xlabel("O/E de estadía sin cortar"); ax[1].set_ylabel("O/E de estadía cortando en el p99"); ax[1].set_xlim(.45, 1.4); ax[1].set_ylim(.45, 1.4)
ax[1].set_title("Cada punto es un hospital (65)")
fig.tight_layout(); fig.savefig(f"{OUT}/g18_p99.png", bbox_inches="tight"); plt.close(fig)
