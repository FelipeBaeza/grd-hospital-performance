"""Análisis de apoyo a dos decisiones: (a) sacar recién nacidos y menores de 18; (b) cortar la estadía en el p99.

Solo lee 2019–2023 (el cortafuegos de 2024 se respeta). Escribe CSV en la carpeta de salida.
Uso: python analisis_decisiones.py <carpeta_cache> <carpeta_salida>
"""
from __future__ import annotations
import glob, json, sys, os
import numpy as np
import polars as pl

CACHE, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

def a_fecha(e):
    s = e.str.strip_chars()
    return pl.coalesce(s.str.strptime(pl.Date, "%Y-%m-%d", strict=False, exact=True),
                       s.str.strptime(pl.Date, "%d-%m-%Y", strict=False, exact=True))

DX2 = [f"DIAGNOSTICO{i}" for i in range(2, 36)]
COLS = ["COD_HOSPITAL", "SEXO", "FECHA_NACIMIENTO", "TIPO_PROCEDENCIA", "TIPO_INGRESO", "TIPO_ACTIVIDAD", "FECHA_INGRESO",
        "FECHAALTA", "TIPOALTA", "DIAGNOSTICO1", "IR_29301_COD_GRD", "IR_29301_MORTALIDAD"] + DX2

def cargar(anio: int) -> pl.DataFrame:
    f = glob.glob(f"{CACHE}/bronce_{anio}_*.parquet")[0]
    lf = pl.scan_parquet(f).select(COLS)
    nac, ing, alt = a_fecha(pl.col("FECHA_NACIMIENTO")), a_fecha(pl.col("FECHA_INGRESO")), a_fecha(pl.col("FECHAALTA"))
    md_i = ing.dt.month().cast(pl.Int32) * 100 + ing.dt.day().cast(pl.Int32)
    md_n = nac.dt.month().cast(pl.Int32) * 100 + nac.dt.day().cast(pl.Int32)
    edad = ing.dt.year().cast(pl.Int32) - nac.dt.year().cast(pl.Int32) - (md_i < md_n).cast(pl.Int32)
    g = pl.col("IR_29301_COD_GRD").str.strip_chars()
    fmt6 = g.str.contains(r"^\d{6}$")
    ta = pl.col("TIPOALTA").str.strip_chars().str.to_uppercase()
    rol = (pl.when(ta.str.starts_with("FALLECIDO")).then(pl.lit("E"))
           .when(ta.str.starts_with("DOMICILIO") | ta.str.starts_with("ALTA VOLUNTARIA") | ta.str.starts_with("FUGA")).then(pl.lit("N"))
           .otherwise(pl.lit("C")))
    est = (alt - ing).dt.total_days().cast(pl.Int32)
    n_sec = pl.sum_horizontal([pl.col(c).str.strip_chars().str.len_chars().fill_null(0).gt(0).cast(pl.Int32) for c in DX2])
    cat3 = pl.col("DIAGNOSTICO1").str.strip_chars().str.to_uppercase().str.replace_all(r"[^A-Z0-9]", "").str.slice(0, 3)
    return lf.select(
        pl.lit(anio).alias("anio"),
        pl.col("COD_HOSPITAL").str.strip_chars().alias("hosp"),
        pl.when(pl.col("SEXO").is_in(["MUJER", "HOMBRE"])).then(pl.col("SEXO")).otherwise(pl.lit("SD")).alias("sexo"),
        pl.when((edad >= 0) & (edad <= 115)).then(edad).alias("edad"),
        pl.col("TIPO_PROCEDENCIA").str.strip_chars().fill_null("SD").alias("proc"),
        pl.col("TIPO_INGRESO").str.strip_chars().fill_null("SD").alias("tipo_ing"),
        pl.col("TIPO_ACTIVIDAD").str.strip_chars().str.to_lowercase().alias("act"),
        pl.when(fmt6).then(g.str.slice(0, 2)).alias("mdc"),
        pl.when(fmt6).then(g).alias("grd"),
        pl.col("IR_29301_MORTALIDAD").str.strip_chars().alias("ir_mort"),
        rol.alias("rol"),
        pl.when(est >= 0).then(est).alias("estancia"),
        n_sec.alias("n_sec"),
        cat3.alias("cat3"),
    ).collect()

ANIOS = [2019, 2020, 2021, 2022, 2023]
df = pl.concat([cargar(a) for a in ANIOS])
print("filas", df.height, flush=True)

df = df.with_columns(
    pl.when(pl.col("mdc").is_null() | (pl.col("mdc") == "99")).then(pl.lit("grd_invalido"))
    .when(pl.col("mdc") == "15").then(pl.lit("1_recien_nacido"))
    .when(pl.col("edad").is_null()).then(pl.lit("edad_desconocida"))
    .when(pl.col("edad") < 18).then(pl.lit("2_menor_18"))
    .otherwise(pl.lit("3_adulto")).alias("grupo")
)
hosp_ok = pl.col("act").str.contains("hospital")  # hospitalización (excluye cirugía ambulatoria)
df = df.with_columns(hosp_ok.alias("es_hosp"))

# ------------------------------------------------------------------ 1. Descriptivo por grupo
def resumen(d: pl.DataFrame) -> pl.DataFrame:
    return d.group_by("grupo").agg(
        pl.len().alias("egresos"),
        (pl.col("es_hosp")).sum().alias("egresos_hospitalizacion"),
        pl.col("hosp").n_unique().alias("hospitales"),
        ((pl.col("rol") == "E") & pl.col("es_hosp")).sum().alias("fallecidos"),
        ((pl.col("rol") != "C") & pl.col("es_hosp")).sum().alias("con_desenlace"),
        pl.col("estancia").filter((pl.col("rol") == "N") & pl.col("es_hosp") & (pl.col("estancia") > 0)).mean().alias("estancia_media_sobrev"),
        pl.col("estancia").filter((pl.col("rol") == "N") & pl.col("es_hosp") & (pl.col("estancia") > 0)).quantile(0.99).alias("estancia_p99_sobrev"),
    ).with_columns((pl.col("fallecidos") / pl.col("con_desenlace")).alias("mortalidad")).sort("grupo")

r_all = resumen(df.filter(pl.col("grupo") != "grd_invalido"))
tot_f = r_all["fallecidos"].sum()
r_all = r_all.with_columns((pl.col("fallecidos") / tot_f).alias("prop_fallecidos_total"))
r_all.write_csv(f"{OUT}/a_grupos_edad.csv")
print(r_all, flush=True)

# recién nacidos: hospitalización vs. otros; riesgo GRD de mortalidad 0..3
rn = df.filter(pl.col("grupo") == "1_recien_nacido" & pl.col("es_hosp")) if False else df.filter((pl.col("grupo") == "1_recien_nacido") & pl.col("es_hosp"))
rn_mort = rn.group_by("ir_mort").agg(pl.len().alias("egresos"), (pl.col("rol") == "E").sum().alias("fallecidos"),
                                    (pl.col("rol") != "C").sum().alias("con_desenlace")).with_columns(
    (pl.col("fallecidos") / pl.col("con_desenlace")).alias("mortalidad")).sort("ir_mort")
rn_mort.write_csv(f"{OUT}/a_rn_por_riesgo_grd.csv"); print(rn_mort, flush=True)

# hospitales: cuánto de su actividad hospitalaria sale al excluir menores y RN
hp = df.filter(pl.col("es_hosp") & (pl.col("grupo") != "grd_invalido")).group_by("hosp").agg(
    pl.len().alias("egresos"),
    (pl.col("grupo") == "1_recien_nacido").sum().alias("rn"),
    (pl.col("grupo") == "2_menor_18").sum().alias("menores"),
    (pl.col("grupo") == "3_adulto").sum().alias("adultos"),
).with_columns(((pl.col("rn") + pl.col("menores")) / pl.col("egresos")).alias("prop_salen")).sort("prop_salen", descending=True)
hp.write_csv(f"{OUT}/a_hospitales_prop_salen.csv")
print(hp.head(8), flush=True)
dist = {"hospitales": hp.height,
        "salen_mas_50": int((hp["prop_salen"] > .5).sum()), "salen_mas_25": int((hp["prop_salen"] > .25).sum()),
        "salen_mas_10": int((hp["prop_salen"] > .10).sum()), "mediana": float(hp["prop_salen"].median())}
json.dump(dist, open(f"{OUT}/a_hospitales_resumen.json", "w"))
print(dist, flush=True)

# ------------------------------------------------------------------ 2. Experimento de modelo (proxy sin Elixhauser)
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import spearmanr  # type: ignore

base = df.filter(pl.col("es_hosp") & (pl.col("rol") != "C") & pl.col("edad").is_not_null() & (pl.col("grupo") != "grd_invalido")
                 & (pl.col("mdc") != "14")).with_columns((pl.col("rol") == "E").cast(pl.Int8).alias("y"))
top_cat = base.filter(pl.col("anio") <= 2022)["cat3"].value_counts().sort("count", descending=True).head(250)["cat3"].to_list()
def codificar(d: pl.DataFrame):
    X = np.column_stack([
        d["edad"].to_numpy().astype(float),
        (d["sexo"] == "HOMBRE").to_numpy().astype(float),
        d["n_sec"].to_numpy().astype(float),
        d["tipo_ing"].cast(pl.Categorical).to_physical().to_numpy().astype(float),
        d["proc"].cast(pl.Categorical).to_physical().to_numpy().astype(float),
        d["cat3"].replace_strict({c: i for i, c in enumerate(top_cat)}, default=len(top_cat), return_dtype=pl.Int32).to_numpy().astype(float),
    ])
    return X
# mapas estables para categóricas
for c in ("tipo_ing", "proc"):
    cats = sorted(base[c].unique().to_list()); mp = {k: i for i, k in enumerate(cats)}
    base = base.with_columns(pl.col(c).replace_strict(mp, default=len(cats), return_dtype=pl.Int32).alias(c + "_c"))
def Xmat(d):
    return np.column_stack([d["edad"].to_numpy().astype(float), (d["sexo"] == "HOMBRE").to_numpy().astype(float), d["n_sec"].to_numpy().astype(float),
                            d["tipo_ing_c"].to_numpy().astype(float), d["proc_c"].to_numpy().astype(float),
                            d["cat3"].replace_strict({c: i for i, c in enumerate(top_cat)}, default=len(top_cat), return_dtype=pl.Int32).to_numpy().astype(float)])
cat_mask = [False, False, False, True, True, True]
tr_all = base.filter(pl.col("anio") <= 2022)
te = base.filter((pl.col("anio") == 2023) & (pl.col("grupo") == "3_adulto"))
te_men = base.filter((pl.col("anio") == 2023) & (pl.col("grupo") == "2_menor_18"))
Xte, yte = Xmat(te), te["y"].to_numpy()
brazos = {"Solo adultos (propuesta)": tr_all.filter(pl.col("grupo") == "3_adulto"),
          "Adultos + menores de 18": tr_all.filter(pl.col("grupo").is_in(["3_adulto", "2_menor_18"]))}
# tercer brazo: con recién nacidos (se vuelven a incorporar los MDC 15)
rn_extra = df.filter(pl.col("es_hosp") & (pl.col("rol") != "C") & pl.col("edad").is_not_null() & (pl.col("grupo") == "1_recien_nacido") & (pl.col("anio") <= 2022)).with_columns(
    (pl.col("rol") == "E").cast(pl.Int8).alias("y"))
for c in ("tipo_ing", "proc"):
    cats = sorted(base[c].unique().to_list()); mp = {k: i for i, k in enumerate(cats)}
    rn_extra = rn_extra.with_columns(pl.col(c).replace_strict(mp, default=len(cats), return_dtype=pl.Int32).alias(c + "_c"))
brazos["Adultos + menores + recién nacidos"] = pl.concat([brazos["Adultos + menores de 18"], rn_extra.select(brazos["Adultos + menores de 18"].columns)])
res, oe_hosp = [], {}
for nombre, d in brazos.items():
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, categorical_features=cat_mask, random_state=1)
    m.fit(Xmat(d), d["y"].to_numpy())
    p = m.predict_proba(Xte)[:, 1]
    r = {"modelo": nombre, "n_entrenamiento": d.height, "auc_adultos_2023": roc_auc_score(yte, p), "brier_adultos_2023": brier_score_loss(yte, p),
         "oe_global_adultos_2023": yte.sum() / p.sum()}
    # calibración en adultos de 65+ (zona donde pesa más el riesgo)
    mk = te["edad"].to_numpy() >= 75
    r["oe_75mas"] = yte[mk].sum() / p[mk].sum()
    mk2 = te["edad"].to_numpy() < 45
    r["oe_menos45"] = yte[mk2].sum() / max(p[mk2].sum(), 1e-9)
    # aplicar el modelo de cada brazo a los menores de 2023 (para ver el sesgo si se quisiera usar)
    pm = m.predict_proba(Xmat(te_men))[:, 1]
    r["oe_menores_2023"] = te_men["y"].sum() / pm.sum()
    res.append(r)
    t = te.with_columns(pl.Series("p", p)).group_by("hosp").agg(pl.col("y").sum().alias("obs"), pl.col("p").sum().alias("esp"), pl.len().alias("n"))
    oe_hosp[nombre] = t
    print(r, flush=True)
pl.DataFrame(res).write_csv(f"{OUT}/a_experimento_modelos.csv")
k = list(oe_hosp)
ref = oe_hosp[k[0]].filter(pl.col("n") >= 500).with_columns((pl.col("obs") / pl.col("esp")).alias("oe0"))
comp = []
for n in k[1:]:
    j = ref.join(oe_hosp[n].with_columns((pl.col("obs") / pl.col("esp")).alias("oe1")).select("hosp", "oe1"), on="hosp")
    d = (j["oe1"] - j["oe0"]).abs()
    comp.append({"comparacion": f"{k[0]} vs {n}", "hospitales": j.height, "spearman": float(spearmanr(j["oe0"], j["oe1"])[0]),
                 "dif_abs_media": float(d.mean()), "dif_abs_max": float(d.max()), "hosp_dif_mayor_005": int((d > .05).sum()),
                 "hosp_dif_mayor_010": int((d > .10).sum())})
    j.write_csv(f"{OUT}/a_oe_hospital_{len(comp)}.csv")
pl.DataFrame(comp).write_csv(f"{OUT}/a_experimento_oe_hospital.csv"); print(comp, flush=True)

# ------------------------------------------------------------------ 3. Estadía y percentil 99
sobr = df.filter(pl.col("es_hosp") & (pl.col("rol") == "N") & (pl.col("estancia") > 0) & (pl.col("grupo") == "3_adulto") & (pl.col("mdc") != "14")
                 & pl.col("edad").is_not_null())
sobr_tr = sobr.filter(pl.col("anio") <= 2022)
CORTE = int(sobr_tr["estancia"].quantile(0.99))
tot_dias = sobr["estancia"].sum()
cola = sobr.filter(pl.col("estancia") > CORTE)
tail = {"corte_p99_2019_2022": CORTE, "egresos": sobr.height, "egresos_cola": cola.height, "prop_egresos_cola": cola.height / sobr.height,
        "prop_dias_en_cola": float(cola["estancia"].sum() / tot_dias), "prop_dias_en_cola_recortado": float((cola.height * CORTE) / (tot_dias - cola["estancia"].sum() + cola.height * CORTE)),
        "media_sin_recorte": float(sobr["estancia"].mean()), "media_recortada": float(sobr["estancia"].clip(upper_bound=CORTE).mean()),
        "var_sin_recorte": float(sobr["estancia"].var()), "var_recortada": float(sobr["estancia"].clip(upper_bound=CORTE).var()),
        "maximo": int(sobr["estancia"].max())}
print(tail, flush=True)
# capítulos CIE-10 / MDC de la cola
cm = cola.group_by("mdc").agg(pl.len().alias("n_cola")).join(sobr.group_by("mdc").agg(pl.len().alias("n_mdc")), on="mdc").with_columns(
    (pl.col("n_cola") / pl.col("n_mdc")).alias("prop_cola_en_mdc"), (pl.col("n_cola") / cola.height).alias("prop_de_la_cola")).sort("n_cola", descending=True)
cm.write_csv(f"{OUT}/b_cola_por_mdc.csv"); print(cm.head(8), flush=True)

def Xe(d):
    return Xmat(d.with_columns(pl.col("tipo_ing").replace_strict({k2: i for i, k2 in enumerate(sorted(base['tipo_ing'].unique().to_list()))}, default=99, return_dtype=pl.Int32).alias("tipo_ing_c"),
                               pl.col("proc").replace_strict({k2: i for i, k2 in enumerate(sorted(base['proc'].unique().to_list()))}, default=99, return_dtype=pl.Int32).alias("proc_c")))
Xtr, Xts = Xe(sobr_tr), Xe(sobr.filter(pl.col("anio") == 2023))
yts = sobr.filter(pl.col("anio") == 2023)
hs = yts["hosp"].to_numpy()
oe = {}
for nombre, f_tr, f_obs in [("Sin cortar", lambda s: s, lambda s: s),
                            ("Cortar en el p99 (limitar a %d días)" % CORTE, lambda s: np.minimum(s, CORTE), lambda s: np.minimum(s, CORTE))]:
    m = HistGradientBoostingRegressor(loss="poisson", max_iter=200, learning_rate=0.1, categorical_features=cat_mask, random_state=1)
    m.fit(Xtr, f_tr(sobr_tr["estancia"].to_numpy().astype(float)))
    p = m.predict(Xts)
    obs = f_obs(yts["estancia"].to_numpy().astype(float))
    t = pl.DataFrame({"hosp": hs, "obs": obs, "esp": p}).group_by("hosp").agg(pl.col("obs").sum(), pl.col("esp").sum(), pl.len().alias("n")).filter(pl.col("n") >= 500)
    oe[nombre] = t.with_columns((pl.col("obs") / pl.col("esp")).alias("oe"))
    print(nombre, "O/E global", obs.sum() / p.sum(), flush=True)
a, b = list(oe.values())
j = a.select("hosp", pl.col("oe").alias("oe_sin"), "n").join(b.select("hosp", pl.col("oe").alias("oe_cortado")), on="hosp")
d = (j["oe_cortado"] - j["oe_sin"]).abs()
# estadía: cuánto de la cola es propia de cada hospital
cola23 = yts.group_by("hosp").agg((pl.col("estancia") > CORTE).mean().alias("prop_cola"), pl.len().alias("n")).filter(pl.col("n") >= 500)
j = j.join(cola23.select("hosp", "prop_cola"), on="hosp")
j.write_csv(f"{OUT}/b_oe_estadia_hospital.csv")
res_b = {"hospitales": j.height, "spearman": float(spearmanr(j["oe_sin"], j["oe_cortado"])[0]), "dif_abs_media": float(d.mean()), "dif_abs_max": float(d.max()),
         "hosp_dif_mayor_005": int((d > .05).sum()), "hosp_dif_mayor_010": int((d > .10).sum()),
         "oe_sin_min": float(j["oe_sin"].min()), "oe_sin_max": float(j["oe_sin"].max()), "oe_cort_min": float(j["oe_cortado"].min()), "oe_cort_max": float(j["oe_cortado"].max()),
         "sin_fuera_095_105": int(((j["oe_sin"] < .95) | (j["oe_sin"] > 1.05)).sum()), "cort_fuera_095_105": int(((j["oe_cortado"] < .95) | (j["oe_cortado"] > 1.05)).sum()),
         "prop_cola_hosp_min": float(j["prop_cola"].min()), "prop_cola_hosp_max": float(j["prop_cola"].max()), "prop_cola_hosp_mediana": float(j["prop_cola"].median())}
res_b.update(tail)
json.dump(res_b, open(f"{OUT}/b_resumen.json", "w"), indent=1)
print(res_b, flush=True)
# distribución para el gráfico
h = sobr["estancia"].clip(upper_bound=120).value_counts().sort("estancia"); h.write_csv(f"{OUT}/b_hist_estancia.csv")
print("FIN", flush=True)
