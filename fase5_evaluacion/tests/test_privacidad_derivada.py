"""Los conteos de 1 a 9 eventos no deben poder reconstruirse a partir de otras columnas (observado = O/E × esperado)."""
import numpy as np
import pandas as pd

from evaluacion import hospitales, privacidad, subgrupos
from evaluacion.config import cargar_protocolo


def test_hospital_con_pocos_eventos_no_publica_oe_ni_intervalo():
    prot = cargar_protocolo()
    rng = np.random.default_rng(1)
    hosp = np.repeat(np.arange(10), 1000)
    p = np.full(len(hosp), 0.03)
    y = (rng.random(len(hosp)) < 0.03).astype(float)
    y[hosp == 3] = 0
    y[np.where(hosp == 3)[0][:4]] = 1                      # un hospital ficticio con 4 muertes y unas 30 esperadas
    tab, _ = hospitales.tabla(hosp, y, p, "mortalidad", prot)
    fila = tab[tab["cod_hospital"] == 3].iloc[0]
    assert str(fila["observado"]).startswith("<")
    assert pd.isna(fila["oe"]) and pd.isna(fila["ic_inf"]) and pd.isna(fila["ic_sup"])
    assert fila["zona"] != "dentro de lo esperado"          # la zona sigue informándose


def test_subgrupo_pequeno_oculta_oe():
    df = pd.DataFrame({"EDAD_ANIOS": [30.0] * 40, "SEXO": ["HOMBRE"] * 20 + ["MUJER"] * 20, "GRUPO_CLINICO": ["I"] * 40, "COD_HOSPITAL": [1] * 20 + [2] * 20,
                       **{f"ELIX_{i:02d}": [0] * 40 for i in range(1, 4)}})
    y = np.array([1, 1] + [0] * 38, float)
    t = subgrupos.tabla(df, y, np.full(40, 0.1), "mortalidad", 10, agrupar_pequenos=False)   # método anterior
    chico = t[t["observado"].isna()]
    assert len(chico) > 0 and chico["oe"].isna().all()
    n = subgrupos.tabla(df, y, np.full(40, 0.1), "mortalidad", 10)                           # método actual (I-11)
    assert (n["observado"].isna() | (n["observado"] >= 10)).all() and n.loc[n["observado"].isna(), "oe"].isna().all()


def test_el_escaner_detecta_filtraciones_derivadas(tmp_path):
    # Valores inventados: no corresponden a ningún hospital ni decil real.
    mala = tmp_path / "oe_hospital_mortalidad.csv"
    pd.DataFrame({"cod_hospital": [1], "observado": ["<10"], "esperado": [25.0], "oe": [0.2], "ic_inf": [0.07]}).to_csv(mala, index=False)
    assert privacidad.filtraciones_derivadas(mala) == 1
    buena = tmp_path / "buena" / "oe_hospital_mortalidad.csv"
    buena.parent.mkdir()
    pd.DataFrame({"cod_hospital": [1], "observado": ["<10"], "esperado": [25.0], "oe": [np.nan], "ic_inf": [np.nan]}).to_csv(buena, index=False)
    assert privacidad.filtraciones_derivadas(buena) == 0
    dec = tmp_path / "calibracion_deciles_mortalidad.csv"
    pd.DataFrame({"decil": [1, 2], "n": [1000, 1000], "observado": [0.0, 0.003], "razon_oe": [0.0, 0.5]}).to_csv(dec, index=False)
    assert privacidad.filtraciones_derivadas(dec) == 2
