import polars as pl
import numpy as np
import scipy.stats as stats
from pathlib import Path

# Load data and hospital stats
df_23 = pl.read_parquet("/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_cache.parquet")
# We have h_m3 from check_hosp_detail
# Let's write a function that, given base O, E, and transfers N_transf, tests what lambda of transfers changes the status

# For Curanilahue (128109):
# Base: O = 136, E = 119.5, N_transf = 511
# If we attribute transfers with mortality rate = lambda * 0.043467:
# Additional deaths O_add = N_transf * lambda * 0.043467
# Additional expected E_add = N_transf * 0.043467 (expected at network average)
# O_new = O + O_add
# E_new = E + E_add
# OE_new = O_new / E_new
# v_w = 1 / O_new
# B_j = tau^2 / (tau^2 + v_w)
# log_th = B_j * ln(OE_new) + (1 - B_j) * mu
# se = sqrt(tau^2 * v_w / (tau^2 + v_w))
# Z = (log_th - ln(1.10)) / se
# P = 1 - norm.cdf(-Z)

tau2 = 0.0472
tau = np.sqrt(tau2)
mu = 0.0261

def test_lambda(O_base, E_base, N_transf, lam):
    O_add = N_transf * lam * 0.043467
    E_add = N_transf * 0.043467
    O_tot = O_base + O_add
    E_tot = E_base + E_add
    oe = O_tot / E_tot
    v = 1.0 / O_tot
    b = tau2 / (tau2 + v)
    lth = b * np.log(oe) + (1.0 - b) * mu
    se = np.sqrt(tau2 * v / (tau2 + v))
    z110 = (lth - np.log(1.10)) / se
    p110 = 1.0 - stats.norm.cdf(-z110)
    return oe, p110

print("1. CURANILAHUE (128109): Búsqueda de lambda_entrada (para alcanzar P >= 0.95)")
for lam in np.arange(1.0, 3.5, 0.1):
    oe, p = test_lambda(136, 119.5, 511, lam)
    if p >= 0.95:
        print(f"  --> Curanilahue entra en alerta en lambda_entrada = {lam:.2f}x (Tasa = {lam*4.35:.1f}%, OE={oe:.3f}, P={p:.3f})")
        break

print("\n2. HOSPITALES EN ALERTA: Búsqueda de lambda_salida (para salir de alerta P < 0.95)")
# Let's test for El Pino, Claudio Vicuña, Van Buren, Tisné, Pereira, La Florida
# Alert hosp data (O, E, N_transf):
alerts = [
    ('113180', 'El Pino', 438, 247.1, 1370),
    ('106103', 'Claudio Vicuña', 320, 195.8, 299),
    ('106100', 'Van Buren', 602, 419.3, 1259),
    ('112101', 'Luis Tisné', 502, 381.3, 908),
    ('106102', 'Eduardo Pereira', 148, 100.1, 48), # transfers from silver
    ('114105', 'La Florida', 596, 457.3, 215),
    ('112103', 'Instituto del Tórax', 157, 70.5, 112)
]

for cod, nom, o, e, nt in alerts:
    # Test if decreasing lambda (e.g. low transfer mortality) reduces P < 0.95
    found = False
    for lam in np.arange(1.0, 0.0, -0.05):
        oe, p = test_lambda(o, e, nt, lam)
        if p < 0.95:
            print(f"  --> {nom} ({cod}): sale de alerta en lambda_salida = {lam:.2f}x (OE={oe:.3f}, P={p:.3f})")
            found = True
            break
    if not found:
        # Check at lambda = 0
        oe, p = test_lambda(o, e, nt, 0.0)
        print(f"  --> {nom} ({cod}): NO sale de alerta ni siquiera con lambda=0.0x (en lam=0: OE={oe:.3f}, P={p:.3f})")

