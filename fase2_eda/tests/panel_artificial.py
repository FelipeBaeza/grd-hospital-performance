"""Panel ARTIFICIAL con una verdad de diseño conocida, para ejercitar cada regla de las Tareas 10 y 11 sin leer archivos.

``panel_artificial`` arma un ``Panel`` a mano (sin construir el panel real) con datos sintéticos de 2019 a 2023 (el 80 % en 2019-2022 y el resto en 2023). Para
que no diverja del panel real usa las funciones puras de ``eda.panel``: los niveles y el tope de 60, el recodificado, ``tabla_de_niveles``, ``tabla_muestreo`` y
``cardinalidad_especialidad``. Las pruebas afirman conjuntos y rangos de la verdad de diseño, no valores numéricos frágiles.

Verdad de diseño (nombres reales del registro):

- EDAD_ANIOS tiene señal alta (AUC cercano a 0,72 por construcción del ``y`` logístico). N_DX_SEC se correlaciona con la edad (Spearman cercano a 0,5, no redundante).
- ELIX_01 y ELIX_02 son binarias con OR conocidos; ELIX_02 tiene OR mayor que 1 en 2019-2022 y menor que 1 en 2023 (cambio de signo).
- ELIX_03 es ruido de prevalencia 5 % con AUC cercano a 0,508 (construido dentro de cada clase) e incremento sobre edad y sexo igual a 0. ELIX_04 a ELIX_27, ELIX_29 y
  ELIX_30 son ruido de prevalencia 5 %. ELIX_28 es constante en cero. ELIX_31 tiene 6 expuestos en la ventana de ranking y 2 en 2023.
- N_COMORB_ELIX, PUNTAJE_VAN_WALRAVEN y PUNTAJE_VAN_WALRAVEN_SIN_5 son exactamente la suma y las combinaciones con los pesos reales de ``e.pesos``.
- ES_FIN_SEMANA y DIA_SEMANA están anidadas (V igual a 1) y DIA_SEMANA tiene una deriva fuerte en 2023. SEXO es ruido.
- GRUPO_CLINICO y CIE10_3C están anidadas (CIE10_3C es más fina; ``y`` depende solo del grupo, así que su AUC en muestra es mayor y el de validación cruzada es igual).
  TIPO_PROCEDENCIA y PROCEDENCIA_AGR están anidadas.
- ESPECIALIDAD_MEDICA y SERVICIO_SALUD son función del hospital (proxy alto); COMUNA tiene una asociación cercana a 0,2 con el hospital (proxy moderado).
- PREVISION es independiente del hospital y tiene un nivel de 150 filas sin ningún evento. ETNIA y PUEBLO_ORIGINARIO no se analizan frente al resultado.
- SERVICIOINGRESO tiene alrededor de 55 % de "(sin dato)". NIVEL_CUIDADO_INGRESO tiene señal media y asociación moderada con el hospital.
- IR_29301_MORTALIDAD es una función ruidosa de ``y`` con AUC cercano a 0,93 (fuga).
- ESTANCIA_DIAS y ES_COHORTE_E tienen un gradiente conocido en dos categorías de Elixhauser (ELIX_22 y ELIX_25).
- COD_HOSPITAL tiene cinco hospitales. TIPO_INGRESO tiene un nivel con 40 casos en 2019-2022 y 4 en 2023.
- ES_EXTRANJERO está informada en la ventana de ranking y solo en 4 filas de 2023 (``n_2023 = 4`` con un ``n`` grande en 2019-2022).
- ``y_modo`` "solo_edad" y "solo_elix" generan un ``y`` que depende solo de la edad o solo de las categorías de Elixhauser (referencias de fuga).
- Con ``con_cadena=True`` se agregan CADENA_A, CADENA_B y CADENA_C (numéricas de prueba, registradas solo en este panel) de una normal multivariada con correlaciones
  objetivo de 0,90 (A con B), 0,85 (B con C) y 0,56 (A con C), con un generador aparte (``default_rng(semilla + 101)``) para no alterar el resto del panel.
- Con ``con_historial=True`` existen N_EGRESOS_12M, DIAS_DESDE_EGRESO_PREVIO y HISTORIA_DISPONIBLE, con los espacios de identificadores [[2019, 2020], [2021, 2022, 2023]]:
  HISTORIA_DISPONIBLE vale 0 en 2019 y 2021.

``umbrales_temporales`` copia ``config/umbrales.toml`` y cambia solo el ``valor`` de los umbrales pedidos (por ejemplo ``max_filas_asociacion_por_anio = 150``,
``cardinalidad_max_categorica = 5`` o ``celda_minima = 5``); ``config/umbrales.toml`` queda intacto.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from eda import config, elixhauser, panel as pn, rutas

HOSPITALES = ("101100", "102100", "103100", "104100", "105100")
P_HOSPITAL = (0.27, 0.23, 0.20, 0.16, 0.14)
PREVALENCIA_Y = 0.13
SIN_EVENTOS_PREVISION = 150
REGIMENES_HISTORIAL = [[2019, 2020], [2021, 2022, 2023]]
CORRELACIONES_CADENA = ((1.0, 0.90, 0.56), (0.90, 1.0, 0.85), (0.56, 0.85, 1.0))
CE_CODIGOS = {"CAP_A": ("I21", "I50", "I63"), "CAP_B": ("J18", "J44", "A09"), "CAP_C": ("K35", "N39")}
TIPOS_PROCEDENCIA = (
    "SERVICIO EMERGENCIA (DOMICILIO)", "APS URGENCIA (SAPU, SUR, SUC)", "OTROS HOSPITALES DE LA RED",
    "CENTRO ESPECIALIDADES (CDT, CRS, CONSULTORIO ADOS. ESP)", "CONSULTA PRIVADA", "APS CONSULTORIO (CESFAM)",
)
COMUNAS = tuple(f"COM_{i:02d}" for i in range(12))
COMUNA_BASE = (0, 3, 6, 9, 11)             # comuna "propia" de cada hospital
PESO_COMUNA_PROPIA = 0.17                    # probabilidad de la comuna propia sobre el azar uniforme: ajusta la V con el hospital a ~0,2


def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrar(logit_sin_constante: np.ndarray, prevalencia: float) -> float:
    """Constante que da la prevalencia media pedida (bisección)."""
    bajo, alto = -20.0, 20.0
    for _ in range(80):
        medio = (bajo + alto) / 2.0
        if _sigmoide(logit_sin_constante + medio).mean() > prevalencia:
            alto = medio
        else:
            bajo = medio
    return (bajo + alto) / 2.0


def _categorica_por_clase(rng: np.random.Generator, y: np.ndarray, niveles: list[str], p_evento: list[float], p_no_evento: list[float]) -> np.ndarray:
    """Nivel de una categórica que depende de ``y`` con las probabilidades dadas por clase."""
    salida = np.empty(y.size, dtype=object)
    for clase, probs in ((1, p_evento), (0, p_no_evento)):
        pos = np.flatnonzero(y == clase)
        salida[pos] = np.asarray(niveles, dtype=object)[rng.choice(len(niveles), size=pos.size, p=np.asarray(probs) / np.sum(probs))]
    return salida


def panel_artificial(n: int = 1200, semilla: int = 3, con_historial: bool = False, con_cadena: bool = False, y_modo: str = "mixto") -> pn.Panel:
    """Panel con la verdad de diseño del módulo. Determinista para una semilla; ``n`` filas repartidas de 2019 a 2023 (80 % en 2019-2022)."""
    if y_modo not in ("mixto", "solo_edad", "solo_elix"):
        raise ValueError(f"y_modo debe ser 'mixto', 'solo_edad' o 'solo_elix' (recibió {y_modo!r})")
    rng = np.random.default_rng(semilla)
    anios_rank = [2019, 2020, 2021, 2022]
    n_rank = int(round(0.8 * n))
    tam = [n_rank // 4 + (1 if i < n_rank % 4 else 0) for i in range(4)] + [n - n_rank]
    anio = np.repeat([2019, 2020, 2021, 2022, 2023], tam)
    en_rank = anio <= 2022
    pos_rank, pos_conf = np.flatnonzero(en_rank), np.flatnonzero(~en_rank)

    # --- hospital, edad y variables que entran al ``y``
    idx_h = rng.choice(len(HOSPITALES), size=n, p=P_HOSPITAL)
    hosp = np.asarray(HOSPITALES, dtype=object)[idx_h]
    edad = np.clip(np.round(rng.normal(62.0, 16.0, n)), 18, 98)
    z = (edad - 62.0) / 16.0
    p_elix = np.full(31, 0.05)
    p_elix[0] = p_elix[1] = 0.20
    elix = (rng.random((n, 31)) < p_elix).astype(np.int8)
    elix[:, 27] = 0                                                      # ELIX_28 constante en cero
    elix[:, 30] = 0
    elix[rng.choice(pos_rank, size=min(6, pos_rank.size), replace=False), 30] = 1          # ELIX_31: 6 expuestos en la ventana de ranking
    elix[rng.choice(pos_conf, size=min(2, pos_conf.size), replace=False), 30] = 1
    nivel_cuidado = (rng.random(n) < np.array([0.04, 0.08, 0.14, 0.20, 0.26])[idx_h]).astype(np.int8)
    grupo_idx = rng.choice(3, size=n, p=[0.35, 0.40, 0.25])
    signo_2023 = np.where(anio == 2023, -1.0, 1.0)

    # --- desenlace
    if y_modo == "mixto":
        sin_cte = 1.0 * z + 0.8 * elix[:, 0] + 0.8 * signo_2023 * elix[:, 1] + 2.0 * nivel_cuidado + 0.5 * (grupo_idx == 0)
    elif y_modo == "solo_edad":
        sin_cte = 1.1 * z
    else:
        sin_cte = 1.6 * elix[:, 0] + 1.4 * elix[:, 1]
    y = (rng.random(n) < _sigmoide(sin_cte + _calibrar(sin_cte, PREVALENCIA_Y))).astype(np.int8)

    # --- ELIX_03: ruido construido dentro de cada clase (AUC cercano a 0,508 e incremento 0)
    elix[:, 2] = 0
    for ventana in (pos_rank, pos_conf):
        for clase, tasa in ((1, 0.066), (0, 0.05)):
            pos = ventana[y[ventana] == clase]
            k = int(round(tasa * pos.size))
            elix[rng.choice(pos, size=k, replace=False), 2] = 1
    pesos = elixhauser.cargar().pesos
    sospechosas = elixhauser.cargar().sospechosas
    w = np.array([pesos[i] for i in elixhauser.IDS])
    w5 = np.array([0 if i in sospechosas else pesos[i] for i in elixhauser.IDS])

    # --- tiempo
    p_dia_rank = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.125, 0.125])
    p_dia_conf = np.array([0.40, 0.30, 0.10, 0.08, 0.06, 0.03, 0.03])                 # deriva fuerte en 2023
    dia = np.where(en_rank, rng.choice(7, size=n, p=p_dia_rank), rng.choice(7, size=n, p=p_dia_conf))
    mes = rng.integers(1, 13, size=n)

    # --- diagnóstico principal y procedencia (anidadas)
    capitulo = np.array(["CAP_A", "CAP_B", "CAP_C"], dtype=object)[grupo_idx]
    cie = np.array([CE_CODIGOS[c][int(rng.integers(0, len(CE_CODIGOS[c])))] for c in capitulo], dtype=object)
    tipo_proc = np.asarray(TIPOS_PROCEDENCIA, dtype=object)[rng.choice(6, size=n, p=[0.40, 0.10, 0.12, 0.20, 0.08, 0.10])]

    # --- proxies del hospital
    especialidad = np.array([f"ESP_{h}_{int(rng.integers(0, 3))}" for h in idx_h], dtype=object)
    servicio_salud = np.array([f"SERVICIO {h % 3 + 1}" for h in idx_h], dtype=object)
    propia = rng.random(n) < PESO_COMUNA_PROPIA
    comuna = np.where(propia, np.array([COMUNAS[COMUNA_BASE[h]] for h in idx_h], dtype=object), np.asarray(COMUNAS, dtype=object)[rng.integers(0, len(COMUNAS), size=n)])
    previsiones = ["FONASA A", "FONASA B", "FONASA C", "FONASA D", "ISAPRE"]
    prevision = np.asarray(previsiones, dtype=object)[rng.choice(5, size=n, p=[0.30, 0.30, 0.15, 0.15, 0.10])]
    elegibles = np.flatnonzero(en_rank & (y == 0))
    prevision[rng.choice(elegibles, size=min(SIN_EVENTOS_PREVISION, elegibles.size), replace=False)] = "PRAIS"           # nivel sin ningún evento
    nacionalidad = np.asarray(["CHILENA", "VENEZOLANA", "PERUANA"], dtype=object)[rng.choice(3, size=n, p=[0.92, 0.05, 0.03])]
    etnia = np.asarray(["NINGUNO", "OTRO", "MAPUCHE", "AYMARA"], dtype=object)[rng.choice(4, size=n, p=[0.70, 0.15, 0.10, 0.05])]
    provincia = np.asarray([f"PROV_{i}" for i in range(4)], dtype=object)[rng.integers(0, 4, size=n)]
    sexo = np.asarray(["MUJER", "HOMBRE"], dtype=object)[rng.choice(2, size=n, p=[0.56, 0.44])]
    derivado = (rng.random(n) < 0.10).astype(np.int8)

    # --- servicio de ingreso (55 % sin dato)
    sin_servicio = rng.random(n) < 0.55
    cuidado_txt = np.asarray(["UCI", "UTI"], dtype=object)[rng.integers(0, 2, size=n)]
    otro_txt = np.asarray(["MEDICINA", "CIRUGÍA", "URGENCIA", "TRAUMATOLOGÍA"], dtype=object)[rng.integers(0, 4, size=n)]
    servicio_ingreso = np.where(sin_servicio, pn.SIN_DATO, np.where(nivel_cuidado == 1, cuidado_txt, otro_txt)).astype(object)

    # --- tipo de ingreso: un nivel con 40 casos en 2019-2022 y 4 en 2023
    tipo_ingreso = np.asarray(["URGENCIA", "PROGRAMADA"], dtype=object)[rng.choice(2, size=n, p=[0.6, 0.4])]
    tipo_ingreso[rng.choice(pos_rank, size=min(40, pos_rank.size), replace=False)] = "OBSTETRICA"
    tipo_ingreso[rng.choice(pos_conf, size=min(4, pos_conf.size), replace=False)] = "OBSTETRICA"

    # --- variables de solo fuga (dependen del desenlace)
    mortalidad = _categorica_por_clase(rng, y, ["0", "1", "2", "3"], [0.01, 0.04, 0.25, 0.70], [0.48, 0.30, 0.18, 0.04])
    severidad = _categorica_por_clase(rng, y, ["0", "1", "2", "3"], [0.05, 0.15, 0.35, 0.45], [0.35, 0.30, 0.22, 0.13])
    peso = np.round(0.4 + 0.35 * severidad.astype(int) + rng.gamma(2.0, 0.25, size=n), 4)
    servicio_alta = _categorica_por_clase(rng, y, ["UCI", "MEDICINA", "CIRUGÍA"], [0.60, 0.25, 0.15], [0.10, 0.55, 0.35])
    uso_pabellon = np.where(rng.random(n) < 0.31, "1", pn.SIN_DATO).astype(object)
    n_proc = rng.poisson(1.0 + 0.5 * y).astype(float)
    n_traslados = rng.poisson(0.2 + 0.4 * y).astype(float)
    igual_alta = (rng.random(n) < np.where(y == 1, 0.5, 0.85)).astype(np.int8)

    # --- estadía: gradiente conocido en ELIX_22 y ELIX_25
    multiplicador = 1.0 + 1.0 * elix[:, 21] + 0.7 * elix[:, 24]
    estancia = np.round(multiplicador * (1.0 + rng.gamma(2.0, 1.5, size=n)) * np.where(y == 1, 0.7, 1.0))
    estancia[rng.random(n) < 0.08] = 0.0
    en_cohorte_e = ((y == 0) & (estancia > 0)).astype(np.int8)

    # --- N_DX_SEC correlacionada con la edad (Spearman cercano a 0,5)
    latente = 0.55 * z + np.sqrt(1.0 - 0.55**2) * rng.normal(size=n)
    n_dx = np.clip(np.floor(4.0 + 2.5 * latente), 0, 34).astype(float)

    # --- ES_EXTRANJERO: informada en la ventana de ranking y solo en 4 filas de 2023
    extranjero = pn.es_extranjero(pl.Series(nacionalidad.tolist(), dtype=pl.String)).to_numpy().astype(float)
    informada = en_rank.copy()
    informada[rng.choice(pos_conf, size=min(4, pos_conf.size), replace=False)] = True

    textos: dict[str, np.ndarray] = {
        "SEXO": sexo, "TIPO_INGRESO": tipo_ingreso, "TIPO_PROCEDENCIA": tipo_proc, "GRUPO_CLINICO": capitulo, "CIE10_3C": cie,
        "MES_INGRESO": mes.astype(str).astype(object), "DIA_SEMANA": dia.astype(str).astype(object), "ESPECIALIDAD_MEDICA": especialidad,
        "SERVICIOINGRESO": servicio_ingreso, "PREVISION": prevision, "NACIONALIDAD": nacionalidad, "PROVINCIA": provincia, "COMUNA": comuna, "ETNIA": etnia,
        "SERVICIO_SALUD": servicio_salud, "COD_HOSPITAL": hosp, "IR_29301_SEVERIDAD": severidad, "IR_29301_MORTALIDAD": mortalidad, "SERVICIOALTA": servicio_alta,
        "USOSPABELLON": uso_pabellon,
    }
    columnas: dict[str, pl.Series] = {
        "anio": pl.Series("anio", anio, dtype=pl.Int32), "y": pl.Series("y", y, dtype=pl.Int8),
        "EDAD_ANIOS": pl.Series("EDAD_ANIOS", edad, dtype=pl.Float64), "N_DX_SEC": pl.Series("N_DX_SEC", n_dx, dtype=pl.Float64),
        "N_COMORB_ELIX": pl.Series("N_COMORB_ELIX", elix.sum(axis=1).astype(float), dtype=pl.Float64),
        "PUNTAJE_VAN_WALRAVEN": pl.Series("PUNTAJE_VAN_WALRAVEN", (elix @ w).astype(float), dtype=pl.Float64),
        "PUNTAJE_VAN_WALRAVEN_SIN_5": pl.Series("PUNTAJE_VAN_WALRAVEN_SIN_5", (elix @ w5).astype(float), dtype=pl.Float64),
        "ES_FIN_SEMANA": pl.Series("ES_FIN_SEMANA", (dia >= 5).astype(np.int8), dtype=pl.Int8),
        "NIVEL_CUIDADO_INGRESO": pl.Series("NIVEL_CUIDADO_INGRESO", nivel_cuidado, dtype=pl.Int8),
        "DERIVADO_OTRO_HOSPITAL": pl.Series("DERIVADO_OTRO_HOSPITAL", derivado, dtype=pl.Int8),
        "ES_EXTRANJERO": pl.Series("ES_EXTRANJERO", np.where(informada, extranjero, np.nan), dtype=pl.Float64).fill_nan(None).cast(pl.Int8),
        "IR_29301_PESO": pl.Series("IR_29301_PESO", peso, dtype=pl.Float64), "N_PROCEDIMIENTOS": pl.Series("N_PROCEDIMIENTOS", n_proc, dtype=pl.Float64),
        "N_TRASLADOS_INTERNOS": pl.Series("N_TRASLADOS_INTERNOS", n_traslados, dtype=pl.Float64), "ESTANCIA_DIAS": pl.Series("ESTANCIA_DIAS", estancia, dtype=pl.Float64),
        "ES_COHORTE_E": pl.Series("ES_COHORTE_E", en_cohorte_e, dtype=pl.Int8), "SERVICIO_INGRESO_IGUAL_ALTA": pl.Series("SERVICIO_INGRESO_IGUAL_ALTA", igual_alta, dtype=pl.Int8),
    }
    for i, nombre in enumerate(elixhauser.IDS):
        columnas[nombre] = pl.Series(nombre, elix[:, i], dtype=pl.Int8)
    columnas["ENTRANTE_TIPO_PROCEDENCIA"] = pn.entrante_tipo_procedencia(pl.Series("ENTRANTE_TIPO_PROCEDENCIA", tipo_proc.tolist(), dtype=pl.String))
    columnas["PROCEDENCIA_AGR"] = pn.agrupar_procedencia(pl.Series("PROCEDENCIA_AGR", tipo_proc.tolist(), dtype=pl.String))
    columnas["PUEBLO_ORIGINARIO"] = pn.es_pueblo_originario(pl.Series("PUEBLO_ORIGINARIO", etnia.tolist(), dtype=pl.String))
    for nombre, valores in textos.items():
        columnas[nombre] = pl.Series(nombre, valores.tolist(), dtype=pl.String)

    regimenes: list[list[int]] = []
    if con_historial:
        regimenes = [list(r) for r in REGIMENES_HISTORIAL]
        viable = ~np.isin(anio, [2019, 2021])
        previos = np.where(viable, rng.poisson(0.4 + 0.3 * (edad > 70)), -1)
        dias = np.where(viable & (previos > 0), rng.integers(5, 361, size=n), -1)
        columnas["N_EGRESOS_12M"] = pl.Series("N_EGRESOS_12M", np.where(previos >= 0, previos, np.nan), dtype=pl.Float64).fill_nan(None)
        columnas["DIAS_DESDE_EGRESO_PREVIO"] = pl.Series("DIAS_DESDE_EGRESO_PREVIO", np.where(dias >= 0, dias, np.nan), dtype=pl.Float64).fill_nan(None)
        columnas["HISTORIA_DISPONIBLE"] = pl.Series("HISTORIA_DISPONIBLE", viable.astype(np.int8), dtype=pl.Int8)

    registro = dict(pn.registro_variables(con_historial))
    if con_cadena:
        L = np.linalg.cholesky(np.array(CORRELACIONES_CADENA))
        cadena = np.random.default_rng(semilla + 101).standard_normal((n, 3)) @ L.T
        for j, letra in enumerate("ABC"):
            nombre = f"CADENA_{letra}"
            columnas[nombre] = pl.Series(nombre, cadena[:, j], dtype=pl.Float64)
            registro[nombre] = pn.Variable(nombre=nombre, tipo="num", grupo="prueba", rol_fase1="FEATURE_BASAL", momento="basal", evaluable="si", uso="candidata", fuentes=())

    # --- niveles (con las funciones puras del panel real), recodificado y tablas de apoyo
    niveles: dict[str, list[str]] = {}
    tabla_niveles = []
    ventanas = [(pn.V_RANK, anios_rank), (pn.V_CONF, [2023])]
    for nombre in [v.nombre for v in registro.values() if v.tipo == "cat" and v.nombre in columnas]:
        crudo = columnas[nombre]
        if registro[nombre].es_referencia:
            niv = pn.calcular_niveles_referencia(crudo)
        else:
            niv = pn.calcular_niveles(crudo, columnas["anio"], anios_rank, int(config.cargar_umbrales()["cardinalidad_max_categorica"]))
        columnas[nombre] = pn.recodificar(crudo, niv)
        niveles[nombre] = niv.niveles
        if not registro[nombre].es_referencia:
            tabla_niveles.append(pn.tabla_de_niveles(nombre, columnas[nombre], columnas["anio"], niv, ventanas))
    orden = ["anio", "y"] + [nombre for nombre in registro if nombre in columnas]
    datos = pl.DataFrame([columnas[c] for c in orden])

    muestreo = pn.tabla_muestreo([
        {"anio": a, "ventana": "ranking" if a in anios_rank else "confirmacion", "n_cohorte_d": int((anio == a).sum()), "n_muestra": int((anio == a).sum()),
         "n_eventos_muestra": int(y[anio == a].sum()), "tope": 200_000}
        for a in (2019, 2020, 2021, 2022, 2023)
    ])
    cardinalidad = pn.cardinalidad_especialidad(
        pl.Series("t", especialidad[en_rank].tolist(), dtype=pl.String), pl.Series("h", hosp[en_rank].tolist(), dtype=pl.String), pn.MIN_FILAS_HOSPITAL
    )
    return pn.Panel(
        datos=datos, variables=registro, niveles=niveles, muestreo=muestreo, niveles_tabla=pl.concat(tabla_niveles), regimenes=regimenes, con_historial=con_historial,
        anios_ranking=anios_rank, anio_confirmacion=2023, cardinalidad_especialidad=cardinalidad,
    )


# ----------------------------------------------------------------------------------------- umbrales
def _toml(valor: Any) -> str:
    if isinstance(valor, bool):
        return "true" if valor else "false"
    if isinstance(valor, (int, np.integer)):
        return str(int(valor))
    if isinstance(valor, (float, np.floating)):
        return repr(float(valor))
    if isinstance(valor, str):
        return '"' + valor.replace('"', '\\"') + '"'
    if isinstance(valor, (list, tuple)):
        return "[" + ", ".join(_toml(v) for v in valor) + "]"
    raise TypeError(f"valor no representable en TOML: {valor!r}")


def umbrales_temporales(tmp_path: Path | str, **valores: Any) -> config.Umbrales:
    """Copia ``config/umbrales.toml`` a ``tmp_path`` cambiando solo el ``valor`` de los umbrales pedidos y devuelve ``config.cargar_umbrales(ruta)``.

    Lanza ``KeyError`` si un umbral no existe. El archivo real queda intacto: ``tests/test_t01_protocolo.py`` sigue leyéndolo.
    """
    texto = (rutas.DIR_CONFIG / "umbrales.toml").read_text(encoding="utf8")
    for nombre, valor in valores.items():
        patron = re.compile(rf"(\[umbral\.{re.escape(nombre)}\]\s*\nvalor\s*=\s*)([^\n]+)")
        if not patron.search(texto):
            raise KeyError(f"el umbral {nombre!r} no existe en umbrales.toml")
        texto = patron.sub(lambda m, v=valor: m.group(1) + _toml(v), texto, count=1)
    ruta = Path(tmp_path) / "umbrales_temporales.toml"
    ruta.write_text(texto, encoding="utf8")
    return config.cargar_umbrales(ruta)
