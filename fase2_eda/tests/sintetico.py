"""Generador de datos SINTÉTICOS con la forma de los archivos de egresos GRD (129 columnas).

Las pruebas usan solo estos datos: no contienen información de pacientes reales. El generador
reproduce las trampas conocidas: BOM y CRLF según el año, columna 2 con otro nombre en 2024,
fechas DD-MM-YYYY solo en 2023, decimales con coma, diagnósticos con y sin punto, centinelas,
IDs de 5 a 7 dígitos en 2019-2020 y de 8 a 9 en 2021-2024 (sin traslape entre 2020 y 2021),
HOSPPROCEDENCIA con autorreferencia, etc.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from eda import esquema
from eda.lector import Fuente

ANIOS = (2019, 2020, 2021, 2022, 2023, 2024)

HOSPITALES = ["101100", "102100", "103100", "104100", "105100", "106100", "107100"]
HOSP_PEDIATRICO = "107100"
SERVICIO_SALUD = {h: f"SERVICIO {i % 3 + 1}" for i, h in enumerate(HOSPITALES)}

TIPOALTA_ETIQUETAS = [
    "DOMICILIO",
    "HOSPITALIZACIÓN DOMICILIARIA",
    "DERIVACIÓN OTRO HOSPITAL DEL SERVICIO",
    "ALTA VOLUNTARIA",
    "DERIVACIÓN OTRO HOSPITAL DE LA RED NACIONAL",
    "DERIVACIÓN A OTROS CENTROS (CÁRCEL, HOGAR DE",  # truncada a 44 caracteres, como en los datos
    "FUGA DEL PACIENTE",
    "DERIVACIÓN INST. PRIVADA (COMPRA DE SERVICIOS",  # truncada a 45 caracteres
    "DERIVACIÓN INST. PRIVADA (VOLUNTARIO)",
    "NO IDENTIFICADA",
]
TIPOALTA_P = np.array([0.90, 0.025, 0.02, 0.012, 0.012, 0.0045, 0.0035, 0.0035, 0.0015, 0.0005])
TIPOALTA_P = TIPOALTA_P / TIPOALTA_P.sum()

COD_COMORB = {
    "CHF": ["I50.0", "I50.9"],
    "ARR": ["I48.9", "I49.9"],
    "HTN": ["I10"],
    "DM": ["E11.9", "E10.9"],
    "DMCX": ["E11.2", "E11.5"],
    "RENAL": ["N18.9", "N18.5"],
    "COPD": ["J44.9", "J44.1"],
    "CANCER": ["C50.9", "C34.9"],
    "METS": ["C78.0", "C79.9"],
    "FLUID": ["E87.1", "E87.6"],
    "COAG": ["D68.9"],
    "OBES": ["E66.9"],
    "DEPR": ["F32.9"],
    "ALC": ["F10.2"],
    "ANEM": ["D50.9"],
    "LIVER": ["K70.3", "K74.6"],
    "PVD": ["I73.9"],
    "PARA": ["G81.9"],
    "HYPOTH": ["E03.9"],
}
PESO_COMORB = {  # contribución al logit de mortalidad (solo en el generador)
    "CHF": 0.7, "ARR": 0.4, "HTN": 0.0, "DM": 0.0, "DMCX": 0.2, "RENAL": 0.6, "COPD": 0.3, "CANCER": 0.5,
    "METS": 1.2, "FLUID": 0.5, "COAG": 0.7, "OBES": -0.2, "DEPR": -0.1, "ALC": 0.1, "ANEM": -0.1,
    "LIVER": 0.9, "PVD": 0.2, "PARA": 0.4, "HYPOTH": 0.0,
}
FAMILIAS = list(COD_COMORB)
COD_OTROS = ["J18.9", "A09", "K35.8", "O80", "Z38.0", "S72.0", "N39.0", "R10.4", "K80.2", "L03.1"]
COD_PRINCIPAL = ["J18.9", "A09", "K35.8", "N39.0", "K80.2", "S72.0", "I21.9", "I63.9", "J44.1", "L03.1", "R10.4"]
COD_PRINCIPAL_OBST = ["O80", "O82", "O60.1"]

TIPO_PROCEDENCIA = [
    "SERVICIO EMERGENCIA (DOMICILIO)",
    "CENTRO ESPECIALIDADES (CDT, CRS, CONSULTORIO ADOS. ESP)",
    "OTROS HOSPITALES DE LA RED",
    "APS URGENCIA (SAPU, SUR, SUC)",
    "CONSULTA PRIVADA",
    "APS CONSULTORIO (CESFAM)",
    "DESCONOCIDO",
]
SERVICIOS = ["MEDICINA INTERNA", "CIRUGÍA", "PEDIATRÍA", "OBSTETRICIA", "UCI", "UTI", "URGENCIA", "TRAUMATOLOGÍA"]
PREVISION = ["FONASA INSTITUCIONAL - (MAI) B", "FONASA INSTITUCIONAL - (MAI) A", "ISAPRE", "PARTICULAR", "DESCONOCIDO"]


def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


_NOMBRES_HOSPITAL: dict[str, str] = {}


def _nombre_hospital(codigo: str) -> str:
    """Nombre en mayúsculas del hospital según el catálogo de la raíz del proyecto (como viene en HOSPPROCEDENCIA)."""
    if not _NOMBRES_HOSPITAL:
        from eda import catalogos, rutas

        h = catalogos.cargar_hospitales(rutas.RAIZ_PROYECTO / "Hospitales.csv")
        _NOMBRES_HOSPITAL.update({c: n.upper() for c, n in zip(h["cod_hospital"], h["nombre"])})
    return _NOMBRES_HOSPITAL.get(str(codigo), f"HOSPITAL SINTETICO {codigo}")


def _fmt_id(pac: np.ndarray, anio: int) -> np.ndarray:
    """Identificadores: regimen A (2019-2020) 6 a 7 dígitos; regimen B (2021-2024) 8 a 9 dígitos."""
    if anio <= 2020:
        base = 100000 + pac * 13 + np.where(pac % 10 == 0, 1_000_000, 0)
    else:
        base = 10_000_000 + pac * 17 + np.where(pac % 50 == 0, 100_000_000, 0)
    return base.astype(np.int64).astype(str)


def generar_anio(
    anio: int,
    n: int = 600,
    semilla: int = 7,
    defectos: dict[str, int] | None = None,
    n_pacientes: int = 500,
) -> pl.DataFrame:
    """DataFrame con las 129 columnas (texto) de un año sintético."""
    defectos = defectos or {}
    rng = np.random.default_rng(semilla * 1000 + anio)
    # --- pacientes (atributos fijos entre años) ---
    rp = np.random.default_rng(semilla)
    pac_sexo = rp.choice(["MUJER", "HOMBRE"], size=n_pacientes, p=[0.58, 0.42])
    pac_edad_2019 = np.concatenate(
        [rp.integers(0, 2, size=int(n_pacientes * 0.10)),
         rp.integers(2, 17, size=int(n_pacientes * 0.10)),
         rp.integers(17, 90, size=n_pacientes - int(n_pacientes * 0.10) * 2)]
    )
    rp.shuffle(pac_edad_2019)
    pac_nac = np.array(
        [dt.date(2019, 7, 1) - dt.timedelta(days=int(e) * 365 + int(rp.integers(0, 364))) for e in pac_edad_2019]
    )
    pac_hosp_pref = rp.choice(HOSPITALES[:-1], size=n_pacientes)
    pac_hosp_pref = np.where(pac_edad_2019 < 15, HOSP_PEDIATRICO, pac_hosp_pref)

    pac = rng.integers(0, n_pacientes, size=n)
    sexo = pac_sexo[pac].astype(object)
    nac = pac_nac[pac]
    cod_hosp = np.where(rng.random(n) < 0.9, pac_hosp_pref[pac], rng.choice(HOSPITALES, size=n))
    # volumen desigual: algunos hospitales con pocos casos
    # --- fechas ---
    dia = rng.integers(0, 365, size=n)
    ingreso = np.array([dt.date(anio, 1, 1) + dt.timedelta(days=int(d)) for d in dia])
    # nadie ingresa antes de nacer: si el ingreso sorteado es anterior al nacimiento, se mueve al día del nacimiento
    ingreso = np.array([max(i, b) for i, b in zip(ingreso, nac)])
    edad = np.array([(i - b).days // 365 for i, b in zip(ingreso, nac)])
    # --- actividad y tipo ---
    activ = rng.choice(
        ["HOSPITALIZACIÓN", "CIRUGÍA MAYOR AMBULATORIA (CMA)", "HOSPITALIZACIÓN DIURNA", "HOSPITALIZACIÓN EN URGENCIA"],
        size=n,
        p=[0.84, 0.13, 0.015, 0.015],
    )
    es_rn = edad == 0
    es_parto = (sexo == "MUJER") & (edad >= 18) & (edad <= 40) & (rng.random(n) < 0.35) & ~es_rn
    tipo_ing = np.where(es_parto, "OBSTETRICA", rng.choice(["URGENCIA", "PROGRAMADA"], size=n, p=[0.6, 0.4]))
    # --- comorbilidad y diagnósticos ---
    deriva = 0.2 * (anio - 2019)
    lam = np.clip(1.2 + 0.035 * edad + deriva, 0.2, 7.0)
    lam = np.where(es_parto | es_rn, 0.3, lam)
    k = np.minimum(rng.poisson(lam), 12)
    dx_sec = np.full((n, 34), None, dtype=object)
    logit = -6.3 + 0.045 * edad + 0.05 * (anio == 2020) + np.where(tipo_ing == "URGENCIA", 0.5, 0.0)
    logit = np.where(es_parto | es_rn, -9.0, logit)
    for i in range(n):
        for j in range(k[i]):
            if rng.random() < 0.62:
                fam = FAMILIAS[int(rng.integers(0, len(FAMILIAS)))]
                cod = COD_COMORB[fam][int(rng.integers(0, len(COD_COMORB[fam])))]
                logit[i] += PESO_COMORB[fam]
            else:
                cod = COD_OTROS[int(rng.integers(0, len(COD_OTROS)))]
            r = rng.random()
            if r < 0.11:
                cod = cod.replace(".", "")
            elif r < 0.113:
                cod = cod.lower()
            elif r < 0.116:
                cod = cod + " "
            dx_sec[i, j] = cod
    principal = np.where(
        es_parto, rng.choice(COD_PRINCIPAL_OBST, size=n), rng.choice(COD_PRINCIPAL, size=n)
    ).astype(object)
    principal = np.array([c.replace(".", "") if rng.random() < 0.11 else c for c in principal], dtype=object)
    # --- desenlace ---
    p_muerte = _sigmoide(logit + (activ != "HOSPITALIZACIÓN") * -1.5)
    muerte = rng.random(n) < p_muerte
    tipoalta = rng.choice(TIPOALTA_ETIQUETAS, size=n, p=TIPOALTA_P).astype(object)
    tipoalta[muerte] = "FALLECIDO"
    # --- estadía ---
    base = np.exp(rng.normal(1.0, 0.9, size=n)) + 0.3 * k
    base = np.where(muerte, base * 0.7, base)
    estancia = np.round(base).astype(int)
    cero = (rng.random(n) < 0.15) | np.isin(activ, ["CIRUGÍA MAYOR AMBULATORIA (CMA)", "HOSPITALIZACIÓN DIURNA"])
    estancia = np.where(cero, 0, estancia)
    estancia = np.where(es_parto, np.clip(estancia, 1, 4), estancia)
    alta = np.array([i + dt.timedelta(days=int(e)) for i, e in zip(ingreso, estancia)])
    # --- GRD ---
    mdc = rng.choice([1, 4, 5, 6, 8, 11, 18, 21], size=n)
    mdc = np.where(es_parto, 14, np.where(es_rn, 15, mdc))
    ambulatorio = np.isin(activ, ["CIRUGÍA MAYOR AMBULATORIA (CMA)", "HOSPITALIZACIÓN DIURNA"])
    tipo_grd = np.where(ambulatorio, rng.choice([2, 3, 5], size=n), rng.choice([1, 4], size=n))
    tipo_grd = np.where(es_parto, 6, np.where(es_rn, 8, tipo_grd))
    sev_p = _sigmoide(logit + 2.2)  # severidad del GRD calculada al alta: se asocia al desenlace (fuga)
    sev = np.where(rng.random(n) < 0.5, np.digitize(sev_p, [0.25, 0.5, 0.75]), rng.integers(0, 4, size=n))
    sev = np.where(muerte, np.maximum(sev, 2), sev)
    grupo = rng.integers(10, 99, size=n)
    grd = np.array([f"{m:02d}{t}{g:02d}{max(1, s)}" for m, t, g, s in zip(mdc, tipo_grd, grupo, sev)], dtype=object)
    no_agrupable = rng.random(n) < 0.004
    grd = np.where(no_agrupable, "999991", grd).astype(object)
    desconocido_grd = rng.random(n) < 0.0008
    grd = np.where(desconocido_grd, "DESCONOCIDO", grd).astype(object)
    peso = np.round(0.35 + 0.35 * sev + rng.gamma(2.0, 0.25, size=n), 4)
    peso_txt = np.array([f"{p:.4f}".replace(".", ",") for p in peso], dtype=object)
    mort_grd = np.where(muerte, rng.integers(2, 4, size=n), rng.integers(0, 3, size=n)).astype(str).astype(object)
    sev_txt = sev.astype(str).astype(object)
    # --- servicios ---
    serv_ing = np.where(muerte & (rng.random(n) < 0.35), "UCI", rng.choice(SERVICIOS, size=n)).astype(object)
    serv_alta = np.where(muerte & (rng.random(n) < 0.6), "UCI", serv_ing).astype(object)
    # --- procedencia ---
    tp = rng.choice(TIPO_PROCEDENCIA, size=n, p=[0.40, 0.22, 0.12, 0.10, 0.06, 0.08, 0.02]).astype(object)
    hosp_proc = np.full(n, None, dtype=object)
    entrante = tp == "OTROS HOSPITALES DE LA RED"
    for i in np.where(entrante)[0]:
        # en los datos reales HOSPPROCEDENCIA trae el NOMBRE del hospital, en mayúsculas
        hosp_proc[i] = _nombre_hospital(cod_hosp[i] if rng.random() < 0.35 else rng.choice(HOSPITALES))
    # autorreferencia: el servicio de urgencia del propio hospital figura como hospital de procedencia (generador aparte para no alterar el flujo principal)
    rng_auto = np.random.default_rng(semilla * 7919 + anio)
    for i in np.where((tp == "SERVICIO EMERGENCIA (DOMICILIO)") & (rng_auto.random(n) < 0.10))[0]:
        hosp_proc[i] = _nombre_hospital(cod_hosp[i])
    # --- IDs ---
    ids = _fmt_id(pac, anio).astype(object)
    sentinel = rng.random(n) < 0.005
    ids = np.where(sentinel, "DESCONOCIDO", ids).astype(object)
    vacio = rng.random(n) < 0.005
    ids = np.where(vacio, None, ids).astype(object)
    # --- ETNIA (recodificación por año) ---
    etnia = rng.choice(["NINGUNO", "OTRO", "MAPUCHE", "AYMARA"], size=n, p=[0.5, 0.4, 0.09, 0.01]).astype(object)
    if anio <= 2021:
        etnia = np.where(etnia == "OTRO", "OTRO ", etnia).astype(object)
    # --- columnas proxy del hospital ---
    esp_idx = {h: i for i, h in enumerate(HOSPITALES)}
    especialidad = np.array(
        [f"ESP_{esp_idx[h]}_{int(rng.integers(0, 3))}" for h in cod_hosp], dtype=object
    )
    provincia = np.array([f"PROV_{esp_idx[h]}" for h in cod_hosp], dtype=object)
    comuna = np.array([f"COM_{esp_idx[h]}_{int(rng.integers(0, 4))}" for h in cod_hosp], dtype=object)

    datos: dict[str, np.ndarray] = {c: np.full(n, None, dtype=object) for c in esquema.COLUMNAS}
    datos["COD_HOSPITAL"] = cod_hosp.astype(object)
    datos["CIP_ENCRIPTADO"] = ids
    datos["SEXO"] = sexo
    datos["FECHA_NACIMIENTO"] = np.array([b.isoformat() for b in nac], dtype=object)
    datos["ETNIA"] = etnia
    datos["PROVINCIA"] = provincia
    datos["COMUNA"] = comuna
    datos["NACIONALIDAD"] = rng.choice(["CHILENA", "VENEZOLANA", "PERUANA"], size=n, p=[0.93, 0.05, 0.02]).astype(object)
    datos["PREVISION"] = rng.choice(PREVISION, size=n, p=[0.5, 0.35, 0.05, 0.05, 0.05]).astype(object)
    datos["SERVICIO_SALUD"] = np.array([SERVICIO_SALUD[h] for h in cod_hosp], dtype=object)
    datos["TIPO_PROCEDENCIA"] = tp
    datos["TIPO_INGRESO"] = tipo_ing.astype(object)
    datos["ESPECIALIDAD_MEDICA"] = especialidad
    datos["TIPO_ACTIVIDAD"] = activ.astype(object)
    datos["FECHA_INGRESO"] = np.array([d.isoformat() for d in ingreso], dtype=object)
    datos["SERVICIOINGRESO"] = serv_ing
    datos["FECHAALTA"] = np.array([d.isoformat() for d in alta], dtype=object)
    datos["SERVICIOALTA"] = serv_alta
    datos["TIPOALTA"] = tipoalta
    datos["DIAGNOSTICO1"] = principal
    for j in range(34):
        datos[f"DIAGNOSTICO{j + 2}"] = dx_sec[:, j]
    # procedimientos con cero inicial
    tiene_proc = rng.random(n) < 0.3
    datos["PROCEDIMIENTO1"] = np.where(tiene_proc, rng.choice(["00.17", "09.1", "47.01", "81.51"], size=n), None).astype(object)
    datos["PROCEDIMIENTO2"] = np.where(tiene_proc & (rng.random(n) < 0.4), "39.95", None).astype(object)
    datos["FECHAINTERV1"] = np.where(tiene_proc, datos["FECHA_INGRESO"], None).astype(object)
    datos["USOSPABELLON"] = np.where(tiene_proc, "1", None).astype(object)
    datos["IR_29301_COD_GRD"] = grd
    datos["IR_29301_PESO"] = peso_txt
    datos["IR_29301_SEVERIDAD"] = sev_txt
    datos["IR_29301_MORTALIDAD"] = mort_grd
    datos["HOSPPROCEDENCIA"] = hosp_proc
    # recién nacidos
    datos["CONDICIONDEALTANEONATO1"] = np.where(es_parto, "VIVO", None).astype(object)
    datos["PESORN1"] = np.where(es_parto, rng.integers(2500, 4200, size=n).astype(str), None).astype(object)
    datos["RN1ESTADO"] = np.where(es_parto, "10", None).astype(object)
    datos["RN2ESTADO"] = np.where(rng.random(n) < 0.95, "0", None).astype(object)
    # traslados internos
    tr = rng.random(n) < 0.12
    datos["FECHATRASLADO1"] = np.where(tr, datos["FECHA_INGRESO"], None).astype(object)
    datos["SERVICIOTRASLADO1"] = np.where(tr, "UCI", None).astype(object)

    # --- defectos plantados ---
    def _elegir(clave: str) -> np.ndarray:
        m = int(defectos.get(clave, 0))
        return rng.choice(n, size=m, replace=False) if m else np.array([], dtype=int)

    for i in _elegir("alta_antes_ingreso"):
        datos["FECHAALTA"][i] = (ingreso[i] - dt.timedelta(days=5)).isoformat()
    for i in _elegir("estancia_larga"):
        datos["FECHAALTA"][i] = (ingreso[i] + dt.timedelta(days=400)).isoformat()
    for i in _elegir("edad_negativa"):
        datos["FECHA_NACIMIENTO"][i] = (ingreso[i] + dt.timedelta(days=30)).isoformat()
    for i in _elegir("edad_mayor_110"):
        datos["FECHA_NACIMIENTO"][i] = (ingreso[i] - dt.timedelta(days=365 * 115)).isoformat()
    for i in _elegir("rut_en_fecha"):
        datos["FECHAPROCEDIMIENTO1"][i] = "12345678-9"
    for i in _elegir("alta_vacia"):
        datos["FECHAALTA"][i] = None
    for i in _elegir("cambio_sexo"):
        datos["SEXO"][i] = "HOMBRE" if datos["SEXO"][i] == "MUJER" else "MUJER"
    for i in _elegir("hospital_fuera_catalogo"):
        datos["COD_HOSPITAL"][i] = "999999"

    df = pl.DataFrame({c: pl.Series(c, datos[c].tolist(), dtype=pl.String) for c in esquema.COLUMNAS})
    dup = int(defectos.get("duplicados", 0))
    if dup:
        df = pl.concat([df, df.head(dup)])
    return df


def _a_dmy(valor: str | None) -> str | None:
    if valor is None or len(valor) != 10 or valor[4] != "-":
        return valor
    return f"{valor[8:10]}-{valor[5:7]}-{valor[0:4]}"


COLS_FECHA_DMY = ("FECHA_INGRESO", "FECHAALTA", "FECHAINTERV1", *esquema.FECHAS_TRASLADO)


def escribir_archivo(
    df: pl.DataFrame,
    ruta: Path,
    *,
    bom: bool = False,
    crlf: bool = True,
    nombre_col2: str = "CIP_ENCRIPTADO",
    dmy: bool = False,
    lineas_extra: list[str] | None = None,
) -> Path:
    """Escribe un DataFrame de 129 columnas como archivo delimitado por ``|``."""
    if dmy:
        df = df.with_columns([pl.col(c).map_elements(_a_dmy, return_dtype=pl.String) for c in COLS_FECHA_DMY])
    encabezado = list(esquema.COLUMNAS)
    encabezado[1] = nombre_col2
    fin = "\r\n" if crlf else "\n"
    lineas = ["|".join(encabezado)]
    for fila in df.iter_rows():
        lineas.append("|".join("" if v is None else v for v in fila))
    if lineas_extra:
        lineas.extend(lineas_extra)
    contenido = fin.join(lineas) + fin
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "wb") as f:
        if bom:
            f.write(b"\xef\xbb\xbf")
        f.write(contenido.encode("utf-8"))
    return ruta


def crear_fuente_sintetica(
    raiz: Path,
    anios: tuple[int, ...] = ANIOS,
    n: int = 600,
    semilla: int = 7,
    defectos_por_anio: dict[int, dict[str, int]] | None = None,
    n_pacientes: int = 500,
) -> Fuente:
    """Crea archivos anuales sintéticos con las características de los extractos reales."""
    defectos_por_anio = defectos_por_anio or {}
    archivos: dict[int, Path] = {}
    for anio in anios:
        df = generar_anio(anio, n=n, semilla=semilla, defectos=defectos_por_anio.get(anio), n_pacientes=n_pacientes)
        ruta = raiz / f"GRD_PUBLICO_{anio}.txt"
        escribir_archivo(
            df,
            ruta,
            bom=anio in (2021, 2022, 2023),
            crlf=True,
            nombre_col2="ID_BENEFICIARIO" if anio == 2024 else "CIP_ENCRIPTADO",
            dmy=anio == 2023,
        )
        archivos[anio] = ruta
    return Fuente(etiqueta="extracto", raiz=raiz, archivos=archivos)
