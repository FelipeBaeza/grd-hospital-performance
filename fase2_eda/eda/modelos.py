"""Modelos exploratorios de la Tarea 11: diseños, ajustes logísticos, referencias, incremento de AUC, modelo L1, importancia por permutación y AUC univariado.

Todo se estima con la ventana de ranking (2019 a 2022) y se evalúa en 2023; 2024 nunca interviene. Ningún modelo multivariado (el L1, la permutación y las
referencias) usa COD_HOSPITAL, IR_29301_MORTALIDAD ni variables de rol excluyente: solo variables con ``uso == "candidata"`` que analizan el resultado y cuyo rol de la
Fase 1 no es excluyente. El incremento de AUC (edad, edad al cuadrado y sexo más una variable) sí se calcula para toda candidata que analiza el resultado, porque es la
única forma de evaluar la regla BAJA_SENAL de las que la Fase 1 excluye.

El relleno de nulos con la mediana del entrenamiento (más un indicador de nulos) existe SOLO aquí, dentro de los modelos exploratorios: la regla R5 de la Fase 1 lo admite
para las líneas base y el panel conserva los nulos. Los diseños que entregan un AUC usan una penalización ridge (``RIDGE_AUC``) que estabiliza los niveles raros sin
eventos (ceros verdaderos); los que entregan un OR con error de Wald usan ridge cero (``RIDGE_WALD``). La separación nunca anula un AUC ni un incremento.

Las funciones de AUC univariado (``ajustar_univariado``, ``auc_univariado`` y ``auc_cv_univariado``) fijan con la ventana de ranking la dirección, los niveles y las tasas
por nivel y las aplican a cualquier conjunto de años; las comparten el ranking, la estabilidad y la fuga para que usen el mismo puntaje y la misma dirección.

Fuentes: Hanley y McNeil (intervalo del AUC), Tibshirani (penalización L1) y Newton para la regresión logística (``estadistica.regresion_logistica``).
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from . import asociacion, decision, elixhauser
from . import estadistica as est

RIDGE_AUC = 1.0
"""Ridge (lambda) de todo diseño que entrega un AUC: modelo base, referencias e incrementos. Equivale a ``C = 1`` de scikit-learn y estabiliza los niveles raros de las
categóricas, que casi siempre incluyen ceros verdaderos sin ningún evento."""
RIDGE_WALD = 0.0
"""Ridge de todo diseño que entrega un OR con error de Wald (OR ajustado y OR de estadía larga de Elixhauser): cero, para no sesgar el coeficiente ni su error estándar."""
TOPE_FILAS_MODELOS: int | None = None
"""Tope de filas de entrenamiento de los modelos. ``None`` (por defecto): los modelos se entrenan con TODAS las filas de la ventana de ranking. Con un entero, si la
ventana lo supera se usa una muestra reproducible (``indices_entrenamiento``); queda como opción para equipos con poca memoria."""
TOPE_FILAS_L1: int | None = 400_000
"""Tope de filas con que se AJUSTA el modelo L1 (el único cálculo de la Fase 2 que no usa todas las filas): con toda la cohorte (más de 2 millones de egresos) el solver
liblinear con validación cruzada tarda muchas horas. La muestra es reproducible (``default_rng(semilla + 13)``); el AUC y la importancia del modelo se evalúan con TODAS las
filas. ``None`` ajusta con todas."""
CS_L1 = np.logspace(-3, 1, 7)
"""Valores de C (inverso de la penalización) que explora el modelo L1."""
PLIEGUES_CV = 3
"""Pliegues de la validación cruzada del L1 (el ``cv=3`` del encargo)."""
REPETICIONES_PERMUTACION = 10
"""Repeticiones de la importancia por permutación de cada variable."""

_MIN_EVENTOS = 3
"""Eventos mínimos por pliegue del L1 y mínimo de eventos (y de no eventos) para ajustar un incremento. No es un umbral del protocolo."""
BASE_EDAD_SEXO: tuple[str, ...] = ("EDAD_ANIOS", "EDAD_ANIOS^2", "SEXO")
"""Variables del modelo base: edad, su cuadrado (estandarizados por separado) y sexo. ``X^2`` es el cuadrado de la numérica ``X``."""
NIVEL_BASE_FIJO = {"SEXO": "MUJER"}
"""Excepción explícita a ``omitir_nivel_base``: el nivel base de SEXO es MUJER aunque HOMBRE sea el más frecuente, para que el modelo base, el OR de Elixhauser y sus
oráculos coincidan."""
VARIABLES_BASE = ("EDAD_ANIOS", "SEXO")
"""Variables que pertenecen al modelo base: su incremento no aplica."""
ESTADO_NO_APLICA = "no aplica: variable del modelo base"
POCOS_EVENTOS_L1 = "no evaluable: pocos eventos"


# =================================================================================================================== tipos
@dataclass(frozen=True)
class EspecDiseno:
    """Lo que un diseño aprende con el entrenamiento: estandarización, relleno de nulos, niveles y columnas."""

    variables: tuple[str, ...]              # variables del diseño, en el orden del registro (una numérica al cuadrado, "X^2", va justo después de X)
    medias: dict[str, float]                # media de cada numérica en el entrenamiento
    desvios: dict[str, float]               # desviación estándar (ddof=0); una numérica con desviación cero se omite
    medianas: dict[str, float]              # relleno de nulos de numéricas y binarias (0 en binarias), solo del entrenamiento
    tiene_nulos: dict[str, bool]            # si el entrenamiento tuvo nulos (entonces existe la columna variable__nulo)
    niveles: dict[str, tuple[str, ...]]     # niveles de cada categórica, en el orden de panel.niveles
    nivel_base: dict[str, str]              # nivel omitido de cada categórica (vacío si omitir_nivel_base es falso)
    columnas: tuple[str, ...]               # nombre de cada columna de X, en orden
    filas_entrenamiento: np.ndarray | None  # posiciones, dentro de la ventana de ranking, de la muestra de entrenamiento (None si no se muestreó)


@dataclass(frozen=True)
class ResultadoL1:
    """Modelo L1 ajustado: coeficientes, C elegido, qué variables sobreviven y si el ajuste convergió."""

    coef: np.ndarray                        # un coeficiente por columna del modelo reajustado (NaN si no se ajustó)
    intercepto: float
    c_elegido: float                        # C elegido por validación cruzada
    columnas: tuple[str, ...]
    grupos: dict[str, list[int]]            # variable -> índices de columna
    sobrevive_por_variable: dict[str, bool]
    estado: str                             # "ok" o "no evaluable: pocos eventos"
    convergio: bool                         # falso si algún ajuste del liblinear agotó las iteraciones


# ============================================================================================================ muestra y diseño
def indices_entrenamiento(n: int, semilla: int) -> np.ndarray:
    """Posiciones de la muestra de entrenamiento dentro de una ventana de ``n`` filas.

    Es la identidad (``np.arange(n)``) si ``n <= TOPE_FILAS_MODELOS``; si no, una muestra sin reemplazo de ese tamaño con ``default_rng(semilla + 7)``, ordenada. Pura:
    el tope se lee del módulo en cada llamada.
    """
    if TOPE_FILAS_MODELOS is None or n <= TOPE_FILAS_MODELOS:
        return np.arange(n)
    return np.sort(np.random.default_rng(semilla + 7).choice(n, size=TOPE_FILAS_MODELOS, replace=False))


def _mascara(panel, anios: Sequence[int]) -> np.ndarray:
    return panel.datos["anio"].is_in([int(a) for a in anios]).to_numpy()


def _base_de(nombre: str) -> tuple[str, bool]:
    return (nombre[:-2], True) if nombre.endswith("^2") else (nombre, False)


def _ordenar(panel, variables: Sequence[str]) -> list[str]:
    """Variables sin repetir, en el orden del registro (el cuadrado de una numérica va justo después de ella)."""
    pedidas = list(dict.fromkeys(variables))
    orden = {n: i for i, n in enumerate(panel.variables)}
    for nombre in pedidas:
        base, cuadrado = _base_de(nombre)
        if base not in orden:
            raise KeyError(f"modelos: la variable {base!r} no está en el registro del panel")
        if base not in panel.datos.columns:
            raise ValueError(f"modelos: la variable {base!r} no tiene columna en el panel ({panel.variables[base].motivo or 'no construida'})")
        if cuadrado and panel.variables[base].tipo != "num":
            raise ValueError(f"modelos: solo una numérica puede ir al cuadrado ({nombre!r})")

    def clave(nombre: str) -> tuple[int, int]:
        base, cuadrado = _base_de(nombre)
        return orden[base], int(cuadrado)

    return sorted(pedidas, key=clave)


def _valores(panel, nombre: str, seleccion: np.ndarray) -> tuple[str, np.ndarray]:
    """Valores crudos de una variable del diseño en las filas de ``seleccion`` (máscara sobre todo el panel).

    Devuelve ``("num"|"bin", float64 con NaN en el nulo)`` o ``("cat", códigos int32)``. ``X^2`` entrega el cuadrado de la numérica.
    """
    base, cuadrado = _base_de(nombre)
    tipo = panel.variables[base].tipo
    if tipo == "cat":
        codigos, _ = panel.codigos(base)
        return "cat", codigos[seleccion].astype(np.int32)
    x = panel.datos[base].cast(pl.Float64).fill_null(float("nan")).to_numpy()[seleccion]
    return tipo, (x * x if cuadrado else x)


def especificar_diseno(panel, variables: Sequence[str], anios_entrenamiento: Sequence[int], semilla: int, omitir_nivel_base: bool = True) -> EspecDiseno:
    """Aprende con el entrenamiento (la muestra de ``indices_entrenamiento`` si la ventana es muy grande) lo que ``aplicar_diseno`` necesita.

    Numéricas: media y desviación; una numérica con desviación cero se omite. Nulos de numéricas y binarias: se rellenan con la mediana del entrenamiento (0 en binarias) y,
    solo si el entrenamiento tuvo nulos, existe la columna indicadora ``variable__nulo``; el panel no se toca (R5). Binarias: 0/1, y una binaria constante en el entrenamiento se
    omite. Categóricas: una columna por nivel de ``panel.niveles`` ("(sin dato)" y OTROS incluidos), sin las columnas constantes; con ``omitir_nivel_base`` se omite el nivel más
    frecuente del entrenamiento (en un empate, el primero del orden de niveles) y se guarda en ``nivel_base``; SEXO es la excepción explícita: su nivel base es MUJER
    (``NIVEL_BASE_FIJO``). Sin ``omitir_nivel_base`` quedan todos los niveles (el L1 los deja todos) y ``nivel_base`` queda vacío.
    """
    orden = _ordenar(panel, variables)
    mascara = _mascara(panel, anios_entrenamiento)
    posiciones = np.flatnonzero(mascara)
    filas_entrenamiento = None
    if TOPE_FILAS_MODELOS is not None and posiciones.size > TOPE_FILAS_MODELOS:
        filas_entrenamiento = indices_entrenamiento(posiciones.size, semilla)
        posiciones = posiciones[filas_entrenamiento]
    seleccion = np.zeros(mascara.size, dtype=bool)
    seleccion[posiciones] = True
    medias: dict[str, float] = {}
    desvios: dict[str, float] = {}
    medianas: dict[str, float] = {}
    tiene_nulos: dict[str, bool] = {}
    niveles: dict[str, tuple[str, ...]] = {}
    nivel_base: dict[str, str] = {}
    columnas: list[str] = []
    for nombre in orden:
        tipo, x = _valores(panel, nombre, seleccion)
        if tipo == "cat":
            base, _ = _base_de(nombre)
            niv = tuple(panel.niveles[base])
            niveles[nombre] = niv
            conteo = np.bincount(x[x >= 0], minlength=len(niv))
            if omitir_nivel_base:
                fijo = NIVEL_BASE_FIJO.get(nombre)
                base_idx = niv.index(fijo) if fijo in niv else int(np.argmax(conteo)) if conteo.size else 0
                nivel_base[nombre] = niv[base_idx] if niv else ""
            else:
                base_idx = -1
                nivel_base[nombre] = ""
            n_ent = int((x >= 0).sum())
            for k, nivel in enumerate(niv):
                if k == base_idx or conteo[k] == 0 or conteo[k] == n_ent:        # el nivel base, y las dummies constantes (siempre 0 o siempre 1), no entran
                    continue
                columnas.append(f"{nombre}={nivel}")
            continue
        nulo = np.isnan(x)
        tiene_nulos[nombre] = bool(nulo.any())
        validos = x[~nulo]
        if tipo == "num":
            if validos.size == 0:
                continue
            media, desvio = float(validos.mean()), float(validos.std())
            if not (np.isfinite(desvio) and desvio > 0.0):
                continue                                                          # desviación cero: se omite
            medias[nombre], desvios[nombre], medianas[nombre] = media, desvio, float(np.median(validos))
        else:
            if validos.size == 0:
                continue
            medianas[nombre] = 0.0
            rellena = np.where(nulo, 0.0, x)
            if rellena.min() == rellena.max():                                    # binaria constante (tras el relleno): se omite
                continue
        columnas.append(nombre)
        if tiene_nulos[nombre]:
            columnas.append(f"{nombre}__nulo")
    return EspecDiseno(
        variables=tuple(orden), medias=medias, desvios=desvios, medianas=medianas, tiene_nulos=tiene_nulos, niveles=niveles, nivel_base=nivel_base, columnas=tuple(columnas),
        filas_entrenamiento=filas_entrenamiento,
    )


def aplicar_diseno(espec: EspecDiseno, panel, anios: Sequence[int], *, entrenamiento: bool = False) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, list[int]]]:
    """Matriz de diseño ``X`` (``csr_matrix`` float64), desenlace ``y`` (int8) y ``grupos`` (variable a índices de columna) de las filas de los años dados.

    Con ``entrenamiento=True`` usa solo la muestra ``espec.filas_entrenamiento`` de la ventana (si la hubo). Numéricas estandarizadas con el entrenamiento; nulos rellenados con
    la mediana del entrenamiento más la columna indicadora cuando el entrenamiento los tuvo. Una variable omitida por el diseño queda con una lista vacía en ``grupos``.
    """
    mascara = _mascara(panel, anios)
    posiciones = np.flatnonzero(mascara)
    if entrenamiento and espec.filas_entrenamiento is not None:
        posiciones = posiciones[espec.filas_entrenamiento]
    seleccion = np.zeros(mascara.size, dtype=bool)
    seleccion[posiciones] = True
    n = int(posiciones.size)
    y = panel.datos["y"].to_numpy()[seleccion].astype(np.int8)
    indice = {c: i for i, c in enumerate(espec.columnas)}
    filas: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    grupos: dict[str, list[int]] = {nombre: [] for nombre in espec.variables}

    def agregar(nombre: str, columna: str, valores: np.ndarray) -> None:
        j = indice[columna]
        nz = np.flatnonzero(valores != 0.0)
        filas.append(nz)
        cols.append(np.full(nz.size, j, dtype=np.int64))
        vals.append(valores[nz].astype(np.float64))
        grupos[nombre].append(j)

    for nombre in espec.variables:
        tipo, x = _valores(panel, nombre, seleccion)
        if tipo == "cat":
            correspondencia = np.full(len(espec.niveles[nombre]), -1, dtype=np.int64)
            for k, nivel in enumerate(espec.niveles[nombre]):
                j = indice.get(f"{nombre}={nivel}")
                if j is not None:
                    correspondencia[k] = j
            destino = np.where(x >= 0, correspondencia[np.clip(x, 0, None)], -1)
            hay = np.flatnonzero(destino >= 0)
            filas.append(hay)
            cols.append(destino[hay])
            vals.append(np.ones(hay.size))
            grupos[nombre].extend(int(j) for j in correspondencia[correspondencia >= 0])
            continue
        if nombre not in indice:
            continue                                                                # variable omitida por el diseño
        nulo = np.isnan(x)
        relleno = np.where(nulo, espec.medianas[nombre], x)
        if tipo == "num":
            agregar(nombre, nombre, (relleno - espec.medias[nombre]) / espec.desvios[nombre])
        else:
            agregar(nombre, nombre, relleno)
        if espec.tiene_nulos.get(nombre) and f"{nombre}__nulo" in indice:
            agregar(nombre, f"{nombre}__nulo", nulo.astype(np.float64))
    if filas:
        X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))), shape=(n, len(espec.columnas)), dtype=np.float64)
    else:
        X = sparse.csr_matrix((n, len(espec.columnas)), dtype=np.float64)
    for nombre in grupos:
        grupos[nombre] = sorted(set(grupos[nombre]))
    return X, y, grupos


# ============================================================================================================== ajustes y AUC
def _auc_logit(coef: np.ndarray, X, y: np.ndarray) -> float:
    """AUC de ``intercepto + X @ coef`` (``coef[0]`` es la intercepción): solo depende del orden de las predicciones."""
    return float(est.auc_por_rangos(_redondear_logit(np.asarray(X @ coef[1:]).ravel() + coef[0]), y)[0])


DECIMALES_LOGIT = 12
"""Decimales a los que se redondea un logit antes de ordenarlo: dos filas con el mismo logit matemático pero ruido de punto flotante de 1e-16 deben empatar, no ordenarse."""


def _redondear_logit(logit: np.ndarray) -> np.ndarray:
    return np.round(logit, DECIMALES_LOGIT)


def _ajustar_auc(X, y: np.ndarray):
    """Regresión logística con ``RIDGE_AUC``; un modelo de AUC se descarta únicamente si no convergió (la separación no interviene)."""
    return est.regresion_logistica(X, y, ridge=RIDGE_AUC)


def _conteos(y: np.ndarray) -> tuple[int, int]:
    n1 = int(np.sum(y == 1))
    return n1, int(y.size - n1)


def variables_ranking(panel) -> list[str]:
    """Candidatas que analizan el resultado (ni ETNIA ni PUEBLO_ORIGINARIO) y tienen columna en el panel, en el orden del registro."""
    return [
        n for n, v in panel.variables.items()
        if v.uso == "candidata" and v.analiza_resultado and v.evaluable != "no" and n in panel.datos.columns
    ]


def variables_modelo(panel) -> list[str]:
    """Variables que pueden entrar a los modelos multivariados (L1, permutación): las del ranking cuyo rol de la Fase 1 no es excluyente.

    Nunca COD_HOSPITAL (referencia), IR_29301_MORTALIDAD (solo fuga) ni variables de rol excluyente.
    """
    return [n for n in variables_ranking(panel) if panel.variables[n].rol_fase1 not in decision.ROLES_EXCLUYENTES]


def ajuste_base(panel, semilla: int) -> dict[str, Any]:
    """Modelo base (edad, su cuadrado y sexo) ajustado en el entrenamiento y evaluado en 2023: lo comparten el incremento de todas las variables y las referencias.

    Pásese a ``incremento_auc(..., base=...)`` para no reajustarlo por cada variable. Trae ``espec``, ``res`` (el ``ResultadoLogit``), los conteos de entrenamiento y los AUC
    de entrenamiento (``auc_ranking``) y de 2023 (``auc_2023``; ``None`` si no convergió o no hay 2023 con ambos desenlaces).
    """
    ranking, conf = list(panel.anios_ranking), panel.anio_confirmacion
    espec = especificar_diseno(panel, BASE_EDAD_SEXO, ranking, semilla, omitir_nivel_base=True)
    X, y, _ = aplicar_diseno(espec, panel, ranking, entrenamiento=True)
    res = _ajustar_auc(X, y)
    out: dict[str, Any] = {"espec": espec, "res": res, "n_entrenamiento": int(y.size), "n_eventos_entrenamiento": int(y.sum()), "auc_ranking": None, "auc_2023": None}
    if res.convergio:
        out["auc_ranking"] = _auc_logit(res.coef, X, y)
        if conf is not None:
            X2, y2, _ = aplicar_diseno(espec, panel, [conf])
            if y2.size and 0 < y2.sum() < y2.size:
                out["auc_2023"] = _auc_logit(res.coef, X2, y2)
    return out


def referencias(panel, semilla: int) -> dict[str, dict[str, Any]]:
    """AUC de los dos modelos de referencia, ajustados en la ventana de ranking y evaluados en las filas de entrenamiento (``auc_ranking``, algo optimista) y en 2023.

    ``MODELO_EDAD_SEXO`` es el modelo base (edad y su cuadrado estandarizados por separado más SEXO con HOMBRE y "(sin dato)" contra MUJER). ``MODELO_ELIX31`` es la
    intercepción más ELIX_01 a ELIX_31, SIN edad ni sexo (lectura literal del encargo), con las columnas constantes del entrenamiento omitidas. Ambos con ``RIDGE_AUC``. Cada
    entrada trae ``auc_ranking``, ``auc_2023`` (nulos si no convergió o no hay 2023), ``estado`` ("ok" o "no convergió"), ``n_columnas`` y los conteos de entrenamiento y de 2023.
    """
    ranking, conf = list(panel.anios_ranking), panel.anio_confirmacion
    salida: dict[str, dict[str, Any]] = {}
    base = ajuste_base(panel, semilla)
    for nombre, variables in (("MODELO_EDAD_SEXO", BASE_EDAD_SEXO), ("MODELO_ELIX31", tuple(elixhauser.IDS))):
        if nombre == "MODELO_EDAD_SEXO":
            espec, res = base["espec"], base["res"]
            n_ent, ev_ent, auc_r, auc_c = base["n_entrenamiento"], base["n_eventos_entrenamiento"], base["auc_ranking"], base["auc_2023"]
        else:
            espec = especificar_diseno(panel, variables, ranking, semilla, omitir_nivel_base=True)
            X, y, _ = aplicar_diseno(espec, panel, ranking, entrenamiento=True)
            res = _ajustar_auc(X, y)
            n_ent, ev_ent = int(y.size), int(y.sum())
            auc_r = auc_c = None
            if res.convergio:
                auc_r = _auc_logit(res.coef, X, y)
                if conf is not None:
                    X2, y2, _ = aplicar_diseno(espec, panel, [conf])
                    if y2.size and 0 < y2.sum() < y2.size:
                        auc_c = _auc_logit(res.coef, X2, y2)
        n_2023, ev_2023 = 0, 0
        if conf is not None:
            y2 = panel.datos.filter(pl.col("anio") == int(conf))["y"].to_numpy()
            n_2023, ev_2023 = int(y2.size), int(y2.sum())
        salida[nombre] = {
            "auc_ranking": auc_r, "auc_2023": auc_c, "estado": "ok" if res.convergio else "no convergió", "n_columnas": len(espec.columnas),
            "n_entrenamiento": n_ent, "n_eventos_entrenamiento": ev_ent, "n_2023": n_2023, "n_eventos_2023": ev_2023,
        }
    return salida


def incremento_auc(panel, variable: str, semilla: int, *, base: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """AUC en 2023 de edad, edad al cuadrado y sexo más la variable, menos el AUC en 2023 del modelo base: mismo entrenamiento y mismas filas, ambos con ``RIDGE_AUC``.

    Se calcula para toda candidata que analiza el resultado, incluidas las que la Fase 1 excluye. ``incremento_estado``: "ok", "no aplica: variable del modelo base"
    (EDAD_ANIOS y SEXO), "constante en el entrenamiento" (la variable no agrega columnas), "pocos eventos", "no convergió" o "no evaluable en 2023". La separación nunca lo
    anula: el único criterio es ``convergio``. ``base`` es el resultado de ``ajuste_base`` (opcional: sin él se ajusta el modelo base en cada llamada).
    """
    ranking, conf = list(panel.anios_ranking), panel.anio_confirmacion
    salida: dict[str, Any] = {
        "variable": variable, "incremento_auc": None, "incremento_estado": "ok", "auc_base_2023": None, "auc_con_variable_2023": None, "n_columnas_diseno": 0,
        "n_entrenamiento": 0, "n_eventos_entrenamiento": 0, "n_2023": 0, "n_eventos_2023": 0,
    }
    base = base if base is not None else ajuste_base(panel, semilla)
    salida["n_entrenamiento"], salida["n_eventos_entrenamiento"] = int(base["n_entrenamiento"]), int(base["n_eventos_entrenamiento"])
    if conf is not None:
        y2 = panel.datos.filter(pl.col("anio") == int(conf))["y"].to_numpy()
        salida["n_2023"], salida["n_eventos_2023"] = int(y2.size), int(y2.sum())
    if variable in VARIABLES_BASE:
        salida["incremento_estado"] = ESTADO_NO_APLICA
        return salida
    espec_base = base["espec"]
    espec = especificar_diseno(panel, (*BASE_EDAD_SEXO, variable), ranking, semilla, omitir_nivel_base=True)
    salida["n_columnas_diseno"] = len(espec.columnas) - len(espec_base.columnas)
    X, y, _ = aplicar_diseno(espec, panel, ranking, entrenamiento=True)
    n1, n0 = _conteos(y)
    if min(n1, n0) < _MIN_EVENTOS:
        salida["incremento_estado"] = "pocos eventos"
        return salida
    if salida["n_columnas_diseno"] <= 0:
        salida["incremento_estado"] = "constante en el entrenamiento"
        return salida
    res = _ajustar_auc(X, y)
    if not res.convergio or not base["res"].convergio:
        salida["incremento_estado"] = "no convergió"
        return salida
    auc_base = base["auc_2023"]
    if conf is None or auc_base is None:
        salida["incremento_estado"] = "no evaluable en 2023"
        return salida
    X2, y2, _ = aplicar_diseno(espec, panel, [conf])
    if not (y2.size and 0 < y2.sum() < y2.size):
        salida["incremento_estado"] = "no evaluable en 2023"
        return salida
    auc_con = _auc_logit(res.coef, X2, y2)
    salida.update({"auc_base_2023": auc_base, "auc_con_variable_2023": auc_con, "incremento_auc": auc_con - auc_base})
    return salida


# ===================================================================================================================== L1
def ajustar_l1(X, y, columnas: Sequence[str], grupos: Mapping[str, Sequence[int]], semilla: int, *, max_iter: int = 200) -> ResultadoL1:
    """Regresión logística con penalización L1 y validación cruzada (``LogisticRegressionCV`` con ``liblinear``, ``scoring="roc_auc"``).

    Es la forma ``l1_ratios=(1.0,)`` del ``penalty="l1"`` del encargo (da los mismos coeficientes y el mismo C, sin la deprecación). Los pliegues son
    ``StratifiedKFold(PLIEGUES_CV, shuffle=True, random_state=semilla)``: las filas del panel van por año y por posición en el archivo, así que sin barajar los bloques serían
    contiguos y el C elegido dependería del orden de los datos. Todo bajo un hilo de BLAS y dentro de ``warnings.catch_warnings()``, que silencia el ``FutureWarning`` de scikit-learn
    y el ``ConvergenceWarning``: la falta de convergencia no se pierde, queda en ``ResultadoL1.convergio`` (falso si algún ajuste agotó ``max_iter`` iteraciones).

    Con menos de ``_MIN_EVENTOS`` eventos por pliegue (o de no eventos) el modelo no se ajusta y ``estado`` es "no evaluable: pocos eventos"; no es fatal. Una variable
    sobrevive si alguna de sus columnas tiene coeficiente distinto de cero.
    """
    yv = np.asarray(y).astype(np.int64)
    columnas = tuple(columnas)
    grp = {k: list(v) for k, v in grupos.items()}
    n1, n0 = _conteos(yv)
    if min(n1, n0) < _MIN_EVENTOS * PLIEGUES_CV:
        return ResultadoL1(
            coef=np.full(len(columnas), np.nan), intercepto=float("nan"), c_elegido=float("nan"), columnas=columnas, grupos=grp,
            sobrevive_por_variable={k: False for k in grp}, estado=POCOS_EVENTOS_L1, convergio=False,
        )
    X_aj, y_aj = X, yv
    if TOPE_FILAS_L1 is not None and yv.size > TOPE_FILAS_L1:
        pos = np.sort(np.random.default_rng(semilla + 13).choice(yv.size, size=TOPE_FILAS_L1, replace=False))
        X_aj, y_aj = X[pos], yv[pos]
    cv = StratifiedKFold(PLIEGUES_CV, shuffle=True, random_state=semilla)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        modelo = LogisticRegressionCV(
            Cs=CS_L1, l1_ratios=(1.0,), solver="liblinear", scoring="roc_auc", cv=cv, random_state=semilla, n_jobs=1, max_iter=max_iter, refit=True,
        )
        modelo.fit(X_aj, y_aj)
        coef = np.asarray(modelo.coef_, dtype=np.float64).ravel().copy()
        intercepto = float(np.asarray(modelo.intercept_, dtype=np.float64).ravel()[0])
        c_elegido = float(np.asarray(modelo.C_, dtype=np.float64).ravel()[0])
        iteraciones = np.asarray(modelo.n_iter_)
    sobrevive = {k: bool(len(v) and np.any(coef[v] != 0.0)) for k, v in grp.items()}
    return ResultadoL1(
        coef=coef, intercepto=intercepto, c_elegido=c_elegido, columnas=columnas, grupos=grp, sobrevive_por_variable=sobrevive, estado="ok",
        convergio=bool(not np.any(iteraciones >= max_iter)),
    )


def auc_l1(modelo: ResultadoL1, X, y: np.ndarray) -> float | None:
    """AUC del modelo L1 sobre ``X`` y ``y`` (``None`` si el modelo no se ajustó o hay una sola clase)."""
    if modelo.estado != "ok" or not (0 < int(np.sum(y)) < len(y)):
        return None
    return _auc_logit(np.concatenate([[modelo.intercepto], modelo.coef]), X, y)


ESQUEMA_PERMUTACION = {"variable": pl.String, "importancia": pl.Float64, "desviacion": pl.Float64, "auc_base": pl.Float64}


def importancia_permutacion(
    X, y: np.ndarray, modelo: ResultadoL1, grupos: Mapping[str, Sequence[int]], repeticiones: int = REPETICIONES_PERMUTACION, semilla: int = 0
) -> pl.DataFrame:
    """Importancia por permutación sobre ``X`` e ``y`` (2023) con el modelo L1 ajustado en el ranking: ``AUC_base - media(AUC_permutado)`` y su desviación (ddof=1).

    Para un modelo lineal la permutación es exacta y barata: la contribución del grupo ``g`` al logit es ``c_g = X[:, g] @ coef[g]`` (``X`` pasa a CSC una vez) y, con una
    permutación ``p`` de las filas, el logit permutado es ``logit - c_g + c_g[p]``; el AUC solo necesita su orden. ``rng = default_rng(semilla + 11)``: para cada variable (en
    el orden de ``grupos``, que es el del registro) con al menos una columna, y cada repetición, se genera una permutación. El grupo de una variable incluye su columna de
    nulos. La caída puede ser negativa y no se recorta. Devuelve ``variable``, ``importancia``, ``desviacion`` (nula con una repetición) y ``auc_base``; vacía si el modelo no
    se ajustó o ``y`` tiene una sola clase.
    """
    f = "importancia_permutacion"
    if not isinstance(repeticiones, (int, np.integer)) or isinstance(repeticiones, bool) or repeticiones < 1:
        raise ValueError(f"{f}: 'repeticiones' debe ser un entero igual o mayor que 1 (recibió {repeticiones!r})")
    yv = np.asarray(y)
    if modelo.estado != "ok" or not (0 < int(yv.sum()) < yv.size):
        return pl.DataFrame(schema=ESQUEMA_PERMUTACION)
    Xc = sparse.csc_matrix(X)
    coef = np.asarray(modelo.coef, dtype=np.float64)
    auc_base = float(est.auc_por_rangos(_redondear_logit(np.asarray(Xc @ coef).ravel() + modelo.intercepto), yv)[0])
    rng = np.random.default_rng(semilla + 11)
    filas = []
    for nombre, cols in grupos.items():
        cols = list(cols)
        if not cols:
            continue
        aporte = np.asarray(Xc[:, cols] @ coef[cols]).ravel()                         # c_g: contribución del grupo al logit
        coef_sin = coef.copy()
        coef_sin[cols] = 0.0
        resto = np.asarray(Xc @ coef_sin).ravel() + modelo.intercepto                 # logit - c_g, calculado aparte para que filas con el mismo patrón empaten exactamente
        aucs = np.array([float(est.auc_por_rangos(_redondear_logit(resto + aporte[rng.permutation(yv.size)]), yv)[0]) for _ in range(int(repeticiones))])
        sd = float(aucs.std(ddof=1)) if aucs.size > 1 else None
        filas.append({"variable": nombre, "importancia": auc_base - float(aucs.mean()), "desviacion": sd, "auc_base": auc_base})
    return pl.DataFrame(filas, schema=ESQUEMA_PERMUTACION)


# ======================================================================================================== AUC univariado compartido
@dataclass(frozen=True)
class AjusteUnivariado:
    """Lo que el ranking fija con la ventana de ranking y aplica a cualquier año: la dirección y, en una categórica, la tasa suavizada por nivel."""

    variable: str
    tipo: str
    direccion: str                          # "+" si el AUC bruto del ranking es al menos 0,5, "-" si no, "" si no se pudo calcular
    auc_bruto_ranking: float                # AUC sin orientar en la ventana de ranking (NaN si no existe)
    niveles: tuple[str, ...]                # niveles de una categórica (vacío en las demás)
    tasas: np.ndarray | None                # tasa suavizada por nivel (solo categóricas)


@dataclass(frozen=True)
class ResultadoAuc:
    """AUC univariado de una ventana con la dirección del ranking."""

    auc_bruto: float
    auc: float                              # orientado: ``auc_bruto`` si la dirección es "+" y ``1 - auc_bruto`` si es "-" (puede quedar bajo 0,5 fuera del ranking)
    n: int                                  # filas usadas (con valor)
    n_eventos: int
    n_no_eventos: int
    ic_bajo: float
    ic_alto: float


def _puntaje(panel, variable: str, seleccion: np.ndarray, tasas: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Puntaje de cada fila de ``seleccion`` (el valor mismo en numéricas y binarias, con NaN en el nulo; la tasa del nivel en las categóricas) y su desenlace."""
    tipo, x = _valores(panel, variable, seleccion)
    y = panel.datos["y"].to_numpy()[seleccion].astype(np.float64)
    if tipo == "cat":
        assert tasas is not None
        return np.where(x >= 0, tasas[np.clip(x, 0, None)], np.nan), y
    return x, y


def ajustar_univariado(panel, variable: str) -> AjusteUnivariado:
    """Fija con la ventana de ranking la dirección y, en una categórica, las tasas por nivel de ``puntaje_categorico`` (suavizadas con ``SUAVIZADO_M``)."""
    tipo = panel.variables[variable].tipo
    seleccion = _mascara(panel, panel.anios_ranking)
    tasas, niveles = None, ()
    if tipo == "cat":
        codigos, niv = panel.codigos(variable)
        niveles = tuple(niv)
        y = panel.datos["y"].to_numpy()[seleccion].astype(np.float64)
        tasas = est.puntaje_categorico(codigos[seleccion], y, len(niv))
    puntaje, y = _puntaje(panel, variable, seleccion, tasas)
    bruto = float(est.auc_por_rangos(puntaje, y)[0])
    # sin una medición válida en el ranking (variable constante, una sola clase, sin valores) no hay dirección: el AUC de cualquier otra ventana no tiene con qué orientarse
    medible = estado_ventana(panel, variable, panel.anios_ranking) == "ok"
    direccion = "" if (not np.isfinite(bruto) or not medible) else ("+" if bruto >= 0.5 else "-")
    return AjusteUnivariado(variable=variable, tipo=tipo, direccion=direccion, auc_bruto_ranking=bruto, niveles=niveles, tasas=tasas)


def _orientar(bruto: float, direccion: str) -> float:
    if not np.isfinite(bruto) or direccion == "":
        return float("nan")
    return bruto if direccion == "+" else 1.0 - bruto


def _resultado_auc(bruto: float, n1: int, n0: int, direccion: str, z: float) -> ResultadoAuc:
    auc = _orientar(bruto, direccion)
    bajo, alto = est.ic_hanley_mcneil(auc, n1, n0, z) if np.isfinite(auc) else (float("nan"), float("nan"))
    return ResultadoAuc(auc_bruto=bruto, auc=auc, n=n1 + n0, n_eventos=n1, n_no_eventos=n0, ic_bajo=float(bajo), ic_alto=float(alto))


def auc_univariado(panel, ajuste: AjusteUnivariado, anios: Sequence[int], z: float = est.Z_95) -> ResultadoAuc:
    """AUC de la variable en las filas de los años dados, con el puntaje y la dirección que fijó el ranking (``ajuste``) y el IC de Hanley y McNeil.

    En la ventana de ranking una categórica es optimista (las tasas se estimaron con esas mismas filas); en 2023 es honesto. Fuera del ranking el AUC orientado puede quedar
    bajo 0,5: es una señal de inestabilidad, no un error. Sin ambas clases (o sin filas con valor) el AUC es NaN.
    """
    seleccion = _mascara(panel, anios)
    puntaje, y = _puntaje(panel, ajuste.variable, seleccion, ajuste.tasas)
    bruto, n1, n0 = est.auc_por_rangos(puntaje, y)
    return _resultado_auc(float(bruto), int(n1), int(n0), ajuste.direccion, z)


def auc_cv_univariado(panel, ajuste: AjusteUnivariado, z: float = est.Z_95) -> ResultadoAuc | None:
    """AUC de una categórica por validación cruzada dejando un año de ranking fuera: la tasa de cada nivel se estima en los otros años y el AUC se calcula UNA vez sobre las
    puntuaciones fuera de muestra agrupadas; el IC de Hanley y McNeil usa los conteos agrupados (los de eventos y no eventos de toda la ventana). ``None`` si la variable no
    es categórica o si solo hay un año de ranking con filas. Se orienta con la dirección del ranking."""
    if ajuste.tipo != "cat":
        return None
    seleccion = _mascara(panel, panel.anios_ranking)
    codigos, niv = panel.codigos(ajuste.variable)
    cod = codigos[seleccion]
    y = panel.datos["y"].to_numpy()[seleccion].astype(np.float64)
    anio = panel.datos["anio"].to_numpy()[seleccion]
    anios = [a for a in sorted(set(anio.tolist()))]
    if len(anios) < 2:
        return None
    puntaje = np.full(cod.size, np.nan)
    for a in anios:
        fuera = anio == a
        tasas = est.puntaje_categorico(cod[~fuera], y[~fuera], len(niv))
        puntaje[fuera] = np.where(cod[fuera] >= 0, tasas[np.clip(cod[fuera], 0, None)], np.nan)
    bruto, n1, n0 = est.auc_por_rangos(puntaje, y)
    return _resultado_auc(float(bruto), int(n1), int(n0), ajuste.direccion, z)


def estado_ventana(panel, variable: str, anios: Sequence[int]) -> str:
    """Estado de la medición de una variable en una ventana: "ok", "constante", "sin ambos desenlaces" o "no evaluable en la ventana".

    "No evaluable" cuando la ventana no tiene filas o la variable no tiene filas con valor; "sin ambos desenlaces" cuando las filas con valor tienen una sola clase del
    desenlace; "constante" cuando la variable toma un solo valor en esas filas.
    """
    seleccion = _mascara(panel, anios)
    tipo, x = _valores(panel, variable, seleccion)
    y = panel.datos["y"].to_numpy()[seleccion]
    valido = (x >= 0) if tipo == "cat" else ~np.isnan(x)
    if not valido.any():
        return "no evaluable en la ventana"
    yv = y[valido]
    if not (0 < int(yv.sum()) < yv.size):
        return "sin ambos desenlaces"
    xv = x[valido]
    if xv.min() == xv.max():
        return "constante"
    return "ok"


def medir_ventana(panel, ajuste: AjusteUnivariado, anios: Sequence[int], z: float = est.Z_95) -> tuple[str, ResultadoAuc]:
    """``(estado, resultado)`` del AUC univariado de una ventana: los conteos siempre salen y las cifras de AUC quedan en NaN si el estado no es "ok".

    El estado es el de ``estado_ventana`` ("ok", "constante", "sin ambos desenlaces" o "no evaluable en la ventana"): una variable constante tendría un AUC de 0,5 por rangos
    empatados, que no es una medición.
    """
    estado = estado_ventana(panel, ajuste.variable, anios)
    res = auc_univariado(panel, ajuste, anios, z)
    if estado != "ok":
        nan = float("nan")
        res = dataclasses.replace(res, auc_bruto=nan, auc=nan, ic_bajo=nan, ic_alto=nan)
    return estado, res


def columna_del_panel(panel, nombre: str, anios: Sequence[int]):
    """``asociacion.Columna`` de cualquier variable del panel (también las de solo fuga) en las filas de los años dados, para medir su asociación con el desenlace."""
    mascara = _mascara(panel, anios)
    v = panel.variables[nombre]
    if v.tipo == "num":
        return asociacion.crear_columna(nombre, "num", panel.datos[nombre].cast(pl.Float64).fill_null(float("nan")).to_numpy()[mascara])
    codigos, niveles = panel.codigos(nombre)
    return asociacion.crear_columna(nombre, v.tipo, codigos[mascara], len(niveles) if v.tipo == "cat" else None)


def valores_del_panel(panel, nombre: str, anios: Sequence[int]) -> tuple[str, np.ndarray]:
    """Valores crudos de una variable del panel en las filas de los años dados: ``("num"|"bin", float64 con NaN en el nulo)`` o ``("cat", códigos int32)``."""
    return _valores(panel, nombre, _mascara(panel, anios))


def desenlace(panel, anios: Sequence[int]) -> np.ndarray:
    """Desenlace ``y`` (0/1, int8) de las filas de los años dados."""
    return panel.datos["y"].to_numpy()[_mascara(panel, anios)].astype(np.int8)


def or_es_relevante(valor: float | None, bajo: float | None, alto: float | None, umbral: float) -> str | None:
    """"si" si el OR es al menos ``umbral`` o a lo más su inverso y el intervalo excluye 1; "no" en otro caso; ``None`` si falta alguna cifra.

    Se compara con los valores redondeados a los decimales con que se publican las tablas.
    """
    if valor is None or bajo is None or alto is None or not (np.isfinite(valor) and np.isfinite(bajo) and np.isfinite(alto)):
        return None
    v, b, a = (float(est.redondear_publicado(x)) for x in (valor, bajo, alto))
    fuera = v >= umbral or v <= 1.0 / umbral
    excluye_uno = b > 1.0 or a < 1.0
    return "si" if fuera and excluye_uno else "no"


# FIN-modelos
