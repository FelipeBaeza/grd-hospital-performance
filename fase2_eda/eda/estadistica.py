"""Funciones estadísticas de la suite. Cada función se crea en la tarea que la usa primero y tiene
una prueba de valor conocido en ``tests/test_estadistica.py``.

Aviso de interpretación: con N de millones los p-valores no discriminan; se decide por tamaño de
efecto, intervalo de confianza y estabilidad por año. Los intervalos ignoran el agrupamiento por
paciente y por hospital, por lo que son indicativos.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse, special, stats
from scipy.cluster import hierarchy
from scipy.sparse import csgraph
from scipy.spatial.distance import squareform
from threadpoolctl import threadpool_limits


# --------------------------------------------------------------------------- Chao1 (Tarea 4)
def chao1(frecuencias: Iterable[int]) -> float:
    """Estimador Chao1 de riqueza (corregido por sesgo, Chao 1987).

    ``frecuencias`` son los conteos por categoría observada (cada uno >= 1).
    S_est = S_obs + f1 (f1 - 1) / (2 (f2 + 1)), con f1 = categorías vistas una vez y f2 = vistas dos veces.
    """
    f = np.asarray([x for x in frecuencias if x and x > 0], dtype=np.int64)
    s_obs = int(f.size)
    if s_obs == 0:
        return 0.0
    f1 = int((f == 1).sum())
    f2 = int((f == 2).sum())
    return float(s_obs + f1 * (f1 - 1) / (2.0 * (f2 + 1)))


def frecuencia_de_frecuencias(conteos: Iterable[int]) -> tuple[int, int, int]:
    """(S_obs, f1, f2) a partir de los conteos por categoría."""
    f = np.asarray([x for x in conteos if x and x > 0], dtype=np.int64)
    return int(f.size), int((f == 1).sum()), int((f == 2).sum())


# ---------------------------------------------------------------------------- PSI (Tarea 4)
def psi(base: Mapping[str, float] | Sequence[float], nuevo: Mapping[str, float] | Sequence[float], eps: float = 1e-4) -> float:
    """Índice de estabilidad poblacional entre dos distribuciones de conteos.

    PSI = sum_i (q_i - p_i) ln(q_i / p_i), con p y q las proporciones base y nueva. Las proporciones
    en cero se reemplazan por ``eps`` (una categoría que aparece o desaparece produce un PSI alto,
    que es la señal buscada).
    """
    if isinstance(base, Mapping) or isinstance(nuevo, Mapping):
        b = dict(base) if isinstance(base, Mapping) else dict(enumerate(base))
        n = dict(nuevo) if isinstance(nuevo, Mapping) else dict(enumerate(nuevo))
        claves = sorted(set(b) | set(n), key=str)
        pb = np.array([float(b.get(k, 0.0)) for k in claves])
        pn = np.array([float(n.get(k, 0.0)) for k in claves])
    else:
        pb = np.asarray(base, dtype=float)
        pn = np.asarray(nuevo, dtype=float)
        if pb.shape != pn.shape:
            raise ValueError("las dos distribuciones deben tener las mismas categorías")
    if pb.sum() <= 0 or pn.sum() <= 0:
        return float("nan")
    p = np.maximum(pb / pb.sum(), eps)
    q = np.maximum(pn / pn.sum(), eps)
    return float(np.sum((q - p) * np.log(q / p)))


def clasificar_psi(valor: float, moderado: float, alto: float) -> str:
    if valor is None or np.isnan(valor):
        return "no evaluable"
    if valor >= alto:
        return "alto"
    if valor >= moderado:
        return "moderado"
    return "estable"


# ----------------------------------------------------------------- proporciones (Tarea 7)
Z_95 = 1.959964


def wilson(k, n, z: float = Z_95) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo de Wilson para una proporción k/n (acepta escalares o arreglos).

    Es el intervalo recomendado para proporciones pequeñas o con n moderado (Newcombe 1998). Devuelve
    (bajo, alto); es NaN donde n es 0. Ignora el agrupamiento por paciente y por hospital.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        z2 = z * z
        den = 1.0 + z2 / n
        centro = (p + z2 / (2.0 * n)) / den
        mitad = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / den
        bajo = np.where(n > 0, np.maximum(centro - mitad, 0.0), np.nan)
        alto = np.where(n > 0, np.minimum(centro + mitad, 1.0), np.nan)
    return bajo, alto


def razon_de_proporciones(k1, n1, k0, n0, z: float = Z_95) -> tuple[float, float, float]:
    """Razón de proporciones (k1/n1) / (k0/n0) con IC de Katz sobre el logaritmo.

    Devuelve (razon, bajo, alto). Si el grupo de referencia no tiene eventos o algún denominador es 0, todo es NaN.
    Si el grupo comparado no tiene eventos, la razón es 0, el límite inferior es 0 y el superior sale del límite
    superior de Wilson de ``k1/n1`` dividido por la proporción de referencia (se ignora la incertidumbre de la
    referencia, que es pequeña cuando ``n0`` es grande).
    """
    k1, n1, k0, n0 = (float(x) for x in (k1, n1, k0, n0))
    if k0 <= 0 or n1 <= 0 or n0 <= 0:
        return float("nan"), float("nan"), float("nan")
    if k1 <= 0:
        _, alto = wilson(0.0, n1, z)
        return 0.0, 0.0, float(alto) / (k0 / n0)
    rr = (k1 / n1) / (k0 / n0)
    se = float(np.sqrt(1.0 / k1 - 1.0 / n1 + 1.0 / k0 - 1.0 / n0))
    return rr, float(rr * np.exp(-z * se)), float(rr * np.exp(z * se))


def percentil_de_conteos(valores: Sequence[float], conteos: Sequence[int], q: float) -> float:
    """Menor valor cuya frecuencia acumulada alcanza la fracción ``q`` (0 < q <= 1) de una distribución (valor, conteo)."""
    v = np.asarray(valores, dtype=float)
    c = np.asarray(conteos, dtype=np.int64)
    if v.size == 0 or c.sum() == 0:
        return float("nan")
    orden = np.argsort(v, kind="stable")
    v, c = v[orden], c[orden]
    acum = np.cumsum(c)
    meta = q * acum[-1]
    return float(v[int(np.searchsorted(acum, meta - 1e-9, side="left"))])


def momentos_de_conteos(valores: Sequence[float], conteos: Sequence[int]) -> tuple[int, float, float]:
    """(n, media, varianza muestral) de una distribución dada como pares (valor, conteo)."""
    v = np.asarray(valores, dtype=float)
    c = np.asarray(conteos, dtype=float)
    n = float(c.sum())
    if n <= 0:
        return 0, float("nan"), float("nan")
    media = float((v * c).sum() / n)
    var = float((c * (v - media) ** 2).sum() / (n - 1.0)) if n > 1 else float("nan")
    return int(n), media, var


# ------------------------------------------------------------- gráfico de embudo (Tarea 9)
def winsorizar(z: Sequence[float], winsor: float) -> np.ndarray:
    """Winsoriza por rango: los ``floor(winsor * n)`` valores más bajos pasan al siguiente más bajo y los más altos al siguiente más alto."""
    x = np.sort(np.asarray([v for v in z if v is not None and not np.isnan(v)], dtype=float))
    n = x.size
    k = int(np.floor(winsor * n))
    if k > 0 and n > 2 * k:
        x[:k] = x[k]
        x[n - k :] = x[n - k - 1]
    return x


def phi_spiegelhalter(z: Sequence[float], winsor: float = 0.10) -> float:
    """Factor de sobredispersión multiplicativo de Spiegelhalter (2005): promedio de los z al cuadrado tras winsorizar.

    Se devuelve al menos 1: si la variación entre instituciones no supera la esperada por azar no se ajusta.
    Con menos de 3 instituciones no hay base para estimarlo y se devuelve 1.
    """
    x = winsorizar(z, winsor)
    if x.size < 3:
        return 1.0
    return float(max(1.0, np.mean(x**2)))


def limites_proporcion(p0: float, n, z: float, phi: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Límites de control de un embudo para una proporción con aproximación normal: p0 ± z sqrt(phi p0 (1 - p0) / n)."""
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mitad = z * np.sqrt(phi * p0 * (1.0 - p0) / n)
    return np.clip(p0 - mitad, 0.0, 1.0), np.clip(p0 + mitad, 0.0, 1.0)


def puntaje_z_proporcion(k, n, p0: float) -> np.ndarray:
    """z de cada institución: (p - p0) / sqrt(p0 (1 - p0) / n). NaN donde n es 0 o p0 es 0 o 1."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (k / n - p0) / np.sqrt(p0 * (1.0 - p0) / n)


# ============================================================================================
# Tareas 10 y 11: asociación entre variables y selección estadística de variables
# ============================================================================================
# Constantes técnicas. Viven aquí, junto a las funciones que las usan, y la Tarea 11 las publica en
# ``t23_parametros_seleccion.csv``. Ninguna es un umbral del protocolo (``config/umbrales.toml`` no cambia).
SUAVIZADO_M = 20.0
"""Pseudoegresos con la tasa global que se suman a cada nivel de una categórica (``puntaje_categorico``). Reduce el
sobreajuste de los niveles raros sin alterar los niveles grandes."""
EPS_COLINEALIDAD = 1e-9
"""Criterio de colinealidad exacta del encargo: una variable es combinación lineal de las demás si R2 > 1 - EPS."""
DECIMALES_PUBLICADOS = 6
"""Decimales con que ``Contexto.escribir_tabla`` escribe los flotantes; los umbrales se comparan con el valor publicado."""
SEPARACION_COEF_MAX = 12.0
"""Un coeficiente de la regresión logística se declara con separación si su valor absoluto supera este valor..."""
SEPARACION_SE_MAX = 1e3
"""... o si su error estándar supera este otro. La bandera es por coeficiente y solo importa donde se usa un error de Wald."""


# ------------------------------------------------------------------------------ validación de entradas
# Una entrada degenerada (constante, una sola clase, tabla casi vacía) es válida y devuelve NaN. Una entrada inválida rompe el
# contrato de la función y lanza ``ValueError`` con un mensaje que nombra la función y el argumento. La validación vive en cada
# función pública porque es la última capa antes del cálculo: ``np.bincount(a * nb + b)`` con un código fuera de rango mezcla
# celdas en silencio y ninguna capa superior lo detectaría.
def _error(funcion: str, mensaje: str) -> ValueError:
    return ValueError(f"{funcion}: {mensaje}")


def _exigir_mismo_largo(funcion: str, a, nombre_a: str, b, nombre_b: str) -> None:
    if len(a) != len(b):
        raise _error(funcion, f"'{nombre_a}' e '{nombre_b}' tienen largos distintos ({len(a)} y {len(b)})")


def _exigir_misma_forma(funcion: str, a, nombre_a: str, b, nombre_b: str) -> None:
    if a.shape != b.shape:
        raise _error(funcion, f"'{nombre_a}' y '{nombre_b}' tienen formas distintas ({a.shape} y {b.shape})")


def _exigir_vector(funcion: str, a, nombre: str) -> np.ndarray:
    """Arreglo float64 unidimensional."""
    v = np.asarray(a, dtype=np.float64)
    if v.ndim != 1:
        raise _error(funcion, f"'{nombre}' debe ser unidimensional (tiene {v.ndim} dimensiones)")
    return v


def _exigir_binaria(funcion: str, y: np.ndarray, nombre: str, admite_nan: bool = True) -> None:
    """``y`` solo puede valer 0, 1 o (si se admite) NaN."""
    malos = (y != 0) & (y != 1)
    if admite_nan:
        malos &= ~np.isnan(y)
    if malos.any():
        raise _error(funcion, f"'{nombre}' solo admite 0, 1{' o NaN' if admite_nan else ''} (hay {int(malos.sum())} valores distintos)")


def _exigir_entero(funcion: str, valor, nombre: str, minimo: int) -> int:
    """Parámetro entero igual o mayor que ``minimo``."""
    if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, (int, np.integer)) or valor < minimo:
        raise _error(funcion, f"'{nombre}' debe ser un entero igual o mayor que {minimo} (recibió {valor!r})")
    return int(valor)


def _exigir_codigos(funcion: str, codigos, n_niveles: int, nombre: str, nombre_n: str) -> np.ndarray:
    """Códigos enteros menores que ``n_niveles``. Un código negativo es válido: significa faltante y quien llama lo descarta."""
    c = np.asarray(codigos)
    if c.ndim != 1:
        raise _error(funcion, f"'{nombre}' debe ser unidimensional (tiene {c.ndim} dimensiones)")
    if c.size and c.dtype.kind not in "iub":
        raise _error(funcion, f"'{nombre}' debe contener códigos enteros (recibió {c.dtype})")
    c = c.astype(np.int64)
    if c.size and int(c.max()) >= n_niveles:
        raise _error(funcion, f"'{nombre}' tiene códigos mayores o iguales que '{nombre_n}' ({int(c.max())} frente a {n_niveles})")
    return c


def _exigir_conteos(funcion: str, valores, nombre: str) -> np.ndarray:
    """Conteos finitos y no negativos."""
    v = np.asarray(valores, dtype=np.float64)
    if not np.all(np.isfinite(v)) or np.any(v < 0):
        raise _error(funcion, f"'{nombre}' debe tener conteos finitos y no negativos")
    return v


def _exigir_probabilidades(funcion: str, p, nombre: str) -> np.ndarray:
    """Probabilidades en [0, 1]; el NaN pasa (es un faltante)."""
    v = np.asarray(p, dtype=np.float64)
    malos = ~np.isnan(v) & ((v < 0) | (v > 1))
    if malos.any():
        raise _error(funcion, f"'{nombre}' debe estar en [0, 1] (hay {int(malos.sum())} valores fuera de rango)")
    return v


def _exigir_cuadrada(funcion: str, m, nombre: str, finita: bool = False) -> np.ndarray:
    """Matriz cuadrada de float64; con ``finita`` también sin NaN ni infinitos."""
    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise _error(funcion, f"'{nombre}' debe ser una matriz cuadrada (forma {a.shape})")
    if finita and not np.all(np.isfinite(a)):
        raise _error(funcion, f"'{nombre}' tiene valores no finitos")
    return a


def _exigir_parametro(funcion: str, cumple: bool, nombre: str, requisito: str) -> None:
    if not cumple:
        raise _error(funcion, f"'{nombre}' {requisito}")


def redondear_publicado(x):
    """Redondea al número de decimales con que se publican las tablas (``DECIMALES_PUBLICADOS``).

    Es el valor que el lector ve en el CSV, así que toda comparación contra un umbral usa este valor y la tabla no puede discrepar
    de la decisión (por ejemplo, 0,7999996 se publica como 0,800000 y debe contarse como redundante). Usa ``np.round`` y no el
    ``round`` de Python porque en los empates decimales exactos difieren (``round(0.7999995, 6)`` da 0.799999 y ``np.round`` da 0.8,
    igual que polars, que es quien escribe el CSV). Acepta escalares y arreglos; el NaN pasa igual.
    """
    return np.round(x, DECIMALES_PUBLICADOS)


# ----------------------------------------------------------------------------------- medidas bivariadas
def spearman(x, y) -> tuple[float, int]:
    """Correlación de rangos de Spearman con rangos promedio en los empates (Pearson de los rangos).

    Descarta los pares con NaN. Devuelve ``(rho, n)`` con ``n`` los pares usados; ``rho`` es NaN con menos de 3 pares o si alguna de
    las dos variables es constante.
    """
    xv, yv = _exigir_vector("spearman", x, "x"), _exigir_vector("spearman", y, "y")
    _exigir_mismo_largo("spearman", xv, "x", yv, "y")
    ok = ~(np.isnan(xv) | np.isnan(yv))
    n = int(ok.sum())
    if n < 3:
        return float("nan"), n
    rx, ry = stats.rankdata(xv[ok]), stats.rankdata(yv[ok])
    if rx.min() == rx.max() or ry.min() == ry.max():
        return float("nan"), n
    dx, dy = rx - rx.mean(), ry - ry.mean()
    rho = float(dx @ dy / np.sqrt((dx @ dx) * (dy @ dy)))
    return float(np.clip(rho, -1.0, 1.0)), n


def phi_pares(X, V) -> tuple[np.ndarray, np.ndarray]:
    """Coeficiente phi entre todas las columnas binarias de ``X``, usando solo las filas válidas en ambas de cada par.

    ``X`` es n por p con ceros y unos (el valor donde ``V`` es falso se ignora) y ``V`` marca las observaciones válidas. Todo sale de
    cuatro productos matriciales (``(X*V).T @ (X*V)``, ``(X*V).T @ V`` y ``V.T @ V``), sin recorrer los pares. Para el par (i, j) con
    ``n`` filas válidas en ambas, ``s_i`` y ``s_j`` unos y ``a`` unos en ambas: ``phi = (n a - s_i s_j) / sqrt(s_i (n - s_i) s_j (n - s_j))``,
    que es el coeficiente de correlación de Matthews. Devuelve ``(phi, n_pares)`` p por p. Un denominador cero da NaN, también en la
    diagonal de una variable sin variación: ``phi_pares`` solo mide y quien la llama impone el 1 de la diagonal.
    """
    Xa, Va = np.asarray(X, dtype=np.float64), np.asarray(V)
    if Xa.ndim != 2:
        raise _error("phi_pares", f"'X' debe ser una matriz (tiene {Xa.ndim} dimensiones)")
    _exigir_misma_forma("phi_pares", Xa, "X", Va, "V")
    if Va.dtype != bool:
        _exigir_parametro("phi_pares", bool(np.all((Va == 0) | (Va == 1))), "V", "solo admite verdadero o falso (0 o 1)")
        Va = Va.astype(bool)
    Xv = np.where(Va, Xa, 0.0)
    _exigir_parametro("phi_pares", bool(np.all((Xv == 0) | (Xv == 1))), "X", "solo admite 0 y 1 donde la observación es válida")
    Vf = Va.astype(np.float64)
    with threadpool_limits(limits=1):
        n_ij = Vf.T @ Vf
        a_ij = Xv.T @ Xv
        s_ij = Xv.T @ Vf          # unos de i entre las filas donde j también es válida
    s_ji = s_ij.T
    num = n_ij * a_ij - s_ij * s_ji
    den = np.sqrt(s_ij * (n_ij - s_ij) * s_ji * (n_ij - s_ji))
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(den > 0, num / den, np.nan)
    return np.clip(phi, -1.0, 1.0), n_ij.astype(np.int64)


def tabla_contingencia(a, b, na: int, nb: int) -> np.ndarray:
    """Tabla de contingencia ``na`` por ``nb`` de dos vectores de códigos enteros (``np.bincount(a * nb + b)``).

    Descarta las filas donde algún código es negativo (faltante). Un código mayor o igual que el número de niveles declarado es
    inválido: mezclaría celdas en silencio.
    """
    na = _exigir_entero("tabla_contingencia", na, "na", 1)
    nb = _exigir_entero("tabla_contingencia", nb, "nb", 1)
    ca = _exigir_codigos("tabla_contingencia", a, na, "a", "na")
    cb = _exigir_codigos("tabla_contingencia", b, nb, "b", "nb")
    _exigir_mismo_largo("tabla_contingencia", ca, "a", cb, "b")
    ok = (ca >= 0) & (cb >= 0)
    return np.bincount(ca[ok] * nb + cb[ok], minlength=na * nb).reshape(na, nb)


def v_cramer_sesgo(tabla) -> float:
    """V de Cramér corregida por sesgo (Bergsma 2013) de una tabla de contingencia.

    ``phi2 = chi2 / n``; ``phi2c = max(0, phi2 - (r-1)(c-1)/(n-1))``; ``r~ = r - (r-1)^2/(n-1)`` y ``c~`` igual; ``V = sqrt(phi2c / min(r~-1, c~-1))``.
    ``r`` y ``c`` cuentan solo las filas y columnas no vacías. Devuelve NaN, sin lanzar excepción, si ``n <= 1``, si ``r < 2`` o ``c < 2``
    o si ``min(r~-1, c~-1) <= 0`` (ocurre cuando ``n`` no supera a ``r`` o a ``c``, porque ``r~ - 1 = (r-1)(n-r)/(n-1)``). El resultado está
    en [0, 1] y nunca supera a la V sin corregir. Atención: ``scipy.stats.contingency.association(correction=True)`` aplica la corrección de
    continuidad de Yates, que es otra cosa.
    """
    t = np.asarray(tabla, dtype=np.float64)
    if t.ndim != 2:
        raise _error("v_cramer_sesgo", f"'tabla' debe ser una matriz (tiene {t.ndim} dimensiones)")
    _exigir_conteos("v_cramer_sesgo", t, "tabla")
    filas, cols = t.sum(axis=1), t.sum(axis=0)
    t = t[filas > 0][:, cols > 0]
    r, c = t.shape
    n = float(t.sum())
    if n <= 1 or r < 2 or c < 2:
        return float("nan")
    esperado = np.outer(t.sum(axis=1), t.sum(axis=0)) / n
    phi2 = float(((t - esperado) ** 2 / esperado).sum() / n)
    phi2c = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_t = r - (r - 1) ** 2 / (n - 1)
    c_t = c - (c - 1) ** 2 / (n - 1)
    den = min(r_t - 1.0, c_t - 1.0)
    if not den > 0:
        return float("nan")
    return float(min(1.0, np.sqrt(phi2c / den)))


def eta_cuadrado(valores, grupos, n_grupos: int | None = None) -> tuple[float, int]:
    """Razón de correlación eta al cuadrado: ``SSB / SST`` de una variable numérica según los grupos (códigos enteros).

    Descarta los NaN y los códigos negativos. Devuelve ``(eta2, n)``; es NaN con un solo grupo no vacío o con ``SST = 0``. Equivale a
    ``F (k-1) / (F (k-1) + n - k)`` del ANOVA de un factor. Con el valor crudo, el sesgo esperado bajo independencia es cercano a
    ``sqrt((k-1)/(n-1))`` para su raíz; es pequeño con los volúmenes del panel y se declara como límite.
    """
    v = _exigir_vector("eta_cuadrado", valores, "valores")
    if n_grupos is not None:
        n_grupos = _exigir_entero("eta_cuadrado", n_grupos, "n_grupos", 1)
    g = np.asarray(grupos)
    _exigir_mismo_largo("eta_cuadrado", v, "valores", g, "grupos")
    g = _exigir_codigos("eta_cuadrado", g, n_grupos if n_grupos is not None else np.iinfo(np.int64).max, "grupos", "n_grupos")
    ok = ~np.isnan(v) & (g >= 0)
    n = int(ok.sum())
    v, g = v[ok], g[ok]
    if n < 2 or v.min() == v.max():
        return float("nan"), n
    k = int(g.max()) + 1 if n_grupos is None else n_grupos
    cuenta = np.bincount(g, minlength=k).astype(np.float64)
    suma = np.bincount(g, weights=v, minlength=k)
    presente = cuenta > 0
    if presente.sum() < 2:
        return float("nan"), n
    media = float(v.mean())
    ssb = float((cuenta[presente] * (suma[presente] / cuenta[presente] - media) ** 2).sum())
    sst = float(((v - media) ** 2).sum())
    if not sst > 0:
        return float("nan"), n
    return float(min(1.0, ssb / sst)), n


def auc_por_rangos(puntaje, y) -> tuple[float, int, int]:
    """Área bajo la curva ROC por rangos (Mann-Whitney): ``(R1 - n1 (n1 + 1) / 2) / (n1 n0)`` con rangos promedio.

    Descarta los NaN de ``puntaje`` y de ``y``. Devuelve ``(auc, n_eventos, n_no_eventos)``; con una sola clase el AUC es NaN.
    """
    p = _exigir_vector("auc_por_rangos", puntaje, "puntaje")
    yv = _exigir_vector("auc_por_rangos", y, "y")
    _exigir_mismo_largo("auc_por_rangos", p, "puntaje", yv, "y")
    _exigir_binaria("auc_por_rangos", yv, "y")
    ok = ~np.isnan(p) & ~np.isnan(yv)
    p, yv = p[ok], yv[ok]
    n1, n0 = int((yv == 1).sum()), int((yv == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), n1, n0
    rangos = stats.rankdata(p)
    auc = (float(rangos[yv == 1].sum()) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc), n1, n0


def ic_hanley_mcneil(auc: float, n1: int, n0: int, z: float = Z_95) -> tuple[float, float]:
    """Intervalo de confianza del AUC de Hanley y McNeil (1982).

    ``Q1 = A / (2 - A)``, ``Q2 = 2 A^2 / (1 + A)`` y ``var = [A (1-A) + (n1-1)(Q1 - A^2) + (n0-1)(Q2 - A^2)] / (n1 n0)``; los límites son
    ``A ± z sqrt(var)`` recortados a [0, 1]. Con ``n1 < 1`` o ``n0 < 1`` (o AUC NaN) devuelve ``(nan, nan)``. Ignora el agrupamiento por
    paciente y por hospital, así que es indicativo.
    """
    a = float(_exigir_probabilidades("ic_hanley_mcneil", auc, "auc"))
    n1 = float(_exigir_conteos("ic_hanley_mcneil", n1, "n1"))
    n0 = float(_exigir_conteos("ic_hanley_mcneil", n0, "n0"))
    if n1 < 1 or n0 < 1 or np.isnan(a):
        return float("nan"), float("nan")
    q1, q2 = a / (2.0 - a), 2.0 * a * a / (1.0 + a)
    var = (a * (1.0 - a) + (n1 - 1.0) * (q1 - a * a) + (n0 - 1.0) * (q2 - a * a)) / (n1 * n0)
    ee = float(np.sqrt(max(var, 0.0)))
    return max(0.0, a - z * ee), min(1.0, a + z * ee)


def or_woolf(a, b, c, d, z: float = Z_95) -> tuple[float, float, float, bool]:
    """Razón de momios (odds ratio) con el intervalo de Woolf sobre el logaritmo.

    ``a`` expuestos con evento, ``b`` expuestos sin evento, ``c`` no expuestos con evento y ``d`` no expuestos sin evento.
    ``OR = ad / (bc)`` y ``EE = sqrt(1/a + 1/b + 1/c + 1/d)``; el IC es ``exp(log OR ± z EE)``. Si alguna celda es 0 se suma 0,5 a las cuatro
    (Haldane-Anscombe) y ``corregido`` es verdadero. Una fila o columna completa en cero no define un OR y devuelve NaN.
    Devuelve ``(or, bajo, alto, corregido)``.
    """
    a, b, c, d = (float(_exigir_conteos("or_woolf", v, nombre)) for v, nombre in ((a, "a"), (b, "b"), (c, "c"), (d, "d")))
    if a + b == 0 or c + d == 0 or a + c == 0 or b + d == 0:
        return float("nan"), float("nan"), float("nan"), False
    corregido = min(a, b, c, d) == 0
    if corregido:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    razon = a * d / (b * c)
    ee = float(np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d))
    return float(razon), float(np.exp(np.log(razon) - z * ee)), float(np.exp(np.log(razon) + z * ee)), bool(corregido)


def p_wald(beta, se):
    """Valor p bilateral de Wald, ``erfc(|beta / se| / sqrt(2))``. NaN si ``se <= 0`` o NaN. Acepta escalares y arreglos."""
    b, s = np.asarray(beta, dtype=np.float64), np.asarray(se, dtype=np.float64)
    if b.shape != s.shape:
        raise _error("p_wald", f"'beta' y 'se' tienen formas distintas ({b.shape} y {s.shape})")
    with np.errstate(divide="ignore", invalid="ignore"):
        p = special.erfc(np.abs(b / s) / np.sqrt(2.0))
    p = np.where(s > 0, p, np.nan)
    return float(p) if p.ndim == 0 else p


def benjamini_hochberg(p) -> np.ndarray:
    """Valores q de Benjamini y Hochberg (1995): ``q_(i) = min_{j >= i} (m p_(j) / j)`` acotado a 1, en el orden original.

    Los NaN se conservan y no cuentan en ``m``.
    """
    pv = _exigir_vector("benjamini_hochberg", p, "p")
    _exigir_probabilidades("benjamini_hochberg", pv, "p")
    q = np.full(pv.shape, np.nan)
    ok = ~np.isnan(pv)
    m = int(ok.sum())
    if m == 0:
        return q
    validos = pv[ok]
    orden = np.argsort(validos, kind="stable")
    ajustado = validos[orden] * m / np.arange(1, m + 1)
    ajustado = np.minimum(np.minimum.accumulate(ajustado[::-1])[::-1], 1.0)
    salida = np.empty(m)
    salida[orden] = ajustado
    q[ok] = salida
    return q


def informacion_mutua_normalizada(a, b) -> float:
    """Información mutua normalizada de dos vectores de códigos enteros: ``2 I / (H(a) + H(b))`` con logaritmo natural.

    Es el promedio aritmético de las entropías, la misma normalización de scikit-learn. Descarta las filas con algún código negativo
    (faltante). Si una variable es constante ``I = 0`` y el resultado es 0,0; si las dos lo son, el cociente es 0/0 y devuelve NaN.
    """
    ca, cb = np.asarray(a), np.asarray(b)
    _exigir_mismo_largo("informacion_mutua_normalizada", ca, "a", cb, "b")
    for codigos, nombre in ((ca, "a"), (cb, "b")):
        if codigos.ndim != 1 or (codigos.size and codigos.dtype.kind not in "iub"):
            raise _error("informacion_mutua_normalizada", f"'{nombre}' debe ser un vector de códigos enteros (recibió {codigos.dtype})")
    ca, cb = ca.astype(np.int64), cb.astype(np.int64)
    ok = (ca >= 0) & (cb >= 0)
    ca, cb = ca[ok], cb[ok]
    n = ca.size
    if n == 0:
        return float("nan")
    _, ia = np.unique(ca, return_inverse=True)
    _, ib = np.unique(cb, return_inverse=True)
    ka, kb = int(ia.max()) + 1, int(ib.max()) + 1
    if ka == 1 and kb == 1:
        return float("nan")
    if ka == 1 or kb == 1:
        return 0.0
    conjunta = np.bincount(ia * kb + ib, minlength=ka * kb).reshape(ka, kb) / n
    pa, pb = conjunta.sum(axis=1), conjunta.sum(axis=0)
    nz = conjunta > 0
    info = float((conjunta[nz] * np.log(conjunta[nz] / np.outer(pa, pb)[nz])).sum())
    ha, hb = float(-(pa * np.log(pa)).sum()), float(-(pb * np.log(pb)).sum())
    return float(min(1.0, max(0.0, 2.0 * info / (ha + hb))))


# ------------------------------------------------------------------------------- regresión logística
@dataclass(frozen=True)
class ResultadoLogit:
    """Resultado de ``regresion_logistica``.

    ``coef``, ``se`` y ``separacion_por_coef`` tienen un elemento por columna de ``X``; si hay intercepción, va PRIMERO (el
    elemento 0) y las columnas de ``X`` empiezan en el 1. ``cov`` es la matriz de covarianza de Wald (sin el ridge).
    ``convergio`` solo dice que el método terminó (la devianza cambió menos que ``tol`` en términos relativos antes de ``max_iter``) y
    es independiente de ``separacion``. ``devianza`` es ``-2 logverosimilitud`` sin el término del ridge y ``n`` el número de filas.
    """

    coef: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    convergio: bool
    n_iter: int
    separacion: bool
    separacion_por_coef: np.ndarray
    devianza: float
    n: int


def regresion_logistica(
    X, y, *, intercepto: bool = True, max_iter: int = 50, tol: float = 1e-8, ridge: float = 0.0, bloque: int = 20_000
) -> ResultadoLogit:
    """Regresión logística por mínimos cuadrados reponderados (Newton) en numpy, con errores estándar de Wald.

    ``X`` es un ``numpy.ndarray`` o cualquier matriz de ``scipy.sparse`` (se convierte a CSR una vez) y ``y`` solo puede valer 0 o 1.
    ``X'WX`` y ``X'(y - mu)`` se acumulan por bloques de ``bloque`` filas y cada bloque se densifica solo mientras se usa, de modo que una
    matriz de 200.000 filas por 450 columnas dispersas no ocupa memoria densa completa. Si la devianza sube, el paso se reduce a la mitad.

    ``ridge`` es el lambda de la función objetivo ``-logverosimilitud + (lambda / 2) * suma(coef_j ** 2)`` sobre los coeficientes que no son
    la intercepción, o sea la convención de scikit-learn con ``C = 1 / ridge``. Los errores estándar y ``cov`` salen de la inversa de ``X'WX``
    SIN el ridge y solo tienen sentido cuando ``ridge = 0``.

    Separación: ``separacion_por_coef[j]`` es verdadero si ``abs(coef[j]) > SEPARACION_COEF_MAX`` o ``se[j] > SEPARACION_SE_MAX``;
    ``separacion`` es verdadero si alguno lo es. Quien necesita un OR lo anula solo si su coeficiente está marcado o ``convergio`` es falso;
    quien necesita un AUC ignora la separación (los niveles raros sin eventos son ceros verdaderos que no invalidan las predicciones).
    Matriz singular: el paso se resuelve con ``lstsq`` (solución de mínima norma) y los coeficientes implicados en la dependencia lineal quedan
    con error estándar infinito, o sea marcados.
    """
    f = "regresion_logistica"
    es_dispersa = sparse.issparse(X)
    if es_dispersa:
        Xs = sparse.csr_matrix(X, dtype=np.float64)
        n, p = Xs.shape
        _exigir_parametro(f, bool(np.all(np.isfinite(Xs.data))), "X", "tiene valores no finitos")
    else:
        Xd = np.asarray(X, dtype=np.float64)
        _exigir_parametro(f, Xd.ndim == 2, "X", f"debe ser una matriz (tiene {Xd.ndim} dimensiones)")
        n, p = Xd.shape
        _exigir_parametro(f, bool(np.all(np.isfinite(Xd))), "X", "tiene valores no finitos")
    yv = _exigir_vector(f, y, "y")
    if len(yv) != n:
        raise _error(f, f"las filas de 'X' y 'y' no coinciden ({n} y {len(yv)})")
    _exigir_binaria(f, yv, "y", admite_nan=False)
    _exigir_entero(f, max_iter, "max_iter", 1)
    _exigir_entero(f, bloque, "bloque", 1)
    _exigir_parametro(f, bool(np.isfinite(ridge)) and ridge >= 0, "ridge", f"no puede ser negativo (recibió {ridge!r})")
    _exigir_parametro(f, bool(np.isfinite(tol)) and tol > 0, "tol", f"debe ser positivo (recibió {tol!r})")
    _exigir_parametro(f, n > 0, "X", "no tiene filas")
    k = p + int(bool(intercepto))
    _exigir_parametro(f, k > 0, "X", "no tiene columnas ni intercepción")
    lam = np.full(k, float(ridge))
    if intercepto:
        lam[0] = 0.0

    def acumular(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """X'WX, X'(y - mu) y la devianza en ``beta``, por bloques de filas."""
        hes, grad, logver = np.zeros((k, k)), np.zeros(k), 0.0
        for i0 in range(0, n, bloque):
            i1 = min(i0 + bloque, n)
            xb = Xs[i0:i1].toarray() if es_dispersa else Xd[i0:i1]
            if intercepto:
                a = np.empty((i1 - i0, k))
                a[:, 0] = 1.0
                a[:, 1:] = xb
            else:
                a = xb
            eta = a @ beta
            mu = special.expit(eta)
            w = mu * (1.0 - mu)
            yb = yv[i0:i1]
            hes += a.T @ (a * w[:, None])
            grad += a.T @ (yb - mu)
            logver += float(np.sum(yb * eta - np.logaddexp(0.0, eta)))
        return hes, grad, -2.0 * logver

    with threadpool_limits(limits=1):
        beta = np.zeros(k)
        hes, grad, dev = acumular(beta)
        obj = dev                       # devianza penalizada (dos veces el objetivo): sirve para decidir el paso
        convergio, n_iter = False, 0
        for it in range(1, max_iter + 1):
            n_iter = it
            paso_newton = np.linalg.lstsq(hes + np.diag(lam), grad - lam * beta, rcond=None)[0]
            paso = 1.0
            aceptado = False
            for _ in range(30):
                nuevo = beta + paso * paso_newton
                hes_n, grad_n, dev_n = acumular(nuevo)
                obj_n = dev_n + float(np.sum(lam * nuevo**2))
                if obj_n <= obj + 1e-10 * (abs(obj) + 0.1):
                    aceptado = True
                    break
                paso /= 2.0
            if not aceptado:            # ya no hay mejora posible con la precisión numérica disponible
                convergio = True
                break
            cambio = abs(obj - obj_n) / (abs(obj_n) + 0.1)
            beta, hes, grad, dev, obj = nuevo, hes_n, grad_n, dev_n, obj_n
            if cambio < tol:
                convergio = True
                break

        # covarianza de Wald sin el ridge, tolerando la singularidad
        valores, vectores = np.linalg.eigh(hes)
        nulos = valores <= 1e-10 * max(float(valores.max()), 1e-300)
        inversos = np.where(nulos, 0.0, 1.0 / np.where(nulos, 1.0, valores))
        cov = (vectores * inversos) @ vectores.T
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        if nulos.any():
            implicados = np.any(np.abs(vectores[:, nulos]) > 1e-6, axis=1)
            se[implicados] = np.inf
            cov[implicados, :] = np.nan
            cov[:, implicados] = np.nan
    por_coef = (np.abs(beta) > SEPARACION_COEF_MAX) | (se > SEPARACION_SE_MAX)
    return ResultadoLogit(
        coef=beta, se=se, cov=cov, convergio=bool(convergio), n_iter=int(n_iter), separacion=bool(por_coef.any()),
        separacion_por_coef=por_coef, devianza=float(dev), n=int(n),
    )


# ----------------------------------------------------------------------------- colinealidad y agrupamiento
def vif_desde_correlacion(R) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Factor de inflación de la varianza de cada variable a partir de la matriz de correlación ``R``.

    Para cada ``j``: ``beta = lstsq(R[o, o], R[o, j])`` con ``o`` las demás variables y ``r2 = R[j, o] @ beta`` recortado a [0, 1]. Es la
    regresión de la variable estandarizada contra las demás (mismas ecuaciones normales) y evita 60 regresiones sobre cientos de miles de
    filas. ``exacta`` es ``r2 > 1 - EPS_COLINEALIDAD`` (colinealidad exacta: el VIF es nulo, NaN) y ``vif = 1 / (1 - r2)`` en otro caso.
    Devuelve ``(vif, r2, exacta, rango)`` con ``rango = matrix_rank(R, tol=1e-9)``. Las variables constantes (fila de ceros) las excluye
    quien llama antes de armar ``R``.
    """
    r = _exigir_cuadrada("vif_desde_correlacion", R, "R", finita=True)
    p = r.shape[0]
    r2 = np.zeros(p)
    with threadpool_limits(limits=1):
        for j in range(p):
            otras = [i for i in range(p) if i != j]
            if not otras:
                continue
            beta = np.linalg.lstsq(r[np.ix_(otras, otras)], r[otras, j], rcond=None)[0]
            r2[j] = float(np.clip(r[j, otras] @ beta, 0.0, 1.0))
        rango = int(np.linalg.matrix_rank(r, tol=1e-9)) if p else 0
    exacta = r2 > 1.0 - EPS_COLINEALIDAD
    with np.errstate(divide="ignore", invalid="ignore"):
        vif = np.where(exacta, np.nan, 1.0 / (1.0 - r2))
    return vif, r2, exacta, rango


_METODOS_ENLACE = ("single", "complete", "average", "weighted")


def orden_por_clusters(distancia, metodo: str = "average") -> tuple[np.ndarray, np.ndarray]:
    """Orden de las hojas de un agrupamiento jerárquico de ``scipy.cluster.hierarchy`` y la matriz de enlace ``Z``.

    Antes de agrupar: simetriza ``(D + D.T) / 2``, recorta a [0, 1], pasa el NaN a 1 (una variable sin medición queda lejos de todas) y
    deja la diagonal en 0. Con menos de 2 hojas el orden es trivial y ``Z`` queda vacía.
    """
    d = _exigir_cuadrada("orden_por_clusters", distancia, "distancia")
    _exigir_parametro("orden_por_clusters", metodo in _METODOS_ENLACE, "metodo", f"debe ser uno de {_METODOS_ENLACE} (recibió {metodo!r})")
    n = d.shape[0]
    if n < 2:
        return np.arange(n), np.zeros((0, 4))
    d = np.clip((d + d.T) / 2.0, 0.0, 1.0)
    d = np.where(np.isnan(d), 1.0, d)
    np.fill_diagonal(d, 0.0)
    z = hierarchy.linkage(squareform(d, checks=False), method=metodo)
    return hierarchy.leaves_list(z).astype(np.int64), z


def componentes_redundantes(A, umbral: float) -> np.ndarray:
    """Componentes conexas del grafo cuyas aristas son los pares ``i != j`` con ``redondear_publicado(A[i, j]) >= umbral``.

    Un componente puede encadenar (A~B y B~C sin que A y C estén asociadas). El NaN no es arista. Devuelve la etiqueta de componente
    de cada variable (0, 1, 2... por orden de la primera variable de cada componente).
    """
    a = _exigir_cuadrada("componentes_redundantes", A, "A")
    _exigir_parametro("componentes_redundantes", bool(np.isfinite(umbral)), "umbral", f"debe ser un número finito (recibió {umbral!r})")
    arista = redondear_publicado(a) >= umbral
    np.fill_diagonal(arista, False)
    _, etiquetas = csgraph.connected_components(sparse.csr_matrix(arista), directed=False)
    return etiquetas.astype(np.int64)


def puntaje_categorico(codigos, y, n_niveles: int, m: float = SUAVIZADO_M) -> np.ndarray:
    """Tasa de eventos suavizada de cada nivel de una categórica: ``(k_l + m p0) / (n_l + m)`` con ``p0 = sum(k) / sum(n)``.

    Es el puntaje con que una categórica entra al AUC univariado: se estima con la ventana de ranking y se aplica a cualquier ventana.
    Un nivel sin observaciones recibe ``p0`` sin tratamiento aparte; sin suavizado (``m = 0``) un nivel con una sola observación
    valdría 0 o 1. Descarta los códigos negativos y los ``y`` NaN. Devuelve un arreglo de largo ``n_niveles`` (todo NaN si no hay datos).
    """
    f = "puntaje_categorico"
    n_niveles = _exigir_entero(f, n_niveles, "n_niveles", 1)
    yv = _exigir_vector(f, y, "y")
    c = _exigir_codigos(f, codigos, n_niveles, "codigos", "n_niveles")
    _exigir_mismo_largo(f, c, "codigos", yv, "y")
    _exigir_binaria(f, yv, "y")
    _exigir_parametro(f, bool(np.isfinite(m)) and m >= 0, "m", f"no puede ser negativo (recibió {m!r})")
    ok = (c >= 0) & ~np.isnan(yv)
    n_l = np.bincount(c[ok], minlength=n_niveles).astype(np.float64)
    k_l = np.bincount(c[ok], weights=yv[ok], minlength=n_niveles)
    total = float(n_l.sum())
    if total == 0:
        return np.full(n_niveles, np.nan)
    p0 = float(k_l.sum()) / total
    den = n_l + m
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, (k_l + m * p0) / den, p0)


def bordes_deciles(valores, n_bins: int = 10) -> np.ndarray:
    """Bordes de los bins por cuantiles (0,1 a 0,9 con ``n_bins = 10``) de los valores no nulos, sin repetidos.

    Si todos los valores son iguales no hay bordes y la variable cae en un solo bin. Se fijan con la ventana de ranking.
    """
    _exigir_entero("bordes_deciles", n_bins, "n_bins", 2)
    v = _exigir_vector("bordes_deciles", valores, "valores")
    v = v[~np.isnan(v)]
    if v.size == 0 or v.min() == v.max():
        return np.zeros(0)
    return np.unique(np.quantile(v, np.arange(1, n_bins) / n_bins))


def asignar_bins(valores, bordes) -> np.ndarray:
    """Código de bin de cada valor: el bin ``i`` es ``bordes[i-1] < v <= bordes[i]`` (``searchsorted`` por la izquierda).

    El valor nulo (NaN) recibe el código ``len(bordes) + 1``, un bin propio.
    """
    v = _exigir_vector("asignar_bins", valores, "valores")
    b = _exigir_vector("asignar_bins", bordes, "bordes")
    _exigir_parametro("asignar_bins", bool(np.all(np.diff(b) > 0)), "bordes", "debe ser estrictamente creciente")
    codigos = np.searchsorted(b, v, side="left").astype(np.int32)
    return np.where(np.isnan(v), np.int32(b.size + 1), codigos).astype(np.int32)


def psi_de_codigos(base, nuevo, n_codigos: int) -> float:
    """Índice de estabilidad poblacional (``psi``) entre dos vectores de códigos enteros (se descartan los negativos)."""
    f = "psi_de_codigos"
    n_codigos = _exigir_entero(f, n_codigos, "n_codigos", 1)
    cb = _exigir_codigos(f, base, n_codigos, "base", "n_codigos")
    cn = _exigir_codigos(f, nuevo, n_codigos, "nuevo", "n_codigos")
    conteo_b = np.bincount(cb[cb >= 0], minlength=n_codigos).astype(np.float64)
    conteo_n = np.bincount(cn[cn >= 0], minlength=n_codigos).astype(np.float64)
    return psi(conteo_b, conteo_n)


# FIN-estadistica
