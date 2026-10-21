"""Tarea 10: asociación entre variables, redundancia y multicolinealidad (CRISP-DM 2.3).

Consume el panel compartido y NO lee datos: ``ctx.lector`` no se llama aquí (la única lectura de las Tareas 10 y 11 es ``panel.construir_panel``). Analiza las
variables candidatas evaluables del panel más la referencia COD_HOSPITAL (que se mide, pero no entra a clusters, pares ni VIF) en la ventana de ranking
(2019 a 2022) y reestima en 2023 solo los pares redundantes.

Medida por tipo de par (todas en [0, 1] y, cuando existe, con signo; la eliminación de faltantes es por pares):

- numérica con numérica: valor absoluto de rho de Spearman (``spearman``);
- binaria con binaria: valor absoluto de phi (``phi``, el coeficiente de correlación de Matthews);
- numérica con binaria: raíz de eta al cuadrado (``eta``), con el signo de la media de x dado 1 menos la media dado 0;
- numérica con categórica: raíz de eta al cuadrado (``eta``), sin signo;
- binaria o categórica con categórica: V de Cramér corregida por sesgo de Bergsma (``v_cramer_bergsma``), sin signo;
- diagonal: ``identidad``, siempre 1.

Casos degenerados: una variable constante en sus filas válidas da valor nulo y ``estado = "constante"`` en todos sus pares y la diagonal 1 con
``estado = "constante"``; menos de tres pares válidos, o una tabla con menos de dos filas o columnas no vacías, da ``estado = "sin datos suficientes"``. En ambos
casos el valor es nulo, la variable no hace aristas y su distancia pasa a 1. Toda comparación contra un umbral usa ``estadistica.redondear_publicado``: el valor
que el lector ve en el CSV.

Las tablas que ``calcular`` devuelve son las reales (sin suprimir); las cifras que alimentan el resumen y las métricas salen de la versión publicable (la misma
supresión que aplica ``ejecutar`` al escribir), de modo que lo que una tabla fuente anuló por celda pequeña no se vuelve a publicar en otro lado.

Tablas: ``matriz_asociacion.csv``, ``pares_redundantes.csv``, ``clusters_variables.csv``, ``t23_dendrograma.csv``, ``t23_vif.csv`` y ``t23_resumen_asociacion.csv``
(más las dos del panel). Figuras: F10_01 a F10_04.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy import stats
from scipy.cluster import hierarchy
from threadpoolctl import threadpool_limits

from . import catalogos, elixhauser, graficos, supresion
from . import estadistica as est
from . import panel as modulo_panel
from .contexto import Contexto
from .informes import Informe, cifra

TAREA = "T10"
N_SECCIONES = 5

AGREGADOS = ("N_COMORB_ELIX", "PUNTAJE_VAN_WALRAVEN", "PUNTAJE_VAN_WALRAVEN_SIN_5", "N_DX_SEC")
"""Agregados de comorbilidad y codificación: salen del conjunto "base" del VIF y entran al conjunto "con_agregados" (DIS 9.4)."""

OK, CONSTANTE, POCOS_DATOS, EXCLUIDA = "ok", "constante", "sin datos suficientes", "excluida"
MOTIVO_CONSTANTE = "constante en la ventana"
MOTIVO_CONSTANTE_COMPLETOS = "constante en los casos completos"
MOTIVO_FALTANTE = "faltante igual o superior al umbral de alerta: se excluye del conjunto de casos completos"
MOTIVO_POCOS_CASOS = "casos completos insuficientes: se necesitan al menos las columnas del conjunto más dos"
_MIN_PARES = 3
"""Pares válidos mínimos para medir un par de variables (DIS 9.1: "menos de 3 pares" da "sin datos suficientes")."""
_TOLERANCIA_CORTE = 1e-12
"""Holgura del corte del dendrograma: un par con distancia 0,19999999999999996 (asociación publicada 0,8) queda en el mismo grupo (NIT 3)."""

ESQUEMA_MATRIZ = {
    "variable_a": pl.String, "variable_b": pl.String, "tipo_a": pl.String, "tipo_b": pl.String, "medida": pl.String, "valor_con_signo": pl.Float64,
    "asociacion_abs": pl.Float64, "n_pares": pl.Int64, "periodo": pl.String, "estado": pl.String, "es_referencia_a": pl.String, "es_referencia_b": pl.String,
    "posicion_a": pl.Int64, "posicion_b": pl.Int64, "grupo_a": pl.Int64, "grupo_b": pl.Int64,
}
ESQUEMA_PARES = {
    "variable_a": pl.String, "variable_b": pl.String, "medida": pl.String, "valor_con_signo": pl.Float64, "asociacion_abs": pl.Float64, "n_pares": pl.Int64,
    "cluster": pl.Int64, "tamano_cluster": pl.Int64, "periodo": pl.String, "asociacion_2023": pl.Float64, "n_pares_2023": pl.Int64, "se_mantiene_2023": pl.String,
}
ESQUEMA_CLUSTERS = {
    "variable": pl.String, "tipo": pl.String, "es_referencia": pl.String, "indice_linkage": pl.Int64, "posicion_dendrograma": pl.Int64, "altura_union": pl.Float64,
    "cluster": pl.Int64, "tamano_cluster": pl.Int64, "grupo_dendrograma": pl.Int64, "n_pares_redundantes": pl.Int64, "max_asociacion_otra": pl.Float64,
    "variable_max_asociacion": pl.String,
}
ESQUEMA_DENDROGRAMA = {
    "tipo_fila": pl.String, "nodo": pl.Int64, "nodo_a": pl.Int64, "nodo_b": pl.Int64, "altura": pl.Float64, "n_hojas": pl.Int64, "variable": pl.String,
    "posicion_dendrograma": pl.Int64, "grupo_dendrograma": pl.Int64,
}
ESQUEMA_VIF = {
    "conjunto": pl.String, "variable": pl.String, "tipo": pl.String, "es_agregado": pl.String, "vif": pl.Float64, "r2": pl.Float64, "colinealidad_exacta": pl.String,
    "clasificacion": pl.String, "motivo": pl.String, "estado": pl.String, "n_filas": pl.Int64, "n_columnas_conjunto": pl.Int64, "rango_conjunto": pl.Int64,
}
COLUMNAS_MATRIZ = list(ESQUEMA_MATRIZ)
COLUMNAS_PARES = list(ESQUEMA_PARES)
COLUMNAS_CLUSTERS = list(ESQUEMA_CLUSTERS)
COLUMNAS_DENDROGRAMA = list(ESQUEMA_DENDROGRAMA)
COLUMNAS_VIF = list(ESQUEMA_VIF)
COLUMNAS_RESUMEN = [
    "periodo", "n_variables", "n_numericas", "n_binarias", "n_categoricas", "n_pares", "n_pares_no_evaluables", "asociacion_max_valor", "asociacion_max_a",
    "asociacion_max_b", "n_pares_redundantes", "n_clusters_multiples", "tamano_cluster_max", "n_variables_en_clusters_multiples",
    "n_pares_redundantes_evaluados_2023", "n_pares_redundantes_estables_2023", "n_filas_ranking", "n_filas_2023", "n_filas_vif", "n_vif_excluidas_faltante",
    "vif_max_base", "vif_max_base_variable", "vif_max_con_agregados", "vif_max_con_agregados_variable", "n_vif_alerta_base", "n_vif_critico_base", "n_exactas_base",
    "n_exactas_con_agregados", "n_columnas_base", "rango_base", "n_columnas_con_agregados", "rango_con_agregados", "n_relaciones_exactas_con_agregados",
    "identidad_n_comorb_max_abs_dif", "identidad_puntaje_max_abs_dif", "identidad_puntaje_sin5_max_abs_dif", "phi_elix_disponible", "figuras_omitidas",
]
"""Columnas de ``t23_resumen_asociacion.csv`` en el orden de DIS 9.6 (``phi_elix_disponible`` y ``figuras_omitidas`` se deciden antes de escribir)."""

# Supresión por tabla (DIS 11). Las usan ``calcular`` (versión publicable que alimenta resumen y métricas) y ``ejecutar`` (escritura): lo publicado y lo que alimenta
# el resumen no pueden divergir.
BLOQUES_MATRIZ = [(["n_pares"], ["valor_con_signo", "asociacion_abs"])]
BLOQUES_PARES = [(["n_pares"], ["valor_con_signo", "asociacion_abs"]), (["n_pares_2023"], ["asociacion_2023", "se_mantiene_2023"])]
BLOQUES_VIF = [(["n_filas"], ["r2", "vif"])]
COLS_N_RESUMEN = ["n_filas_ranking", "n_filas_2023", "n_filas_vif"]


# ============================================================================================================== columnas y medidas
@dataclass(eq=False)
class Columna:
    """Una variable del panel lista para medir, en las filas de una ventana.

    ``valores`` es float64 con NaN en una numérica, int8 con -1 para el nulo en una binaria e int32 en una categórica (el código es la posición del nivel en
    ``Panel.niveles``, y "(sin dato)" es un nivel, no un nulo). ``validas`` marca las filas con valor. ``n_niveles`` vale 0 en una numérica, 2 en una binaria y el
    número de niveles declarados en una categórica. ``rangos`` son los rangos promedio globales de una numérica sin nulos (así un par sin nulos no los recalcula).
    ``constante`` es verdadero si la variable toma un solo valor distinto en sus filas válidas.
    """

    nombre: str
    tipo: str
    valores: np.ndarray
    validas: np.ndarray
    n_niveles: int
    rangos: np.ndarray | None = None
    constante: bool = False

    @property
    def n_validas(self) -> int:
        return int(self.validas.sum())


@dataclass(frozen=True)
class ParAsociacion:
    """Resultado de medir un par: ``valor`` absoluto en [0, 1] (NaN si no existe), ``con_signo`` (NaN si no hay signo), filas válidas en ambas y ``estado``."""

    medida: str
    valor: float
    con_signo: float
    n: int
    estado: str


def crear_columna(nombre: str, tipo: str, valores: Any, n_niveles: int | None = None) -> Columna:
    """Arma una ``Columna`` desde un arreglo: calcula las filas válidas, si es constante y, en una numérica sin nulos, los rangos globales."""
    if tipo == "num":
        v = np.asarray(valores, dtype=np.float64)
        validas, n_niv = ~np.isnan(v), 0
    elif tipo == "bin":
        v = np.asarray(valores).astype(np.int8)
        if not np.isin(v, (-1, 0, 1)).all():
            raise ValueError(f"crear_columna: la binaria {nombre!r} solo admite 0, 1 y -1 (nulo)")
        validas, n_niv = v >= 0, 2
    elif tipo == "cat":
        v = np.asarray(valores).astype(np.int32)
        validas = v >= 0
        n_niv = int(n_niveles) if n_niveles is not None else (int(v.max()) + 1 if v.size else 0)
        if v.size and int(v.max()) >= n_niv:
            raise ValueError(f"crear_columna: la categórica {nombre!r} tiene códigos mayores o iguales que el número de niveles ({int(v.max())} frente a {n_niv})")
    else:
        raise ValueError(f"crear_columna: tipo desconocido {tipo!r} (debe ser 'num', 'bin' o 'cat')")
    constante = bool(validas.any() and v[validas].min() == v[validas].max())
    rangos = stats.rankdata(v) if tipo == "num" and v.size and bool(validas.all()) else None
    return Columna(nombre=nombre, tipo=tipo, valores=v, validas=validas, n_niveles=n_niv, rangos=rangos, constante=constante)


def nombres_analizados(panel: modulo_panel.Panel) -> list[str]:
    """Variables que analiza la Tarea 10, en el orden del registro: candidatas con ``evaluable`` distinto de "no" más la referencia (COD_HOSPITAL)."""
    return [
        n for n, v in panel.variables.items()
        if ((v.uso == "candidata" and v.evaluable != "no") or v.uso == "referencia") and n in panel.datos.columns
    ]


def preparar_columnas(panel: modulo_panel.Panel, anios: Sequence[int]) -> dict[str, Columna]:
    """``Columna`` de cada variable analizada en las filas de los años dados (la lista de años de la ventana, como ``Panel.ventana``; S11).

    Usa los códigos y los niveles del panel, de modo que 2023 se numera igual que la ventana de ranking. Lo reutiliza la Tarea 11 para que las medidas coincidan.
    """
    mascara = panel.datos["anio"].is_in([int(a) for a in anios]).to_numpy()
    out: dict[str, Columna] = {}
    for nombre in nombres_analizados(panel):
        v = panel.variables[nombre]
        if v.tipo == "num":
            valores, n_niv = panel.datos[nombre].cast(pl.Float64).fill_null(float("nan")).to_numpy()[mascara], None
        else:
            codigos, niveles = panel.codigos(nombre)
            valores, n_niv = codigos[mascara], len(niveles)
        out[nombre] = crear_columna(nombre, v.tipo, valores, n_niv)
    return out


_MEDIDA_POR_TIPOS = {
    ("num", "num"): "spearman", ("bin", "bin"): "phi", ("bin", "num"): "eta", ("cat", "num"): "eta", ("bin", "cat"): "v_cramer_bergsma", ("cat", "cat"): "v_cramer_bergsma",
}


def medida_de_tipos(tipo_a: str, tipo_b: str) -> str:
    """Nombre de la medida que corresponde a un par de tipos (DIS 9.1). Simétrica en los tipos."""
    return _MEDIDA_POR_TIPOS[tuple(sorted((tipo_a, tipo_b)))]  # type: ignore[index]


def _par(medida: str, valor: float, con_signo: float, n: int) -> ParAsociacion:
    """Par medido: un valor no finito es "sin datos suficientes"; si no, se acota a [0, 1] (y el signo a [-1, 1])."""
    if not np.isfinite(valor):
        return ParAsociacion(medida, float("nan"), float("nan"), n, POCOS_DATOS)
    s = float(np.clip(con_signo, -1.0, 1.0)) if np.isfinite(con_signo) else float("nan")
    return ParAsociacion(medida, float(min(1.0, max(0.0, valor))), s, n, OK)


def _sin_valor(medida: str, n: int, a: Columna, b: Columna) -> ParAsociacion | None:
    """Pares que no se miden: alguna variable constante ("constante") o menos de tres pares válidos ("sin datos suficientes"). ``None`` si se puede medir."""
    if a.constante or b.constante:
        return ParAsociacion(medida, float("nan"), float("nan"), n, CONSTANTE)
    if n < _MIN_PARES:
        return ParAsociacion(medida, float("nan"), float("nan"), n, POCOS_DATOS)
    return None


def _par_binario(a: Columna, b: Columna, phi: float) -> ParAsociacion:
    n = int((a.validas & b.validas).sum())
    return _sin_valor("phi", n, a, b) or _par("phi", abs(phi), phi, n)


def medida_par(a: Columna, b: Columna) -> ParAsociacion:
    """Medida de asociación de un par de variables en las mismas filas (DIS 9.1), con eliminación por pares.

    En numérica con numérica, si las dos no tienen nulos se reutilizan los rangos globales; si no, los rangos se recalculan en el subconjunto de filas válidas en
    ambas. Una binaria se trata como categórica de dos niveles frente a una categórica y como 0/1 con las demás, de modo que su nulo siempre se descarta. El
    resultado es simétrico en el valor, en el signo y en ``n``.
    """
    if a.valores.size != b.valores.size:
        raise ValueError(f"medida_par: {a.nombre!r} y {b.nombre!r} tienen largos distintos ({a.valores.size} y {b.valores.size})")
    if a.nombre > b.nombre:                                  # orden canónico: el resultado no depende del orden de los argumentos ni siquiera en el último bit
        a, b = b, a
    medida = medida_de_tipos(a.tipo, b.tipo)
    n = int((a.validas & b.validas).sum())
    previo = _sin_valor(medida, n, a, b)
    if previo is not None:
        return previo
    tipos = {a.tipo, b.tipo}
    if a.tipo == "num" and b.tipo == "num":
        if a.rangos is not None and b.rangos is not None:
            dx, dy = a.rangos - a.rangos.mean(), b.rangos - b.rangos.mean()
            rho = float(dx @ dy / np.sqrt((dx @ dx) * (dy @ dy)))
        else:
            rho = est.spearman(a.valores, b.valores)[0]
        return _par(medida, abs(rho), rho, n)
    if a.tipo == "bin" and b.tipo == "bin":
        phi, _ = est.phi_pares(np.column_stack([a.valores, b.valores]), np.column_stack([a.validas, b.validas]))
        return _par_binario(a, b, float(phi[0, 1]))
    if "num" in tipos:
        x, g = (a, b) if a.tipo == "num" else (b, a)
        eta2, _ = est.eta_cuadrado(x.valores, g.valores, g.n_niveles)
        if not np.isfinite(eta2):
            return _par(medida, float("nan"), float("nan"), n)
        raiz = float(np.sqrt(eta2))
        if g.tipo == "cat":
            return _par(medida, raiz, float("nan"), n)
        ok = x.validas & g.validas
        dif = x.valores[ok & (g.valores == 1)].mean() - x.valores[ok & (g.valores == 0)].mean()
        return _par(medida, raiz, float(np.sign(dif)) * raiz, n)
    tabla = est.tabla_contingencia(a.valores, b.valores, a.n_niveles, b.n_niveles)
    return _par(medida, est.v_cramer_sesgo(tabla), float("nan"), n)


def medida_diagonal(a: Columna) -> ParAsociacion:
    """Diagonal de la matriz: ``identidad``, valor 1 siempre (también si la variable es constante), ``n`` = filas válidas de la variable."""
    n = a.n_validas
    estado = CONSTANTE if a.constante else (POCOS_DATOS if n < _MIN_PARES else OK)
    return ParAsociacion("identidad", 1.0, 1.0 if a.tipo != "cat" else float("nan"), n, estado)


def con_resultado(columna: Columna, y: Any) -> tuple[float, str, int]:
    """Asociación de una variable con el desenlace ``y`` (0/1, sin nulos): ``(valor, medida, n)``.

    Una numérica usa la raíz de eta al cuadrado (``eta``, grupos = el desenlace); una binaria o categórica usa la V de Cramér corregida de Bergsma
    (``v_cramer_bergsma``), porque contra el resultado se prefiere la V corregida al phi sin corregir. El valor es NaN si la medida no existe. Lo usa la Tarea 11.
    """
    yv = np.asarray(y)
    if yv.size != columna.valores.size:
        raise ValueError(f"con_resultado: {columna.nombre!r} e 'y' tienen largos distintos ({columna.valores.size} y {yv.size})")
    if not np.isin(yv, (0, 1)).all():
        raise ValueError("con_resultado: 'y' solo admite 0 y 1")
    g = yv.astype(np.int64)
    if columna.tipo == "num":
        eta2, n = est.eta_cuadrado(columna.valores, g, 2)
        return (float(np.sqrt(eta2)) if np.isfinite(eta2) else float("nan")), "eta", n
    tabla = est.tabla_contingencia(columna.valores, g, columna.n_niveles, 2)
    return est.v_cramer_sesgo(tabla), "v_cramer_bergsma", int(tabla.sum())


def matriz_de_medidas(columnas: Mapping[str, Columna]) -> dict[str, np.ndarray]:
    """Medidas de todos los pares de ``columnas`` (en su orden): matrices k por k simétricas ``absoluto``, ``signo``, ``n``, ``estado`` y ``medida``.

    La diagonal es la identidad. Los pares binaria con binaria salen de una sola llamada vectorizada a ``phi_pares`` y coinciden con ``medida_par`` par a par.
    Cada par se mide una vez (``i < j``) y se refleja, así que la matriz es exactamente simétrica.
    """
    cols = list(columnas.values())
    k = len(cols)
    absoluto, signo = np.full((k, k), np.nan), np.full((k, k), np.nan)
    n_pares = np.zeros((k, k), dtype=np.int64)
    estado = np.full((k, k), POCOS_DATOS, dtype=object)
    medida = np.full((k, k), "", dtype=object)
    for i, c in enumerate(cols):
        p = medida_diagonal(c)
        absoluto[i, i], signo[i, i], n_pares[i, i], estado[i, i], medida[i, i] = p.valor, p.con_signo, p.n, p.estado, p.medida
    binarias = [i for i, c in enumerate(cols) if c.tipo == "bin"]
    phi, lugar = None, {i: r for r, i in enumerate(binarias)}
    if len(binarias) >= 2:
        phi, _ = est.phi_pares(np.column_stack([cols[i].valores for i in binarias]), np.column_stack([cols[i].validas for i in binarias]))
    for i in range(k):
        for j in range(i + 1, k):
            if phi is not None and cols[i].tipo == "bin" and cols[j].tipo == "bin":
                fila, columna = (lugar[i], lugar[j]) if cols[i].nombre <= cols[j].nombre else (lugar[j], lugar[i])     # la misma orientación que usa ``medida_par``
                p = _par_binario(cols[i], cols[j], float(phi[fila, columna]))
            else:
                p = medida_par(cols[i], cols[j])
            for r, s in ((i, j), (j, i)):
                absoluto[r, s], signo[r, s], n_pares[r, s], estado[r, s], medida[r, s] = p.valor, p.con_signo, p.n, p.estado, p.medida
    return {"absoluto": absoluto, "signo": signo, "n": n_pares, "estado": estado, "medida": medida}


# ======================================================================================================== clusters y dendrograma
def _renumerar(etiquetas: np.ndarray, orden: np.ndarray) -> np.ndarray:
    """Etiquetas numeradas desde 1 por la primera aparición de cada una al recorrer las hojas en el orden del dendrograma."""
    nuevo: dict[int, int] = {}
    salida = np.zeros(len(etiquetas), dtype=np.int64)
    for hoja in orden:
        e = int(etiquetas[hoja])
        salida[hoja] = nuevo.setdefault(e, len(nuevo) + 1)
    return salida


def agrupar_variables(A: Any, umbral: float) -> dict[str, Any]:
    """Orden del dendrograma, grupos del corte y clusters de redundancia (componentes conexas) de una matriz de asociación absoluta ``A`` (m por m).

    La distancia es ``1 - redondear_publicado(A)`` (el NaN pasa a 1 dentro de ``orden_por_clusters``), el enlace es promedio y los grupos son
    ``fcluster(Z, t=1 - umbral + 1e-12, criterion="distance")`` (NIT 3). Los grupos y los clusters se numeran desde 1 por la primera hoja de cada uno en el orden
    del dendrograma. Un componente puede encadenar (A~B, B~C, A no asociada a C) y un grupo no: cada grupo está contenido en un componente, pero no al revés.
    Devuelve ``orden``, ``Z``, ``posicion`` (1 a m por variable), ``grupo``, ``cluster``, ``tamano_cluster`` y ``altura_union`` (altura de la primera unión de la hoja).
    """
    A = np.asarray(A, dtype=np.float64)
    m = A.shape[0]
    if m == 0:
        vacio = np.zeros(0, dtype=np.int64)
        return {"orden": vacio, "Z": np.zeros((0, 4)), "posicion": vacio, "grupo": vacio, "cluster": vacio, "tamano_cluster": vacio, "altura_union": np.zeros(0)}
    orden, Z = est.orden_por_clusters(1.0 - est.redondear_publicado(A))
    posicion = np.empty(m, dtype=np.int64)
    posicion[orden] = np.arange(1, m + 1)
    crudo = hierarchy.fcluster(Z, t=1.0 - umbral + _TOLERANCIA_CORTE, criterion="distance") if m >= 2 else np.ones(1, dtype=np.int64)
    cluster = _renumerar(est.componentes_redundantes(A, umbral), orden)
    altura = np.full(m, np.nan)
    for fila in Z:                                           # las filas de Z van de menor a mayor altura: la primera aparición de una hoja es su primera unión
        for hijo in (int(fila[0]), int(fila[1])):
            if hijo < m and np.isnan(altura[hijo]):
                altura[hijo] = fila[2]
    return {
        "orden": orden, "Z": Z, "posicion": posicion, "grupo": _renumerar(crudo, orden), "cluster": cluster, "tamano_cluster": np.bincount(cluster)[cluster],
        "altura_union": altura,
    }


def tabla_dendrograma(nombres: Sequence[str], estructura: Mapping[str, Any]) -> pl.DataFrame:
    """``t23_dendrograma.csv`` (sin suprimir): una fila por hoja y una por unión, con las que se reconstruye ``Z`` exactamente (``[nodo_a, nodo_b, altura, n_hojas]``).

    Los nodos usan la numeración de scipy: las hojas son 0 a m-1 en el orden de ``nombres`` (la entrada del enlace) y la unión r es el nodo m + r.
    """
    m = len(nombres)
    filas: list[dict[str, Any]] = [
        {"tipo_fila": "hoja", "nodo": i, "nodo_a": None, "nodo_b": None, "altura": None, "n_hojas": 1, "variable": n,
         "posicion_dendrograma": int(estructura["posicion"][i]), "grupo_dendrograma": int(estructura["grupo"][i])}
        for i, n in enumerate(nombres)
    ]
    filas += [
        {"tipo_fila": "union", "nodo": m + r, "nodo_a": int(a), "nodo_b": int(b), "altura": float(h), "n_hojas": int(c), "variable": None, "posicion_dendrograma": None,
         "grupo_dendrograma": None}
        for r, (a, b, h, c) in enumerate(estructura["Z"])
    ]
    return pl.DataFrame(filas, schema=ESQUEMA_DENDROGRAMA)


# ================================================================================================================= VIF
def calcular_vif(columnas: Mapping[str, Columna], referencias: Sequence[str], u: Any) -> pl.DataFrame:
    """``t23_vif.csv`` (sin suprimir) en formato largo: el VIF de las numéricas y binarias en los conjuntos "base" y "con_agregados" (DIS 9.4).

    El conjunto "base" no trae los agregados (``AGREGADOS``); "con_agregados" los incluye. Ambos se calculan sobre las MISMAS filas: los casos completos de la unión.
    Se excluyen, cada una con su fila y su ``motivo``: las variables constantes en la ventana y toda variable cuya proporción de faltantes (redondeada) en la ventana
    sea igual o superior a ``faltante_alerta`` (regla general, no nombra variables), más las que quedan constantes dentro de los casos completos. Con menos de
    ``n_columnas + 2`` casos completos en un conjunto no se calcula y sus filas dicen "sin datos suficientes". La correlación de Pearson sale de una sola
    multiplicación matricial en float64 con un hilo de BLAS y ``vif_desde_correlacion``; ``clasificacion`` compara el VIF redondeado con ``vif_alerta`` y
    ``vif_critico`` con igualdad. La colinealidad exacta (VIF nulo) se informa con ``colinealidad_exacta = "si"`` y ``clasificacion = "exacta"``.
    """
    faltante_alerta, vif_alerta, vif_critico = float(u["faltante_alerta"]), float(u["vif_alerta"]), float(u["vif_critico"])
    elegibles = [n for n, c in columnas.items() if c.tipo in ("num", "bin") and n not in referencias]
    if not elegibles:
        return pl.DataFrame(schema=ESQUEMA_VIF)
    excluidas: dict[str, str] = {}
    for n in elegibles:
        c = columnas[n]
        total = c.validas.size
        faltante = 1.0 - c.n_validas / total if total else 1.0
        if c.constante:
            excluidas[n] = MOTIVO_CONSTANTE
        elif est.redondear_publicado(faltante) >= faltante_alerta:
            excluidas[n] = MOTIVO_FALTANTE
    incluidas = [n for n in elegibles if n not in excluidas]
    completa = np.ones(columnas[elegibles[0]].validas.size, dtype=bool)
    for n in incluidas:
        completa &= columnas[n].validas
    n_filas = int(completa.sum())
    correlacion, lugar = np.zeros((0, 0)), {}
    if incluidas and n_filas >= 2:
        X = np.empty((n_filas, len(incluidas)), dtype=np.float64)
        for j, n in enumerate(incluidas):
            X[:, j] = columnas[n].valores[completa]
        variables_constantes = X.min(axis=0) == X.max(axis=0)
        for j in np.flatnonzero(variables_constantes):
            excluidas[incluidas[j]] = MOTIVO_CONSTANTE_COMPLETOS
        incluidas = [n for j, n in enumerate(incluidas) if not variables_constantes[j]]
        X = X[:, ~variables_constantes]
        if incluidas:
            X -= X.mean(axis=0)
            X /= np.sqrt(np.einsum("ij,ij->j", X, X) / n_filas)
            with threadpool_limits(limits=1):
                correlacion = X.T @ X / n_filas
            correlacion = np.clip((correlacion + correlacion.T) / 2.0, -1.0, 1.0)
            np.fill_diagonal(correlacion, 1.0)
            lugar = {n: j for j, n in enumerate(incluidas)}
        del X
    filas: list[dict[str, Any]] = []
    for conjunto, miembros in (("base", [n for n in elegibles if n not in AGREGADOS]), ("con_agregados", elegibles)):
        activos = [n for n in miembros if n in incluidas]
        resultado = None
        if activos and lugar and n_filas >= len(activos) + 2:
            pos = [lugar[n] for n in activos]
            resultado = est.vif_desde_correlacion(correlacion[np.ix_(pos, pos)])
        rango = int(resultado[3]) if resultado is not None else None
        for n in miembros:
            c = columnas[n]
            fila: dict[str, Any] = {
                "conjunto": conjunto, "variable": n, "tipo": c.tipo, "es_agregado": "si" if n in AGREGADOS else "no", "vif": None, "r2": None,
                "colinealidad_exacta": "no", "clasificacion": EXCLUIDA, "motivo": excluidas.get(n, ""), "estado": EXCLUIDA, "n_filas": n_filas,
                "n_columnas_conjunto": len(activos), "rango_conjunto": rango,
            }
            if n in incluidas and resultado is None:
                fila.update(clasificacion="no evaluable", estado=POCOS_DATOS, motivo=MOTIVO_POCOS_CASOS)
            elif n in incluidas:
                vif, r2, exacta, _ = resultado
                j = activos.index(n)
                fila.update(estado=OK, r2=float(r2[j]), motivo="")
                if exacta[j]:
                    fila.update(colinealidad_exacta="si", clasificacion="exacta")
                else:
                    v = float(est.redondear_publicado(vif[j]))
                    fila.update(vif=float(vif[j]), clasificacion="critico" if v >= vif_critico else "alerta" if v >= vif_alerta else "bajo")
            filas.append(fila)
    return pl.DataFrame(filas, schema=ESQUEMA_VIF)


def identidades(panel: modulo_panel.Panel, anios: Sequence[int]) -> dict[str, float | None]:
    """Colinealidad exacta verificada con los datos (DIS 9.5): diferencia máxima absoluta de cada agregado contra su definición en las filas de la ventana.

    ``identidad_n_comorb`` compara N_COMORB_ELIX con la suma de ELIX_01 a ELIX_31; ``identidad_puntaje`` compara el puntaje de van Walraven con la combinación con
    los pesos de ``e.pesos``, e ``identidad_puntaje_sin5`` el puntaje alternativo sin las cinco categorías sospechosas. Se espera 0 exacto; si algún día no lo fuera
    (por ejemplo, un cambio de jerarquía), la diferencia lo muestra. Una columna ausente del panel da ``None``.
    """
    e = elixhauser.cargar()
    d = panel.ventana(anios)
    salida: dict[str, float | None] = {"identidad_n_comorb": None, "identidad_puntaje": None, "identidad_puntaje_sin5": None}
    if d.height == 0 or any(i not in d.columns for i in e.ids):
        return salida
    elix = d.select(list(e.ids)).to_numpy().astype(np.float64)
    pesos = np.array([e.pesos[i] for i in e.ids], dtype=np.float64)
    pesos5 = np.array([0.0 if i in e.sospechosas else e.pesos[i] for i in e.ids], dtype=np.float64)
    for clave, columna, esperado in (
        ("identidad_n_comorb", "N_COMORB_ELIX", elix.sum(axis=1)), ("identidad_puntaje", "PUNTAJE_VAN_WALRAVEN", elix @ pesos),
        ("identidad_puntaje_sin5", "PUNTAJE_VAN_WALRAVEN_SIN_5", elix @ pesos5),
    ):
        if columna in d.columns:
            salida[clave] = float(np.abs(d[columna].cast(pl.Float64).to_numpy() - esperado).max())
    return salida


# ================================================================================================================ tablas
def _serie(valores: np.ndarray, dtype: Any = pl.Float64) -> pl.Series:
    """Serie desde un arreglo aplanado: el NaN pasa a nulo (la tabla en memoria dice lo mismo que el CSV)."""
    s = pl.Series(np.asarray(valores).ravel(), dtype=dtype)
    return s.fill_nan(None) if dtype == pl.Float64 else s


def _tabla_matriz(
    panel: modulo_panel.Panel, nombres: Sequence[str], referencias: Sequence[str], med: Mapping[str, np.ndarray], posicion: Mapping[str, int], grupo: Mapping[str, int],
    periodo: str,
) -> pl.DataFrame:
    """``matriz_asociacion.csv`` (sin suprimir): formato largo y simétrico, cada par en ambos órdenes, diagonal 1, orden del registro en ``a`` y luego en ``b``."""
    k = len(nombres)
    ia, ib = np.repeat(np.arange(k), k), np.tile(np.arange(k), k)
    tipo = [panel.variables[n].tipo for n in nombres]
    es_ref = ["si" if n in referencias else "no" for n in nombres]
    pos, grp = [posicion.get(n) for n in nombres], [grupo.get(n) for n in nombres]          # nulos para la referencia
    df = pl.DataFrame(
        {
            "variable_a": [nombres[i] for i in ia], "variable_b": [nombres[j] for j in ib], "tipo_a": [tipo[i] for i in ia], "tipo_b": [tipo[j] for j in ib],
            "medida": med["medida"].ravel().tolist(), "valor_con_signo": _serie(med["signo"]), "asociacion_abs": _serie(med["absoluto"]),
            "n_pares": _serie(med["n"], pl.Int64), "periodo": [periodo] * (k * k), "estado": med["estado"].ravel().tolist(),
            "es_referencia_a": [es_ref[i] for i in ia], "es_referencia_b": [es_ref[j] for j in ib],
            "posicion_a": [pos[i] for i in ia], "posicion_b": [pos[j] for j in ib], "grupo_a": [grp[i] for i in ia], "grupo_b": [grp[j] for j in ib],
        },
        schema=ESQUEMA_MATRIZ,
    )
    return df


def _tabla_pares(
    nombres: Sequence[str], A: np.ndarray, med: Mapping[str, np.ndarray], estructura: Mapping[str, Any], cols_confirmacion: Mapping[str, Columna] | None, umbral: float,
    periodo: str,
) -> pl.DataFrame:
    """``pares_redundantes.csv`` (sin suprimir): pares ``a < b`` sin la referencia, en el orden del registro, con ``redondear_publicado(asociacion_abs) >= umbral``.

    La estabilidad de 2023 reestima cada par con la misma medida y los códigos del panel: ``se_mantiene_2023`` es "si" si el valor de 2023 redondeado alcanza el
    umbral, "no" si es menor y "no evaluable" si falta (también sin año de confirmación). ``nombres``, ``A`` y ``med`` no incluyen la referencia.
    """
    m = len(nombres)
    filas: list[dict[str, Any]] = []
    for i in range(m):
        for j in range(i + 1, m):
            if not bool(est.redondear_publicado(A[i, j]) >= umbral):                 # el NaN no es arista
                continue
            a, b = nombres[i], nombres[j]
            cluster = int(estructura["cluster"][i])
            fila: dict[str, Any] = {
                "variable_a": a, "variable_b": b, "medida": str(med["medida"][i, j]), "valor_con_signo": _nulo(med["signo"][i, j]), "asociacion_abs": float(A[i, j]),
                "n_pares": int(med["n"][i, j]), "cluster": cluster, "tamano_cluster": int(estructura["tamano_cluster"][i]), "periodo": periodo,
                "asociacion_2023": None, "n_pares_2023": None, "se_mantiene_2023": "no evaluable",
            }
            if cols_confirmacion is not None:
                p = medida_par(cols_confirmacion[a], cols_confirmacion[b])
                fila["n_pares_2023"] = p.n
                if np.isfinite(p.valor):
                    fila["asociacion_2023"] = p.valor
                    fila["se_mantiene_2023"] = "si" if bool(est.redondear_publicado(p.valor) >= umbral) else "no"
            filas.append(fila)
    return pl.DataFrame(filas, schema=ESQUEMA_PARES)


def _nulo(x: float) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def _tabla_clusters(
    panel: modulo_panel.Panel, nombres: Sequence[str], referencias: Sequence[str], estructura: Mapping[str, Any], A_nr: np.ndarray, A_pub: np.ndarray, umbral: float,
) -> pl.DataFrame:
    """``clusters_variables.csv``: una fila por variable (la referencia con el cluster, el grupo y las posiciones nulos). No decide representantes.

    ``indice_linkage`` es la posición de la variable en la entrada del enlace (0 a m-1, sin la referencia) y es el número de nodo de su hoja en ``t23_dendrograma.csv``.
    ``max_asociacion_otra`` es la mayor asociación publicable con otra variable (la referencia no cuenta como "otra" para las demás, pero la referencia sí se mide
    contra todas); ``n_pares_redundantes`` cuenta con cuántas otras variables del análisis hace una arista.
    """
    nombres_nr = [n for n in nombres if n not in referencias]
    lugar = {n: i for i, n in enumerate(nombres_nr)}
    arista = np.asarray(est.redondear_publicado(A_nr) >= umbral)
    if arista.size:
        np.fill_diagonal(arista, False)
    ref_col = np.array([n in referencias for n in nombres])
    filas = []
    for r, n in enumerate(nombres):
        fila: dict[str, Any] = {
            "variable": n, "tipo": panel.variables[n].tipo, "es_referencia": "si" if n in referencias else "no", "indice_linkage": None, "posicion_dendrograma": None,
            "altura_union": None, "cluster": None, "tamano_cluster": None, "grupo_dendrograma": None, "n_pares_redundantes": None, "max_asociacion_otra": None,
            "variable_max_asociacion": None,
        }
        otras = A_pub[r].copy()
        otras[r] = np.nan
        if n in lugar:
            i = lugar[n]
            fila.update(
                indice_linkage=i, posicion_dendrograma=int(estructura["posicion"][i]), altura_union=_nulo(estructura["altura_union"][i]), cluster=int(estructura["cluster"][i]),
                tamano_cluster=int(estructura["tamano_cluster"][i]), grupo_dendrograma=int(estructura["grupo"][i]), n_pares_redundantes=int(arista[i].sum()),
            )
            otras[ref_col] = np.nan
        if np.isfinite(otras).any():
            t = int(np.nanargmax(otras))
            fila.update(max_asociacion_otra=float(otras[t]), variable_max_asociacion=nombres[t])
        filas.append(fila)
    return pl.DataFrame(filas, schema=ESQUEMA_CLUSTERS)


def _verificar_matriz(absoluto: np.ndarray, nombres: Sequence[str]) -> None:
    """Verificación interna de la matriz: simétrica, diagonal 1 y valores en [0, 1]; si no, ``ValueError`` con las variables."""
    if not np.array_equal(absoluto, absoluto.T, equal_nan=True):
        raise ValueError("asociacion.calcular: la matriz de asociación no es simétrica")
    if absoluto.size and not np.all(np.diag(absoluto) == 1.0):
        raise ValueError("asociacion.calcular: la diagonal de la matriz de asociación no vale 1")
    fuera = ~np.isnan(absoluto) & ((absoluto < 0) | (absoluto > 1))
    if fuera.any():
        i, j = np.argwhere(fuera)[0]
        raise ValueError(f"asociacion.calcular: la asociación de {nombres[i]} con {nombres[j]} sale de [0, 1] ({absoluto[i, j]})")


_TEXTO_RESUMEN = {"periodo", "asociacion_max_a", "asociacion_max_b", "vif_max_base_variable", "vif_max_con_agregados_variable", "phi_elix_disponible", "figuras_omitidas"}
_REAL_RESUMEN = {"asociacion_max_valor", "vif_max_base", "vif_max_con_agregados", "identidad_n_comorb_max_abs_dif", "identidad_puntaje_max_abs_dif", "identidad_puntaje_sin5_max_abs_dif"}
ESQUEMA_RESUMEN = {c: pl.String if c in _TEXTO_RESUMEN else pl.Float64 if c in _REAL_RESUMEN else pl.Int64 for c in COLUMNAS_RESUMEN}


def _phi_elix_disponible(ctx: Contexto) -> str:
    """"si" si la Tarea 8 dejó ``t23_elix_phi.csv`` con filas de la ventana de ranking (sin ella F10_04 se omite)."""
    if not (ctx.dir_tablas / "t23_elix_phi.csv").is_file():
        return "no"
    t = ctx.leer_tabla("t23_elix_phi.csv", columnas_texto=["periodo", "elix_a", "elix_b"])
    return "si" if t.filter(pl.col("periodo") == modulo_panel.V_RANK).height > 0 else "no"


def _tabla_resumen(
    ctx: Contexto, panel: modulo_panel.Panel, nombres_nr: Sequence[str], A_nr: np.ndarray, A_pub_nr: np.ndarray, estructura: Mapping[str, Any], pares: pl.DataFrame,
    pares_pub: pl.DataFrame, matriz: pl.DataFrame, dendrograma: pl.DataFrame, vif: pl.DataFrame, periodo: str,
) -> pl.DataFrame:
    """``t23_resumen_asociacion.csv`` (una fila) con las cifras de los informes; las cifras estadísticas salen de las versiones publicables de las tablas fuente."""
    celda = ctx.celda_minima
    m = len(nombres_nr)
    tipos = [panel.variables[n].tipo for n in nombres_nr]
    iu = np.triu_indices(m, 1)
    r: dict[str, Any] = {c: None for c in COLUMNAS_RESUMEN}
    r.update(
        periodo=periodo, n_variables=m, n_numericas=tipos.count("num"), n_binarias=tipos.count("bin"), n_categoricas=tipos.count("cat"), n_pares=int(iu[0].size),
        n_pares_no_evaluables=int(np.isnan(A_nr[iu]).sum()), n_pares_redundantes=pares.height,
    )
    publicables = A_pub_nr[iu]
    if np.isfinite(publicables).any():
        t = int(np.nanargmax(publicables))
        r.update(asociacion_max_valor=float(publicables[t]), asociacion_max_a=nombres_nr[iu[0][t]], asociacion_max_b=nombres_nr[iu[1][t]])
    cuentas = np.bincount(estructura["cluster"], minlength=1)
    multiples = cuentas[cuentas >= 2]
    r.update(
        n_clusters_multiples=int(multiples.size), tamano_cluster_max=int(cuentas.max()) if m else 0, n_variables_en_clusters_multiples=int(multiples.sum()),
        n_pares_redundantes_evaluados_2023=pares_pub.filter(pl.col("se_mantiene_2023").is_in(["si", "no"])).height,
        n_pares_redundantes_estables_2023=pares_pub.filter(pl.col("se_mantiene_2023") == "si").height,
        n_filas_ranking=panel.ventana(panel.anios_ranking).height, n_filas_2023=panel.ventana([panel.anio_confirmacion]).height if panel.anio_confirmacion is not None else 0,
    )
    n_vif = int(vif["n_filas"][0]) if vif.height else 0
    suprimido = 1 <= n_vif < celda                                      # la tabla fuente anuló r2 y vif: lo derivado de ellos tampoco se publica
    r.update(n_filas_vif=n_vif, n_vif_excluidas_faltante=vif.filter((pl.col("conjunto") == "con_agregados") & (pl.col("motivo") == MOTIVO_FALTANTE)).height)
    for conjunto, sufijo in (("base", "base"), ("con_agregados", "con_agregados")):
        s = vif.filter(pl.col("conjunto") == conjunto)
        if not s.height:
            continue
        r.update({f"n_columnas_{sufijo}": int(s["n_columnas_conjunto"][0]), f"rango_{sufijo}": s["rango_conjunto"][0]})
        if suprimido:
            continue
        con_vif = s.filter(pl.col("vif").is_not_null())
        if con_vif.height:
            top = con_vif.sort("vif", descending=True, maintain_order=True).row(0, named=True)
            r.update({f"vif_max_{sufijo}": float(top["vif"]), f"vif_max_{sufijo}_variable": top["variable"]})
        r[f"n_exactas_{sufijo}"] = s.filter(pl.col("colinealidad_exacta") == "si").height
        if conjunto == "base":
            r.update(n_vif_alerta_base=s.filter(pl.col("clasificacion") == "alerta").height, n_vif_critico_base=s.filter(pl.col("clasificacion") == "critico").height)
    if r["rango_con_agregados"] is not None:
        r["n_relaciones_exactas_con_agregados"] = int(r["n_columnas_con_agregados"]) - int(r["rango_con_agregados"])
    ident = identidades(panel, panel.anios_ranking)
    r.update(
        identidad_n_comorb_max_abs_dif=ident["identidad_n_comorb"], identidad_puntaje_max_abs_dif=ident["identidad_puntaje"],
        identidad_puntaje_sin5_max_abs_dif=ident["identidad_puntaje_sin5"], phi_elix_disponible=_phi_elix_disponible(ctx),
    )
    omitidas = [
        figura for figura, vacia in (("F10_01", matriz.height == 0), ("F10_02", dendrograma.height == 0), ("F10_03", vif.height == 0), ("F10_04", r["phi_elix_disponible"] == "no"))
        if vacia
    ]
    r["figuras_omitidas"] = ";".join(omitidas) if omitidas else "ninguna"
    return pl.DataFrame([r], schema=ESQUEMA_RESUMEN)


# ================================================================================================================ calcular
def calcular(ctx: Contexto, panel: modulo_panel.Panel | None = None, cat: catalogos.Catalogos | None = None) -> dict[str, Any]:
    """Calcula todo lo de la Tarea 10 sin escribir nada. Si ``panel`` es ``None`` lo construye con ``panel.construir_panel(ctx, cat)``; con un panel no lee datos.

    Devuelve ``panel``, ``matriz`` (formato largo), ``A`` (matriz simétrica de valores absolutos en el orden del registro, con la referencia y la diagonal 1), ``nombres``
    (las variables de ``A``), ``pares``, ``clusters``, ``dendrograma``, ``Z`` (matriz de enlace de scipy), ``componentes`` (cluster de cada variable sin la referencia),
    ``vif`` y ``resumen``. Las tablas son las reales (sin suprimir); el resumen sale de las versiones publicables. La Tarea 11 recibe este diccionario para no recalcular.
    """
    if panel is None:
        panel = modulo_panel.construir_panel(ctx, cat)
    u = ctx.umbrales
    umbral, celda, periodo = float(u["redundancia_abs"]), ctx.celda_minima, modulo_panel.V_RANK
    nombres = nombres_analizados(panel)
    referencias = [n for n in nombres if panel.variables[n].es_referencia]
    columnas = preparar_columnas(panel, panel.anios_ranking)
    med = matriz_de_medidas(columnas)
    A = med["absoluto"]
    _verificar_matriz(A, nombres)
    idx = [i for i, n in enumerate(nombres) if n not in referencias]
    nombres_nr = [nombres[i] for i in idx]
    ix = np.ix_(idx, idx)
    A_nr = A[ix]
    estructura = agrupar_variables(A_nr, umbral)
    posicion = dict(zip(nombres_nr, estructura["posicion"].tolist()))
    grupo = dict(zip(nombres_nr, estructura["grupo"].tolist()))
    A_pub = np.where((med["n"] >= 1) & (med["n"] < celda), np.nan, A)

    matriz = _tabla_matriz(panel, nombres, referencias, med, posicion, grupo, periodo)
    cols_conf = preparar_columnas(panel, [panel.anio_confirmacion]) if panel.anio_confirmacion is not None else None
    pares = _tabla_pares(nombres_nr, A_nr, {k: v[ix] for k, v in med.items()}, estructura, cols_conf, umbral, periodo)
    pares_pub = supresion.suprimir_bloques(pares, BLOQUES_PARES, minimo=celda)[0]
    clusters = _tabla_clusters(panel, nombres, referencias, estructura, A_nr, A_pub, umbral)
    dendrograma = tabla_dendrograma(nombres_nr, estructura)
    vif = calcular_vif(columnas, referencias, u)
    resumen = _tabla_resumen(ctx, panel, nombres_nr, A_nr, A_pub[ix], estructura, pares, pares_pub, matriz, dendrograma, vif, periodo)
    return {
        "panel": panel, "matriz": matriz, "A": A, "nombres": nombres, "pares": pares, "clusters": clusters, "dendrograma": dendrograma, "Z": estructura["Z"],
        "componentes": dict(zip(nombres_nr, estructura["cluster"].tolist())), "vif": vif, "resumen": resumen,
    }


# =============================================================================================================== figuras
_PALETA_GRUPOS = [graficos.OKABE_ITO[i] for i in (0, 1, 2, 3, 4, 5)]
"""Azul, naranja, verde, bermellón, celeste y púrpura: Okabe-Ito sin el amarillo (poco contraste sobre blanco) ni el negro (reservado a texto y cortes)."""
_GRIS_ENLACE = "#7F7F7F"
_COLOR_RECUADRO = graficos.OKABE_ITO[3]
_COLOR_CORTE = "#333333"


def _bloques_contiguos(etiquetas: Sequence[Any]) -> list[tuple[int, int]]:
    """Bloques ``(inicio, tamano)`` de más de un elemento donde la misma etiqueta se repite en posiciones contiguas (la etiqueta nula no forma bloque)."""
    bloques, i, n = [], 0, len(etiquetas)
    while i < n:
        j = i
        while etiquetas[i] is not None and j + 1 < n and etiquetas[j + 1] == etiquetas[i]:
            j += 1
        if j > i:
            bloques.append((i, j - i + 1))
        i = j + 1
    return bloques


def _texto_decimal(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def figura_mapa(ctx: Contexto) -> bool:
    """F10_01: mapa de calor de la matriz de asociación agrupada por similitud. Se arma leyendo SOLO ``matriz_asociacion.csv``.

    Las variables van en el orden del dendrograma (``posicion_a``) con la referencia al final, separada con una línea gruesa; los grupos del dendrograma con más de un
    miembro (``grupo_a``) se enmarcan. Las celdas sin valor (constantes, datos insuficientes o celdas suprimidas) se ven en gris. Devuelve ``False`` si la tabla está vacía.
    """
    t = ctx.leer_tabla(
        "matriz_asociacion.csv", columnas_texto=["variable_a", "variable_b", "tipo_a", "tipo_b", "medida", "periodo", "estado", "es_referencia_a", "es_referencia_b"]
    )
    if t.height == 0:
        return False
    filas = t.select("variable_a", "posicion_a", "grupo_a", "es_referencia_a").unique(subset=["variable_a"], maintain_order=True).rows()
    normales = sorted((f for f in filas if f[3] != "si"), key=lambda f: (f[1] if f[1] is not None else 10**9))
    referencias = [f for f in filas if f[3] == "si"]
    orden = [f[0] for f in normales + referencias]
    matriz, _, _ = graficos.tabla_a_matriz(t, "variable_a", "variable_b", "asociacion_abs", orden_filas=orden, orden_columnas=orden)
    fig, ax = graficos.nueva_figura(figsize=(12.0, 11.5))
    barra = graficos.mapa_calor(
        fig, ax, matriz, orden, orden, cmap=graficos.CMAP_SECUENCIAL, vmin=0.0, vmax=1.0, etiqueta_barra="Asociación absoluta", anotar=False, rotar_x=90,
    )
    ax.tick_params(axis="both", labelsize=6.5, length=2)
    for etiqueta in ax.get_xticklabels():
        etiqueta.set_ha("center")                                              # con 90 grados el alineamiento "right" de mapa_calor desplaza la etiqueta
    barra.ax.tick_params(labelsize=7)
    graficos.formato_barra_decimal(barra)
    graficos.recuadros_de_grupo(ax, _bloques_contiguos([f[2] for f in normales]), color=_COLOR_RECUADRO, ancho=1.2)
    if referencias and normales:
        graficos.linea_gruesa(ax, len(normales))
    ax.set_xlabel("Variable (ordenadas por similitud)")
    ax.set_ylabel("Variable (ordenadas por similitud)")
    ctx.figuras.guardar_figura(
        fig, id_figura="F10_01", tarea=10, titulo="Asociación entre variables, agrupada por similitud (2019 a 2022)",
        descripcion_alt="Mapa de calor cuadrado con la asociación absoluta, entre cero y uno, de cada par de variables del panel de adultos hospitalizados, ordenadas por un "
                        "agrupamiento jerárquico; los recuadros marcan los grupos de variables redundantes y la última fila y columna son el hospital de egreso, la referencia.",
        csv="matriz_asociacion.csv", anios="2019 a 2022", decision="", uso_interno=False,
        extra_pie="Cada tipo de par usa su medida: Spearman, phi, razón de correlación o V de Cramér corregida. Celdas grises: sin valor (variable constante, datos insuficientes "
                  "o suprimido). Recuadros: grupos que quedan juntos al corte de redundancia. Última fila y columna: el hospital (referencia).",
    )
    return True


def figura_dendrograma(ctx: Contexto) -> bool:
    """F10_02: dendrograma de redundancia reconstruido solo desde ``t23_dendrograma.csv`` (hojas y uniones).

    Los subárboles que quedan bajo el corte de redundancia (``1 - redundancia_abs``) se pintan con colores Okabe-Ito, uno por grupo, y lo que está sobre el corte, en gris.
    Un nodo está bajo el corte si todas sus hojas tienen el mismo ``grupo_dendrograma`` (así el dibujo coincide con la tabla aunque el redondeo mueva una altura).
    La primera hoja del orden del dendrograma queda arriba. Devuelve ``False`` si la tabla no tiene hojas.
    """
    t = ctx.leer_tabla("t23_dendrograma.csv", columnas_texto=["tipo_fila", "variable"])
    hojas = t.filter(pl.col("tipo_fila") == "hoja").sort("nodo")
    if hojas.height == 0:
        return False
    uniones = t.filter(pl.col("tipo_fila") == "union").sort("nodo")
    m = hojas.height
    corte = 1.0 - float(ctx.umbrales["redundancia_abs"])
    etiquetas = hojas["variable"].to_list()
    fig, ax = graficos.nueva_figura(figsize=(9.0, graficos.altura_filas(m)))
    if uniones.height:
        Z = np.column_stack([uniones["nodo_a"].to_numpy(), uniones["nodo_b"].to_numpy(), uniones["altura"].to_numpy(), uniones["n_hojas"].to_numpy()]).astype(float)
        conjunto = {i: {g} for i, g in enumerate(hojas["grupo_dendrograma"].to_list())}
        color: dict[int, str] = {}
        for r, (a, b) in enumerate(zip(Z[:, 0], Z[:, 1])):
            conjunto[m + r] = conjunto[int(a)] | conjunto[int(b)]
            color[m + r] = _PALETA_GRUPOS[(next(iter(conjunto[m + r])) - 1) % len(_PALETA_GRUPOS)] if len(conjunto[m + r]) == 1 else _GRIS_ENLACE
        hierarchy.dendrogram(Z, orientation="right", labels=etiquetas, ax=ax, link_color_func=lambda k: color[k], leaf_font_size=6.5)
        ax.invert_yaxis()                                                          # la primera hoja del orden del dendrograma (posición 1) queda arriba
    else:
        ax.set_yticks([0])
        ax.set_yticklabels(etiquetas)
        ax.set_ylim(-0.5, 0.5)
    ax.axvline(corte, color=_COLOR_CORTE, linestyle="--", linewidth=1.0)
    ax.text(corte, 1.0, " corte de redundancia", transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=7, color=_COLOR_CORTE)
    ax.set_xlim(left=0.0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=6.5, length=2)
    graficos.formato_decimal(ax, "x")
    ax.set_xlabel("Distancia (1 menos la asociación absoluta)")
    ctx.figuras.guardar_figura(
        fig, id_figura="F10_02", tarea=10, titulo="Dendrograma de redundancia entre variables (2019 a 2022)",
        descripcion_alt="Dendrograma horizontal de las variables del panel con enlace promedio sobre la distancia de uno menos la asociación absoluta; una línea punteada marca el "
                        "corte de redundancia y los grupos que se unen antes del corte tienen un color propio.",
        csv="t23_dendrograma.csv", anios="2019 a 2022", decision="", uso_interno=False,
        extra_pie="Cada color es un grupo que se une antes del corte (las uniones de asociación casi perfecta quedan pegadas al eje); en gris, lo que se une después. "
                  "El hospital (referencia) no entra al dendrograma.",
    )
    return True


def figura_vif(ctx: Contexto) -> bool:
    """F10_03: VIF de las numéricas y binarias en dos paneles (conjunto base y con agregados), solo desde ``t23_vif.csv``.

    Barras horizontales ordenadas, escala logarítmica, líneas en ``vif_alerta`` y ``vif_critico`` y leyenda fuera de los datos. La colinealidad exacta (VIF nulo) se dibuja
    hasta el borde del eje con trama diagonal y el rótulo "exacta", nunca como un número. Devuelve ``False`` si la tabla está vacía.
    """
    t = ctx.leer_tabla("t23_vif.csv", columnas_texto=["conjunto", "variable", "tipo", "es_agregado", "colinealidad_exacta", "clasificacion", "motivo", "estado"])
    if t.height == 0:
        return False
    alerta, critico = float(ctx.umbrales["vif_alerta"]), float(ctx.umbrales["vif_critico"])
    paneles = [("base", "Conjunto base"), ("con_agregados", "Con agregados de comorbilidad")]
    # Solo se dibuja lo que tiene valor o es colinealidad exacta: un VIF nulo con ``colinealidad_exacta = "no"`` es una celda suprimida (n_filas entre 1 y 9) y no se dibuja
    # (si se tomara como VIF nulo a secas, se mostraría como "exacta").
    dibujables = {
        c: t.filter((pl.col("conjunto") == c) & (pl.col("estado") == "ok") & (pl.col("vif").is_not_null() | (pl.col("colinealidad_exacta") == "si"))) for c, _ in paneles
    }
    suprimidos = {c: t.filter((pl.col("conjunto") == c) & (pl.col("estado") == "ok")).height > 0 for c, _ in paneles}
    finitos = [v for d in dibujables.values() for v in d["vif"].drop_nulls().to_list()]
    limite = max([2.0 * critico, *[1.6 * v for v in finitos]])
    # Dos filas: los paneles y una banda angosta para la leyenda, fuera de los datos y sin chocar con el pie que agrega ``guardar_figura``.
    fig, ejes = graficos.nueva_figura_multiple(
        2, 2, figsize=(13.0, graficos.altura_filas(max([d.height for d in dibujables.values()] + [1])) + 0.5), gridspec_kw={"height_ratios": [1.0, 0.035]},
    )
    hueco_leyenda = ejes[1][0]
    for hueco in ejes[1]:
        hueco.axis("off")
    colores = {"bajo": graficos.OKABE_ITO[0], "alerta": graficos.OKABE_ITO[1], "critico": graficos.OKABE_ITO[3]}
    presentes: set[str] = set()
    for ax, (conjunto, titulo) in zip(ejes[0], paneles):
        d = dibujables[conjunto]
        ax.set_xscale("log")
        ax.set_xlim(1.0, limite)
        if d.height == 0:
            ax.text(
                0.5, 0.5, "n/d: celda suprimida (menos de diez casos)" if suprimidos[conjunto] else "sin datos suficientes", transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#555555",
            )
            ax.set_yticks([])
        else:
            ordenado = pl.concat([d.filter(pl.col("vif").is_not_null()).sort("vif", maintain_order=True), d.filter(pl.col("vif").is_null())])      # los exactos arriba
            for y, fila in enumerate(ordenado.iter_rows(named=True)):
                if fila["vif"] is None:
                    ax.barh(y, limite, color="#D9D9D9", edgecolor="#707070", hatch="//", linewidth=0.0)
                    ax.text(1.1, y, "exacta", va="center", ha="left", fontsize=6.5, color="black", bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"})
                    presentes.add("exacta")
                else:
                    ax.barh(y, fila["vif"], color=colores[fila["clasificacion"]])
                    ax.text(fila["vif"] * 1.06, y, _texto_decimal(fila["vif"]), va="center", ha="left", fontsize=6.5)
                    presentes.add(fila["clasificacion"])
            ax.set_yticks(range(ordenado.height))
            ax.set_yticklabels(ordenado["variable"].to_list(), fontsize=6.5)
            ax.set_ylim(-0.7, ordenado.height - 0.3)
        ax.axvline(alerta, color="#555555", linestyle="--", linewidth=0.9)
        ax.axvline(critico, color="#000000", linestyle=":", linewidth=1.1)
        ticks = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10000) if v <= limite]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(titulo, fontsize=9)
        ax.set_xlabel("VIF (escala logarítmica)")
        ax.grid(axis="y", visible=False)
    etiquetas = {"bajo": "VIF bajo", "alerta": "VIF de alerta", "critico": "VIF crítico", "exacta": "colinealidad exacta"}
    handles = [
        Patch(facecolor="#D9D9D9" if c == "exacta" else colores[c], edgecolor="#707070" if c == "exacta" else "none", hatch="//" if c == "exacta" else None, label=etiquetas[c])
        for c in ("bajo", "alerta", "critico", "exacta") if c in presentes
    ]
    handles += [Line2D([], [], color="#555555", linestyle="--", label="umbral de alerta"), Line2D([], [], color="#000000", linestyle=":", label="umbral crítico")]
    leyenda = hueco_leyenda.legend(handles=handles, loc="center", bbox_to_anchor=(1.0, 0.5), ncol=len(handles), fontsize=7, frameon=False)       # centrada entre los dos paneles
    leyenda.set_in_layout(False)
    ctx.figuras.guardar_figura(
        fig, id_figura="F10_03", tarea=10, titulo="VIF de las variables numéricas y binarias (2019 a 2022)",
        descripcion_alt="Dos paneles de barras horizontales con el factor de inflación de la varianza de cada variable numérica o binaria en escala logarítmica, sin y con los "
                        "agregados de comorbilidad; la colinealidad exacta se muestra como una barra con trama hasta el borde y el rótulo exacta.",
        csv="t23_vif.csv", anios="2019 a 2022", decision="", uso_interno=False,
        extra_pie="Casos completos de la unión de ambos conjuntos. Las variables constantes o con faltante igual o superior al umbral de alerta quedan fuera (ver la tabla).",
    )
    return True


def figura_phi_elix(ctx: Contexto) -> bool:
    """F10_04: matriz phi de las 31 categorías de Elixhauser ordenada por clusters, reconstruida solo desde ``t23_elix_phi.csv`` (Tarea 8).

    Lee el período de la ventana de ranking, ordena con ``orden_por_clusters(1 - abs(phi))`` calculado sobre esa misma tabla y enmarca los grupos de más de un miembro al
    corte ``1 - redundancia_abs``. El rango de color es simétrico: el máximo fuera de la diagonal más 0,049, redondeado a un decimal (como ``comorbilidad.figura_phi``).
    Devuelve ``False`` (sin error) si la tabla no existe o no tiene filas de la ventana de ranking: la Tarea 8 no se corrió.
    """
    if not (ctx.dir_tablas / "t23_elix_phi.csv").is_file():
        return False
    t = ctx.leer_tabla("t23_elix_phi.csv", columnas_texto=["periodo", "elix_a", "elix_b", "nombre_a", "nombre_b"])
    s = t.filter(pl.col("periodo") == modulo_panel.V_RANK)
    if s.height == 0:
        return False
    ids = sorted(s["elix_a"].unique().to_list())
    nombres = dict(zip(s["elix_a"], s["nombre_a"]))
    phi, _, _ = graficos.tabla_a_matriz(s, "elix_a", "elix_b", "phi", orden_filas=ids, orden_columnas=ids)
    orden, Z = est.orden_por_clusters(1.0 - np.abs(phi))
    bloques: list[tuple[int, int]] = []
    if len(ids) >= 2:
        grupos = hierarchy.fcluster(Z, t=1.0 - float(ctx.umbrales["redundancia_abs"]) + _TOLERANCIA_CORTE, criterion="distance")
        bloques = _bloques_contiguos([int(grupos[i]) for i in orden])
    ordenados = [ids[i] for i in orden]
    m = phi[np.ix_(orden, orden)]
    fuera = np.abs(m[~np.eye(len(ids), dtype=bool)])
    tope = max(0.1, round(float(np.nanmax(fuera)) + 0.049, 1)) if np.isfinite(fuera).any() else 0.5
    etiquetas = [f"{i[-2:]} {nombres.get(i, i)}" for i in ordenados]
    fig, ax = graficos.nueva_figura(figsize=(10.5, 9.5))
    barra = graficos.mapa_calor(
        fig, ax, m, etiquetas, etiquetas, cmap=graficos.CMAP_DIVERGENTE, vmin=-tope, vmax=tope, etiqueta_barra="Coeficiente phi", anotar=False, rotar_x=90,
    )
    ax.tick_params(axis="both", labelsize=6.5, length=2)
    for etiqueta in ax.get_xticklabels():
        etiqueta.set_ha("center")
    barra.ax.tick_params(labelsize=7)
    graficos.formato_barra_decimal(barra)
    graficos.recuadros_de_grupo(ax, bloques, color="black", ancho=1.2)
    ctx.figuras.guardar_figura(
        fig, id_figura="F10_04", tarea=10, titulo="Matriz phi de las categorías de Elixhauser ordenada por clusters (2019 a 2022)",
        descripcion_alt="Mapa de calor cuadrado con el coeficiente phi entre las categorías de Elixhauser, ordenadas por un agrupamiento jerárquico; los recuadros marcan los "
                        "grupos de categorías redundantes al corte de redundancia y la diagonal vale uno.",
        csv="t23_elix_phi.csv", anios="2019 a 2022", decision="8",
        extra_pie="La base de la Tarea 8 (adultos hospitalizados) difiere ligeramente de la del panel de esta tarea. El rango de color es simétrico y excluye la diagonal.",
    )
    return True


# ====================================================================================================================== ejecutar
def _escribir(ctx: Contexto, r: Mapping[str, Any]) -> None:
    cols, deps = BLOQUES_MATRIZ[0]
    ctx.escribir_tabla(
        "matriz_asociacion.csv", r["matriz"], tarea=TAREA, cols_n=cols, dependientes=deps,
        descripcion="Asociación entre cada par de variables del panel en 2019 a 2022 (formato largo y simétrico, diagonal 1, con la posición y el grupo del dendrograma)",
    )
    modulo_panel.escribir_por_bloques(
        ctx, "pares_redundantes.csv", r["pares"], BLOQUES_PARES, tarea=TAREA,
        descripcion="Pares de variables con asociación igual o superior al umbral de redundancia, con su cluster y la reestimación de 2023 en columnas aparte",
    )
    ctx.escribir_tabla("clusters_variables.csv", r["clusters"], tarea=TAREA, descripcion="Cluster de redundancia, grupo del dendrograma y vecino más asociado de cada variable (sin representante)")
    ctx.escribir_tabla("t23_dendrograma.csv", r["dendrograma"], tarea=TAREA, descripcion="Hojas y uniones del dendrograma de redundancia (con ellas se reconstruye la matriz de enlace)")
    cols, deps = BLOQUES_VIF[0]
    ctx.escribir_tabla(
        "t23_vif.csv", r["vif"], tarea=TAREA, cols_n=cols, dependientes=deps,
        descripcion="Factor de inflación de la varianza de las variables numéricas y binarias, sin y con los agregados de comorbilidad (colinealidad exacta con VIF nulo)",
    )
    ctx.escribir_tabla("t23_resumen_asociacion.csv", r["resumen"], tarea=TAREA, cols_n=COLS_N_RESUMEN, descripcion="Cifras de la Tarea 10 para los informes (una fila)")


def registrar_metricas(ctx: Contexto, r: Mapping[str, Any]) -> None:
    """Registra las métricas de la Tarea 10 desde el resumen (que sale de las versiones publicables). Ninguna es un conteo de casos y ninguna cubre 2024."""
    res = r["resumen"].row(0, named=True)
    m = ctx.registrar_metrica
    base = {"tarea": TAREA, "conteo": False, "alcance": modulo_panel.V_RANK}
    m("asociacion.n_variables", res["n_variables"], tabla="t23_resumen_asociacion.csv", **base)
    m("asociacion.n_pares_redundantes", res["n_pares_redundantes"], tabla="pares_redundantes.csv", **base)
    m("asociacion.n_clusters_multiples", res["n_clusters_multiples"], tabla="clusters_variables.csv", **base)
    par = f"{res['asociacion_max_a']} con {res['asociacion_max_b']}" if res["asociacion_max_a"] else ""
    m("asociacion.max_abs", res["asociacion_max_valor"], tabla="matriz_asociacion.csv", nota=par, **base)
    hay = r["panel"].anio_confirmacion is not None
    m(
        "asociacion.n_pares_redundantes_estables_2023", res["n_pares_redundantes_estables_2023"] if hay else None, tabla="pares_redundantes.csv", tarea=TAREA, conteo=False,
        alcance=modulo_panel.V_CONF, nota="" if hay else "confirmación no disponible",
    )
    for clave, columna in (
        ("vif.max_base", "vif_max_base"), ("vif.max_con_agregados", "vif_max_con_agregados"), ("vif.n_exactas_base", "n_exactas_base"),
        ("vif.n_exactas_con_agregados", "n_exactas_con_agregados"), ("vif.rango_base", "rango_base"), ("vif.rango_con_agregados", "rango_con_agregados"),
    ):
        m(clave, res[columna], tabla="t23_vif.csv", **base)


def ejecutar(ctx: Contexto, panel: modulo_panel.Panel | None = None, cat: catalogos.Catalogos | None = None) -> dict[str, Any]:
    """Calcula, escribe las seis tablas de la Tarea 10 (y las dos del panel), registra las métricas, dibuja F10_01 a F10_04 y devuelve el mismo diccionario que ``calcular``.

    Una figura se omite solo si su tabla está vacía (F10_04 también si falta ``t23_elix_phi.csv``): sin error y anotado en ``figuras_omitidas`` del resumen.
    """
    r = calcular(ctx, panel, cat)
    modulo_panel.escribir_tablas(ctx, r["panel"], tarea=TAREA)
    _escribir(ctx, r)
    registrar_metricas(ctx, r)
    omitidas = set(str(r["resumen"]["figuras_omitidas"][0]).split(";"))
    for id_figura, dibujar in (("F10_01", figura_mapa), ("F10_02", figura_dendrograma), ("F10_03", figura_vif), ("F10_04", figura_phi_elix)):
        if id_figura not in omitidas:
            dibujar(ctx)
    return r


# ====================================================================================================================== informe
def agregar_secciones(inf: Informe, ctx: Contexto, n0: int = 1) -> None:
    """Agrega al informe de exploración (CRISP-DM 2.3) las cinco secciones de la Tarea 10, con las convenciones de ``hospitales.agregar_secciones``.

    Las cifras salen de ``t23_resumen_asociacion.csv`` (``cifra``), los umbrales de ``descripcion.Cifra_umbral``, las tablas se incrustan con ``inf.tabla(df=...)`` y las
    figuras con ``inf.figura`` solo si la figura se generó en esta corrida (nunca una figura omitida). La prosa son plantillas con marcadores: sin dígitos sueltos.
    Las frases que mencionan el extracto solo se agregan si la etiqueta del contexto es "extracto".
    """
    from .descripcion import Cifra_umbral

    textos = ["periodo", "asociacion_max_a", "asociacion_max_b", "vif_max_base_variable", "vif_max_con_agregados_variable", "phi_elix_disponible", "figuras_omitidas"]
    rs = lambda col: cifra(ctx, "t23_resumen_asociacion.csv", col, columnas_texto=textos)  # noqa: E731
    res = ctx.leer_tabla("t23_resumen_asociacion.csv", columnas_texto=textos).row(0, named=True)
    figuras = set(ctx.figuras.ids())
    clusters = ctx.leer_tabla("clusters_variables.csv", columnas_texto=["variable", "tipo", "es_referencia", "variable_max_asociacion"])

    # --- 1. qué se midió -----------------------------------------------------------------------------------
    inf.seccion(f"{n0}. Qué se midió y con qué medidas")
    _tope_panel = int(ctx.umbrales["max_filas_asociacion_por_anio"])
    _cifras_qm = dict(
        n_var=rs("n_variables"), n_num=rs("n_numericas"), n_bin=rs("n_binarias"), n_cat=rs("n_categoricas"), periodo=rs("periodo"), n_rank=rs("n_filas_ranking"),
        n_conf=rs("n_filas_2023"),
    )
    if _tope_panel > 0:
        _cifras_qm["tope"] = Cifra_umbral(ctx, "max_filas_asociacion_por_anio")
    inf.parrafo(
        "Se analizan {n_var} variables del panel de adultos hospitalizados con desenlace conocido: {n_num} numéricas, {n_bin} binarias y {n_cat} categóricas. Se agrega el hospital de "
        "egreso como referencia: se mide, pero no entra a los grupos de redundancia ni al VIF. Cada tipo de par usa su propia medida y todas van de cero a uno: el valor absoluto de "
        "rho de Spearman entre dos numéricas, el valor absoluto de phi entre dos binarias, la raíz de eta al cuadrado entre una numérica y una binaria o una categórica, y la V de "
        "Cramér corregida por sesgo (Bergsma) cuando interviene una categórica y no una numérica. Los faltantes se eliminan por pares, sin imputar, y el nivel sin dato de una "
        "categórica cuenta como un nivel más. La ventana de ranking ({periodo|texto}) reúne {n_rank} egresos y la de confirmación (2023), {n_conf}. "
        + ("El panel toma como máximo {tope|entero} egresos por año, con un muestreo reproducible." if _tope_panel > 0 else "El panel usa todos los egresos de la cohorte de cada año, sin muestreo."),
        **_cifras_qm,
    )
    por_tipo = [(etiqueta, clusters.filter((pl.col("tipo") == t) & (pl.col("es_referencia") == "no")).height) for t, etiqueta in (("num", "Numéricas"), ("bin", "Binarias"), ("cat", "Categóricas"))]
    por_tipo.append(("Referencia (hospital de egreso)", clusters.filter(pl.col("es_referencia") == "si").height))
    inf.tabla(
        "clusters_variables.csv", df=pl.DataFrame({"tipo_variable": [f[0] for f in por_tipo], "n_variables": [f[1] for f in por_tipo]}), columnas=["tipo_variable", "n_variables"],
        encabezados={"tipo_variable": "Tipo de variable", "n_variables": "Variables"}, leyenda="Variables del análisis por tipo",
    )

    # --- 2. la matriz ----------------------------------------------------------------------------------------
    inf.seccion(f"{n0 + 1}. La matriz de asociación y su lectura")
    inf.parrafo(
        "La figura F10_01 ordena las variables por similitud con un agrupamiento jerárquico de enlace promedio sobre la distancia de uno menos la asociación absoluta, de modo que las "
        "variables parecidas quedan juntas. Los recuadros marcan los grupos que se mantienen unidos al corte de redundancia ({umbral|dec2}) y la última fila y columna son la "
        "referencia. La asociación más alta entre dos variables distintas es {maximo|dec3}, entre {a|texto} y {b|texto}. De {n_pares} pares, {n_red} alcanzan el umbral de redundancia; "
        "forman {n_cl} clusters de variables encadenadas por pares redundantes, que reúnen {n_en_cl} variables, y el mayor tiene {tam_max} variables. Las celdas grises no tienen valor: "
        "la variable es constante, no hay datos suficientes o la celda se suprimió.",
        umbral=Cifra_umbral(ctx, "redundancia_abs"), maximo=rs("asociacion_max_valor"), a=rs("asociacion_max_a"), b=rs("asociacion_max_b"), n_pares=rs("n_pares"),
        n_red=rs("n_pares_redundantes"), n_cl=rs("n_clusters_multiples"), n_en_cl=rs("n_variables_en_clusters_multiples"), tam_max=rs("tamano_cluster_max"),
    )
    if "F10_01" in figuras:
        inf.figura("F10_01")

    # --- 3. pares y clusters -----------------------------------------------------------------------------------
    inf.seccion(f"{n0 + 2}. Pares y clusters redundantes")
    inf.parrafo(
        "Un cluster de redundancia es una componente conexa: reúne las variables enlazadas por pares con asociación igual o superior al umbral, aunque dos de ellas no estén asociadas "
        "entre sí (A con B y B con C bastan). El grupo del dendrograma es lo que queda junto al cortar el árbol (F10_02): cada grupo está contenido en un cluster, pero un cluster "
        "puede partirse en varios grupos. La selección de variables usa los clusters y la figura muestra los grupos. De los {n_eval} pares redundantes con medición en 2023, "
        "{n_est} mantienen una asociación igual o superior al umbral.",
        n_eval=rs("n_pares_redundantes_evaluados_2023"), n_est=rs("n_pares_redundantes_estables_2023"),
    )
    if "F10_02" in figuras:
        inf.figura("F10_02")
    if "F10_04" in figuras:
        inf.parrafo(
            "La figura F10_04 repite el ordenamiento por clusters con las categorías de Elixhauser, a partir de la matriz phi de la Tarea 8. Esa matriz se calcula sobre adultos "
            "hospitalizados y no sobre la cohorte del panel, de modo que las dos bases difieren ligeramente."
        )
        inf.figura("F10_04")
    elif res.get("phi_elix_disponible") == "no":
        inf.parrafo("La figura F10_04 no se generó porque la Tarea 8 no dejó la matriz phi de las categorías de Elixhauser en esta corrida.")
    pares = ctx.leer_tabla("pares_redundantes.csv", columnas_texto=["variable_a", "variable_b", "medida", "periodo", "se_mantiene_2023"])
    inf.tabla(
        "pares_redundantes.csv", df=pares.sort("asociacion_abs", descending=True, nulls_last=True, maintain_order=True).head(15),
        columnas=["variable_a", "variable_b", "medida", "asociacion_abs", "asociacion_2023", "se_mantiene_2023", "cluster"],
        encabezados={"variable_a": "Variable A", "variable_b": "Variable B", "medida": "Medida", "asociacion_abs": "Asociación 2019-2022", "asociacion_2023": "Asociación 2023",
                     "se_mantiene_2023": "Se mantiene en 2023", "cluster": "Cluster"},
        formatos={"asociacion_abs": "dec2", "asociacion_2023": "dec2"}, leyenda="Los quince pares más asociados, con su valor de 2023 y si se mantiene",
    )
    multiples = clusters.filter(pl.col("tamano_cluster") > 1).sort(["cluster", "posicion_dendrograma"], maintain_order=True)
    if multiples.height:
        tabla_cl = multiples.group_by("cluster", maintain_order=True).agg(pl.len().alias("n_variables"), pl.col("variable").str.join(", ").alias("variables"))
        inf.tabla(
            "clusters_variables.csv", df=tabla_cl, columnas=["cluster", "n_variables", "variables"], encabezados={"cluster": "Cluster", "n_variables": "Variables", "variables": "Miembros"},
            leyenda="Clusters de redundancia con más de una variable",
        )

    # --- 4. VIF ------------------------------------------------------------------------------------------------
    inf.seccion(f"{n0 + 3}. VIF y colinealidad exacta")
    inf.parrafo(
        "El factor de inflación de la varianza (VIF) mide cuánto se amplía la varianza de un coeficiente por la correlación de la variable con las demás. Solo se calcula para variables "
        "numéricas y binarias, sobre los {n_vif} egresos que tienen todos los datos. Quedan fuera de ese conjunto de casos completos las variables constantes y las que tienen "
        "faltante igual o superior a {faltante|prop1} ({n_exc} en esta corrida), para no reducir la población de todo el cálculo. Sin los agregados de comorbilidad, el VIF más alto es "
        "{vmax_b|dec2} ({vmax_b_var|texto}); con ellos, sin contar las variables con colinealidad exacta, es {vmax_c|dec2} ({vmax_c_var|texto}). En el conjunto base, {n_alerta} variables "
        "tienen VIF de alerta (desde {alerta|entero}) y {n_crit} son críticas (desde {critico|entero}).",
        n_vif=rs("n_filas_vif"), n_exc=rs("n_vif_excluidas_faltante"), faltante=Cifra_umbral(ctx, "faltante_alerta"), vmax_b=rs("vif_max_base"),
        vmax_b_var=rs("vif_max_base_variable"), vmax_c=rs("vif_max_con_agregados"), vmax_c_var=rs("vif_max_con_agregados_variable"), n_alerta=rs("n_vif_alerta_base"),
        alerta=Cifra_umbral(ctx, "vif_alerta"), n_crit=rs("n_vif_critico_base"), critico=Cifra_umbral(ctx, "vif_critico"),
    )
    n_rel = res.get("n_relaciones_exactas_con_agregados")
    cifras_exactas = dict(
        n_col=rs("n_columnas_con_agregados"), rango=rs("rango_con_agregados"), n_rel=rs("n_relaciones_exactas_con_agregados"), n_exactas=rs("n_exactas_con_agregados"),
        d1=rs("identidad_n_comorb_max_abs_dif"), d2=rs("identidad_puntaje_max_abs_dif"), d3=rs("identidad_puntaje_sin5_max_abs_dif"),
    )
    if n_rel is None:
        inf.parrafo("Con los agregados de comorbilidad el VIF no se calculó: no hay casos completos suficientes.")
    else:
        inf.parrafo(
            ("Con los agregados de comorbilidad aparece colinealidad exacta: " if n_rel > 0 else "Con los agregados de comorbilidad no aparece colinealidad exacta: ")
            + "la matriz de correlación tiene {n_col} columnas y rango {rango}, es decir {n_rel} relaciones lineales exactas independientes, y {n_exactas} variables participan en "
            "ellas (su VIF no se calcula y la figura lo muestra como exacta). Se verificó con los datos que el conteo de categorías es la suma de los indicadores y que cada puntaje de "
            "van Walraven es la combinación con sus pesos: la diferencia máxima absoluta es {d1|dec2} para el conteo, {d2|dec2} para el puntaje completo y {d3|dec2} para el puntaje sin "
            "las cinco categorías sospechosas.",
            **cifras_exactas,
        )
    if "F10_03" in figuras:
        inf.figura("F10_03")
    vif = ctx.leer_tabla("t23_vif.csv", columnas_texto=["conjunto", "variable", "tipo", "es_agregado", "colinealidad_exacta", "clasificacion", "motivo", "estado"])
    base = vif.filter((pl.col("conjunto") == "base") & (pl.col("estado") == "ok")).sort("vif", descending=True, nulls_last=True, maintain_order=True).head(10)
    inf.tabla(
        "t23_vif.csv", df=base, columnas=["variable", "vif", "r2", "clasificacion"],
        encabezados={"variable": "Variable", "vif": "VIF", "r2": "R cuadrado", "clasificacion": "Clasificación"}, formatos={"vif": "dec2", "r2": "dec3"},
        leyenda="Las diez variables con mayor VIF en el conjunto base",
    )

    # --- 5. límites ---------------------------------------------------------------------------------------------
    inf.seccion(f"{n0 + 4}. Límites")
    lineas = []
    if ctx.etiqueta == "extracto":
        lineas.append(
            "- Los datos son un extracto y no una muestra aleatoria simple: nada de lo anterior es concluyente. Las asociaciones y los VIF prueban la mecánica y no sirven para decidir "
            "qué variables se descartan."
        )
    lineas += [
        "- Las asociaciones tratan cada egreso como independiente: ignoran el agrupamiento por paciente y por hospital y no llevan intervalo de confianza; los intervalos de la selección "
        "de variables tampoco lo consideran.",
        "- La V de Cramér depende del número de niveles de las variables: por eso se corrige por sesgo y las categóricas tienen un tope de niveles, con el resto agrupado en OTROS.",
        "- El VIF solo cubre variables numéricas y binarias; las categóricas se resumen con la V de Cramér, que no es comparable con un VIF.",
        "- La razón de correlación eta tiene un sesgo pequeño hacia valores altos cuando hay muchos niveles y pocos casos por nivel.",
        "- La estabilidad en 2023 solo se evalúa en los pares redundantes de la ventana de ranking: un par que no alcanzó el umbral en 2019 a 2022 no se reestima.",
        "- La matriz phi de las categorías de Elixhauser (F10_04) viene de la Tarea 8 y se calcula sobre adultos hospitalizados, no sobre la cohorte del panel; las dos bases difieren "
        "ligeramente.",
        "- La tabla de pares redundantes lleva la reestimación de 2023 en columnas aparte de la misma fila; por eso se escribe por bloques de supresión en lugar de separarla en dos "
        "tablas, que es la regla general de privacidad del proyecto. Cada bloque suprime solo sus propias cifras y todos los conteos quedan registrados para el escáner de privacidad.",
    ]
    inf.texto_libre_sin_cifras("\n".join(lineas))


# FIN-asociacion
