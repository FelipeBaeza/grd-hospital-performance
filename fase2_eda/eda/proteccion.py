"""Supresión complementaria (cambio 1.6 del protocolo): ningún conteo de 1 a 9 se puede despejar desde lo publicado.

La supresión primaria (``supresion.py``) anula cada celda de conteo de 1 a 9. No basta: si la celda anulada es parte de una suma
que el lector conoce (el total de un año, el total de la fila, un período agregado que suma sus años), y es la única oculta de esa
suma, se despeja restando. La revisión de octubre de 2026 encontró casos así en la corrida completa, por ejemplo:

- ``t24_rut_por_columna.csv``: 22 valores en 2019 (publicados en ``t24_reglas.csv``) menos 15 de USOSPABELLON dejan 7 para la otra columna;
- ``t23_panel_niveles.csv``: el total de la cohorte de 2023 menos mujeres y hombres deja el conteo de sexo «sin dato»;
- ``t22_conciliacion.csv``: las filas leídas menos las filas con dato dejan las filas sin dato.

Este módulo declara, para cada tabla con riesgo, esas sumas (**identidades**) y aplica la regla clásica de la supresión complementaria:
**ninguna identidad puede tener exactamente una celda oculta**. Si la tiene, se oculta además la celda publicada de menor valor de
esa misma identidad (la de menor valor pierde menos información; en un empate, la primera en el orden de la tabla) y se repite hasta
que ninguna identidad quede con una sola celda oculta. Las cifras derivadas de la fila (proporciones, tasas, intervalos) se anulan junto
con cualquier celda oculta de la fila, igual que en la supresión primaria.

Dos clases de identidad:

``Grupo``
    Las celdas de un conteo que comparten los valores de ``por`` suman un total conocido (o deducible de otra tabla). ``donde`` restringe
    las filas (por ejemplo, un período agregado y los años que suma). Con ``total`` (una columna de la tabla con el total del grupo), el
    resto ``total - suma`` es una celda implícita: si vale de 1 a 9 y no hay ninguna celda oculta, se oculta una.
``Fila``
    Celdas de una misma fila unidas por una igualdad (por ejemplo ``n_filas = n_nulos + n_informados``). Con ``resto_de``, el resto
    ``resto_de - suma de las demás`` es una celda implícita, como en ``Grupo``.

Las celdas nulas en la tabla original no son parte de ninguna identidad (no hay valor que despejar: un año bloqueado no entra en la suma).

Supuestos y límites (declarados): se supone que el lector conoce el total de cada grupo declarado aunque la tabla no lo traiga (es el
lado seguro); las identidades entre tablas distintas se cubren declarando el mismo grupo en cada tabla (la elección determinista de la
celda de menor valor hace que dos tablas con las mismas celdas y las mismas identidades oculten las mismas). Una combinación de varias
identidades a la vez (sistemas de ecuaciones) no se resuelve en general; la regla de dos celdas por identidad es la práctica habitual.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Grupo:
    por: tuple[str, ...]
    donde: tuple[tuple[str, tuple[str, ...]], ...] = ()
    total: str | None = None
    columnas: tuple[str, ...] | None = None
    preferir: tuple[str, ...] = ()
    total_oculto: str | None = None
    """Columna booleana: en los grupos donde vale verdadero, el total del grupo no se publica y tampoco se puede dejar deducir (por ejemplo,
    porque restado de un total público deja un conteo pequeño). Entonces cada grupo necesita al menos una celda oculta."""


@dataclass(frozen=True)
class Fila:
    columnas: tuple[str, ...]
    resto_de: str | None = None
    donde: tuple[tuple[str, tuple[str, ...]], ...] = ()
    preferir: tuple[str, ...] = ()
    """Columnas que se ocultan primero como complemento (aunque valgan cero), porque las demás se publican también en otras tablas."""


@dataclass(frozen=True)
class Proteccion:
    grupos: tuple[Grupo, ...] = ()
    filas: tuple[Fila, ...] = ()
    dependientes: tuple[str, ...] = ()
    conteos_extra: tuple[str, ...] = ()
    """Columnas de conteo que la llamada no declara en ``cols_n`` y que también se protegen (por ejemplo conteos copiados de un documento)."""
    dependencias: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """Para cada cifra derivada, los conteos de los que sale: se anula solo si alguno de ellos quedó oculto. Las derivadas sin entrada aquí
    se anulan si cualquier conteo de la fila quedó oculto (el lado seguro)."""


def donde(**filtros: Sequence[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Filtro de filas para ``Grupo`` y ``Fila``: columna -> valores admitidos (se comparan como texto)."""
    return tuple((c, tuple(str(v) for v in vals)) for c, vals in filtros.items())


@dataclass
class _Identidad:
    celdas: list[tuple[str, int]]
    resto: tuple[str, list[tuple[str, int]]] | None = None   # (celda total, celdas que se restan) para el resto implícito
    total_valor: float | None = None                         # total de columna (Grupo con total)
    preferir: tuple[str, ...] = ()
    total_oculto: bool = False                               # el total del grupo es secreto: el grupo necesita al menos una celda oculta


@dataclass
class Resultado:
    tabla: pl.DataFrame
    primarias: int
    complementarias: int
    identidades: int
    ocultas: dict[str, np.ndarray] = field(default_factory=dict)


def _texto(df: pl.DataFrame, c: str) -> list[str]:
    return [("" if v is None else str(v)) for v in df[c].to_list()]


def _filas_donde(df: pl.DataFrame, filtro: tuple[tuple[str, tuple[str, ...]], ...]) -> np.ndarray:
    m = np.ones(df.height, dtype=bool)
    for c, vals in filtro:
        if c not in df.columns:
            return np.zeros(df.height, dtype=bool)
        t = np.array(_texto(df, c), dtype=object)
        m &= np.isin(t, np.array(vals, dtype=object))
    return m


def _valores(df: pl.DataFrame, c: str) -> np.ndarray:
    return df[c].cast(pl.Float64, strict=False).to_numpy().astype(float)


def identidades(df: pl.DataFrame, cols: Sequence[str], prot: Proteccion) -> list[_Identidad]:
    """Lista de identidades de la tabla según la protección declarada (las celdas nulas en ``df`` no entran)."""
    V = {c: _valores(df, c) for c in df.columns if c in cols or any(c in f.columnas for f in prot.filas) or any(g.total == c for g in prot.grupos)}
    out: list[_Identidad] = []
    for g in prot.grupos:
        if any(k not in df.columns for k in g.por):
            continue
        filas = np.flatnonzero(_filas_donde(df, g.donde))
        if filas.size == 0:
            continue
        claves = list(zip(*[_texto(df, k) for k in g.por])) if g.por else [()] * df.height
        grupos: dict[tuple, list[int]] = {}
        for i in filas:
            grupos.setdefault(claves[i], []).append(int(i))
        columnas = [c for c in (g.columnas or cols) if c in V]
        for _, idx in sorted(grupos.items(), key=lambda kv: kv[1][0]):
            for c in columnas:
                celdas = [(c, i) for i in idx if not np.isnan(V[c][i])]
                if len(celdas) < 1:
                    continue
                tot = None
                if g.total is not None and g.total in V:
                    vals = {V[g.total][i] for i in idx if not np.isnan(V[g.total][i])}
                    tot = vals.pop() if len(vals) == 1 else None
                oculto = bool(g.total_oculto in df.columns and any(bool(df[g.total_oculto][i]) for i in idx)) if g.total_oculto else False
                out.append(_Identidad(celdas, total_valor=tot, preferir=g.preferir, total_oculto=oculto))
    for f in prot.filas:
        cols_f = [c for c in f.columnas if c in V]
        if len(cols_f) < 2:
            continue
        filas = np.flatnonzero(_filas_donde(df, f.donde))
        for i in filas:
            celdas = [(c, int(i)) for c in cols_f if not np.isnan(V[c][i])]
            if len(celdas) < 1:
                continue
            resto = None
            if f.resto_de is not None and f.resto_de in cols_f and not np.isnan(V[f.resto_de][i]):
                resto = ((f.resto_de, int(i)), [(c, int(i)) for c in cols_f if c != f.resto_de])
            out.append(_Identidad(celdas, resto=resto, preferir=f.preferir))
    return out


def _resto(ide: _Identidad, V: Mapping[str, np.ndarray]) -> float | None:
    """Valor de la celda implícita de la identidad (None si no tiene)."""
    if ide.resto is not None:
        (ct, it), menos = ide.resto
        if any(np.isnan(V[c][i]) for c, i in menos):
            return None
        return float(V[ct][it] - sum(V[c][i] for c, i in menos))
    if ide.total_valor is not None:
        return float(ide.total_valor - sum(V[c][i] for c, i in ide.celdas))
    return None


def _necesita(ide: _Identidad, V: Mapping[str, np.ndarray], H: Mapping[str, np.ndarray], minimo: int) -> bool:
    """¿La identidad deja despejar una celda oculta, o la suma de sus celdas ocultas cuando esa suma es de 1 a 9?

    Ocultar dos celdas pequeñas de la misma identidad no basta si su suma se deduce y también es pequeña (por ejemplo, adultos más edad
    desconocida de un hospital pediátrico): la unión de las dos categorías quedaría publicada con menos de diez casos.
    """
    ocultas = [(c, i) for c, i in ide.celdas if H[c][i]]
    k = len(ocultas)
    if ide.total_oculto:
        return k == 0            # con todas las celdas a la vista, el total secreto se deduciría sumando (con una oculta, ya no)
    r = _resto(ide, V)
    if r is None or abs(r) < 1e-9:
        if k == 1 and len(ide.celdas) >= 2:
            return True          # una sola celda oculta se despeja
        # varias celdas ocultas, todas pequeñas, cuya suma (que se deduce) también es pequeña. Sin signos en la identidad, la regla solo
        # mira el caso en que todas son pequeñas: así cualquier combinación de ellas es pequeña y conviene ocultar una más
        suma = float(sum(V[c][i] for c, i in ocultas))
        return 2 <= k < len(ide.celdas) and all(V[c][i] < minimo for c, i in ocultas) and 1 <= suma < minimo
    # con resto implícito, lo que se deduce es el resto más las partes ocultas (si el total está a la vista)
    if ide.resto is not None:
        (ct, it), menos = ide.resto
        if H[ct][it]:
            return False         # el total está oculto: ni el resto ni las partes ocultas se deducen
        d = r + float(sum(V[c][i] for c, i in menos if H[c][i]))
    else:
        d = r + float(sum(V[c][i] for c, i in ocultas))
    # si ya no queda ninguna celda a la vista, lo que se deduce es un total que se publica en otra parte: nada más que ocultar aquí
    return 1 <= d < minimo and k < len(ide.celdas)


def _elegir(ide: _Identidad, V: Mapping[str, np.ndarray], H: Mapping[str, np.ndarray], preferidas: Sequence[str]) -> tuple[str, int] | None:
    visibles = [(c, i) for c, i in ide.celdas if not H[c][i]]
    if not visibles:
        return None
    primero = [(c, i) for c, i in visibles if c in ide.preferir]
    if primero:
        return min(primero, key=lambda ci: (ide.preferir.index(ci[0]), V[ci[0]][ci[1]], ci[1]))
    orden = {c: k for k, c in enumerate(preferidas)}

    def clave(ci: tuple[str, int]) -> tuple:
        c, i = ci
        v = V[c][i]
        return (0 if c in orden else 1, 0 if v > 0 else 1, v if v > 0 else 0.0, i, orden.get(c, len(orden)))

    return min(visibles, key=clave)


def complementar(
    df: pl.DataFrame, cols_n: Sequence[str], prot: Proteccion, minimo: int, dependientes: Sequence[str] = (),
    forzadas: Mapping[str, np.ndarray] | None = None,
) -> Resultado:
    """Supresión primaria más complementaria de ``df`` (sin suprimir) según ``prot``. Devuelve la tabla protegida y los conteos.

    ``cols_n`` son las celdas de conteo (se ocultan si valen de 1 a 9 y pueden ocultarse como complemento); las demás columnas que
    aparecen en una identidad (por ejemplo el total de la fila) también pueden ocultarse como complemento, pero solo si ninguna celda de
    conteo sirve. ``forzadas`` (columna -> máscara) son celdas que otra tabla con las mismas celdas ya ocultó: se ocultan desde el
    comienzo y cuentan como complementarias. Lanza ``AssertionError`` si al terminar alguna identidad sigue con una sola celda oculta.
    """
    cols = [c for c in dict.fromkeys([*cols_n, *prot.conteos_extra]) if c in df.columns]
    ides = identidades(df, cols, prot)
    forzadas = {c: m for c, m in (forzadas or {}).items() if c in df.columns}
    usadas = list(dict.fromkeys([*cols, *(c for ide in ides for c, _ in ide.celdas), *forzadas]))
    V = {c: _valores(df, c) for c in usadas}
    P = {c: ((V[c] >= 1) & (V[c] < minimo)) if c in cols else np.zeros(df.height, dtype=bool) for c in usadas}
    H = {c: P[c].copy() for c in usadas}
    complementarias = 0
    for c, m in forzadas.items():
        nuevas = np.asarray(m, dtype=bool) & ~np.isnan(V[c]) & ~H[c]
        H[c] |= nuevas
        complementarias += int(nuevas.sum())
    for _ in range(10_000):
        cambio = False
        for ide in ides:
            if _necesita(ide, V, H, minimo):
                elegida = _elegir(ide, V, H, cols)
                if elegida is None:
                    continue
                c, i = elegida
                H[c][i] = True
                complementarias += 1
                cambio = True
        if not cambio:
            break
    pendientes = [ide for ide in ides if _necesita(ide, V, H, minimo)]
    assert not pendientes, f"supresión complementaria incompleta: {len(pendientes)} identidades con una sola celda oculta"
    fila_oculta = np.zeros(df.height, dtype=bool)
    for c in usadas:
        fila_oculta |= H[c]
    exprs = []
    for c in usadas:
        if H[c].any():
            exprs.append(pl.when(pl.Series(H[c])).then(None).otherwise(pl.col(c)).alias(c))
    deps = [c for c in dict.fromkeys([*dependientes, *prot.dependientes]) if c in df.columns and c not in usadas]
    origen = dict(prot.dependencias)
    for c in deps:
        if c in origen and all(o in H for o in origen[c]):
            m = np.zeros(df.height, dtype=bool)
            for o in origen[c]:
                m |= H[o]
        else:
            m = fila_oculta
        exprs.append(pl.when(pl.Series(m)).then(None).otherwise(pl.col(c)).alias(c))
    tabla = df.with_columns(exprs) if exprs else df
    return Resultado(tabla, int(sum(P[c].sum() for c in cols)), complementarias, len(ides), H)


def una_sola_oculta(df_original: pl.DataFrame, publicada: pl.DataFrame, cols_n: Sequence[str], prot: Proteccion, minimo: int) -> int:
    """Verificación independiente sobre la tabla publicada: identidades con exactamente una celda de conteo oculta (debe ser 0).

    Una celda está oculta si es nula en ``publicada`` y no lo era en ``df_original``. Sirve de auditoría en las pruebas.
    """
    cols = [c for c in dict.fromkeys([*cols_n, *prot.conteos_extra]) if c in df_original.columns]
    ides = identidades(df_original, cols, prot)
    V = {c: _valores(df_original, c) for c in {c for ide in ides for c, _ in ide.celdas} | set(cols)}
    H = {c: (~np.isnan(V[c])) & np.isnan(_valores(publicada, c)) for c in V}
    return sum(1 for ide in ides if _necesita(ide, V, H, minimo))


def suma_publicada(s: pl.Series) -> int | None:
    """Suma de una columna de conteo de una tabla ya publicada: nula si alguna celda está oculta (una suma parcial engaña y, junto al total,
    deja ver lo oculto). Las métricas del ledger que agregan celdas de una tabla protegida se calculan con esto y no con la tabla sin proteger."""
    return None if s.null_count() > 0 else int(s.sum())


# ====================================================================================================== auditoría entre tablas
def _celdas(df: pl.DataFrame, claves: Sequence[str], col: str) -> dict[tuple, float | None]:
    out: dict[tuple, float | None] = {}
    if any(k not in df.columns for k in [*claves, col]):
        return out
    for fila in df.select([*claves, col]).iter_rows():
        *k, v = fila
        out[tuple("" if x is None else str(x) for x in k)] = None if v is None else float(v)
    return out


def auditar_pares(pares: Sequence[tuple[str, pl.DataFrame, Sequence[str], str, str, pl.DataFrame, Sequence[str], str]]) -> pl.DataFrame:
    """Celdas comunes a dos tablas publicadas que una oculta y la otra muestra (debe salir vacía).

    Cada par es ``(nombre_a, tabla_a, claves_a, conteo_a, nombre_b, tabla_b, claves_b, conteo_b)``; las claves de las dos tablas van en el
    mismo orden y se comparan como texto. Una celda que una tabla oculta y la otra publica deja sin efecto la supresión de la primera.
    """
    filas = []
    for na, ta, ka, ca, nb, tb, kb, cb in pares:
        a, b = _celdas(ta, ka, ca), _celdas(tb, kb, cb)
        for k in sorted(set(a) & set(b)):
            va, vb = a[k], b[k]
            if (va is None) != (vb is None):
                filas.append({"tabla_oculta": na if va is None else nb, "tabla_publica": nb if va is None else na, "clave": " | ".join(k),
                              "valor_publicado": vb if va is None else va})
    return pl.DataFrame(filas, schema={"tabla_oculta": pl.String, "tabla_publica": pl.String, "clave": pl.String, "valor_publicado": pl.Float64})
