"""Escáner de privacidad por propiedad (plan de corrección 4.2; hallazgo H-09).

El escáner original (privacidad.py) revisa **patrones**: columnas con nombre de conteo y celdas de 1 a 9. La
auditoría mostró que eso no basta: un conteo oculto se puede reconstruir combinando lo publicado. Este módulo revisa
la **propiedad** que importa: que ningún conteo de 1 a 9 personas se pueda obtener desde lo publicado.

Reconstrucciones que intenta, dentro de cada carpeta de salidas (un mismo año y una misma evaluación):

1. **Tasa × tamaño.** Cada tasa o proporción por hospital (``tasa_*``, ``proporcion_*``; ``*_pct`` en porcentaje),
   multiplicada por cada tamaño publicado del mismo hospital en cualquier tabla de la carpeta (``n``,
   ``n_egresos``, ``egresos*``). Teniendo en cuenta los decimales con que se escribió la tasa, si el producto fija un
   único entero de 1 a 9, es un hallazgo.
2. **O/E × esperado.** En una misma fila, un observado oculto con O/E y esperado publicados.
3. **Complementos.** En las tablas por subgrupo (``variable``, ``grupo``), si una variable tiene todos sus grupos
   publicados, su suma es el total. En otra variable con un solo grupo oculto, ese grupo es el total menos los demás.

Además, ``valores_en_textos`` busca en el código de pruebas y en los documentos números que coincidan con el
esperado o los límites de una fila suprimida. La prueba de privacidad original usaba esos valores reales.

Las pruebas plantan una fuga de cada tipo (canarios) y comprueban que el escáner la detecta.
"""
from __future__ import annotations

import math
import re
from decimal import Decimal
from pathlib import Path

import pandas as pd

COL_N = re.compile(r"^(n|n_egresos|egresos.*)$")
COL_TASA = re.compile(r"^(tasa_.*|proporcion_.*)$")
COL_PCT = re.compile(r"^.*_pct$")
COL_CONTEO = re.compile(r"^(observado.*|n|muertes|casos)$")
CLAVES_CONTEXTO = ("tarea", "diseno", "anio")
EXTENSIONES_TEXTO = {".py", ".md", ".txt", ".csv", ".json", ".toml"}
RE_NUMERO = re.compile(r"(?<![\d.])\d+\.\d{6,}")


def _leer(ruta: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(ruta, dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return None


def _num(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def _media_unidad(s: str) -> float:
    """Media unidad del último decimal escrito: el error máximo por redondeo de ese número."""
    try:
        e = Decimal(s.strip()).as_tuple().exponent
    except Exception:
        return math.inf
    return 0.5 * 10.0 ** e if isinstance(e, int) else math.inf


def _entero_fijado(valor_txt: str, factor: float, escala: float = 1.0) -> int | None:
    """Si (valor ± redondeo) / escala × factor contiene un único entero, lo devuelve."""
    v = _num(valor_txt)
    if math.isnan(v) or factor <= 0:
        return None
    d = _media_unidad(valor_txt)
    lo, hi = (v - d) / escala * factor, (v + d) / escala * factor
    lo, hi = lo - 1e-9 * max(1.0, abs(lo)), hi + 1e-9 * max(1.0, abs(hi))
    enteros = range(math.ceil(lo), math.floor(hi) + 1)
    return enteros[0] if len(enteros) == 1 else None


def reconstrucciones(carpeta: Path | str, minimo: int = 10) -> list[dict]:
    """Hallazgos de conteos de 1 a minimo-1 reconstruibles desde los CSV de ``carpeta`` (recursivo, por subcarpeta)."""
    carpeta = Path(carpeta)
    hallazgos = []
    por_dir: dict[Path, dict[Path, pd.DataFrame]] = {}
    for ruta in sorted(carpeta.rglob("*.csv")):
        d = _leer(ruta)
        if d is not None and len(d):
            por_dir.setdefault(ruta.parent, {})[ruta] = d
    for dir_, tablas in por_dir.items():
        # tamaños publicados por hospital
        tamanos: dict[str, list[tuple[str, float]]] = {}
        for ruta, d in tablas.items():
            if "cod_hospital" not in d.columns:
                continue
            for c in d.columns:
                if COL_N.match(c):
                    for cod, x in zip(d["cod_hospital"], d[c]):
                        n = _num(x)
                        if n > 0 and float(n).is_integer():
                            tamanos.setdefault(str(cod), []).append((f"{ruta.name}:{c}", n))
        for ruta, d in tablas.items():
            rel = str(ruta.relative_to(carpeta))
            # 1. tasa × tamaño
            if "cod_hospital" in d.columns:
                for c in d.columns:
                    escala = 100.0 if COL_PCT.match(c) else (1.0 if COL_TASA.match(c) else None)
                    if escala is None:
                        continue
                    for cod, x in zip(d["cod_hospital"], d[c]):
                        for fuente, n in tamanos.get(str(cod), []):
                            k = _entero_fijado(x, n, escala)
                            if k is not None and 1 <= k < minimo:
                                hallazgos.append({"tipo": "tasa × tamaño", "archivo": rel, "columna": c, "cod_hospital": str(cod), "con": fuente, "conteo": k})
            # 2. O/E × esperado con observado oculto
            if {"observado", "oe", "esperado"} <= set(d.columns):
                for i, fila in d.iterrows():
                    obs = fila["observado"].strip()
                    if (obs == "" or obs.startswith("<")) and fila["oe"].strip() != "":
                        k = _num(fila["oe"]) * _num(fila["esperado"])
                        if not math.isnan(k) and k < minimo - 0.5:
                            hallazgos.append({"tipo": "O/E × esperado", "archivo": rel, "fila": int(i), "conteo": int(round(k))})
            # 3. complementos en tablas por subgrupo: el total sale de la variable con la suma más alta y todos sus grupos
            #    publicados; en otra variable, lo que falta (celdas ocultas o filas omitidas) es total − publicados
            if {"variable", "grupo"} <= set(d.columns):
                ctx = [k for k in CLAVES_CONTEXTO if k in d.columns]
                for clave, sub in (d.groupby(ctx) if ctx else [((), d)]):
                    for c in [c for c in sub.columns if COL_CONTEO.match(c)]:
                        vals = sub[c].map(_num)
                        completas = {v: float(vals[sub["variable"] == v].sum()) for v in sub["variable"].unique() if vals[sub["variable"] == v].notna().all()}
                        if not completas:
                            continue
                        total = max(completas.values())
                        for v in sub["variable"].unique():
                            k_ = vals[sub["variable"] == v]
                            resto = total - float(k_.sum())
                            if 1 <= round(resto) < minimo and abs(resto - round(resto)) < 1e-6:
                                ocultas = int(k_.isna().sum())
                                hallazgos.append({"tipo": "complemento", "archivo": rel, "contexto": "/".join(map(str, clave if isinstance(clave, tuple) else (clave,))),
                                                  "columna": c, "variable": v, "conteo": int(round(resto)),
                                                  "detalle": ("una celda oculta" if ocultas == 1 else f"{ocultas} celdas ocultas" if ocultas else "filas omitidas")})
    return hallazgos


def valores_sensibles(carpeta: Path | str) -> dict[str, str]:
    """Números que acompañan a un conteo suprimido (esperado, O/E y límites de su fila), con 10 cifras significativas."""
    out = {}
    for ruta in sorted(Path(carpeta).rglob("*.csv")):
        d = _leer(ruta)
        if d is None or "observado" not in d.columns:
            continue
        ocultas = d[d["observado"].str.strip().str.startswith("<") | (d["observado"].str.strip() == "")]
        for c in [c for c in d.columns if c in ("esperado", "oe") or c.startswith(("lim_", "ic_"))]:
            for x in ocultas[c]:
                v = _num(x)
                if not math.isnan(v) and v != 0:
                    out[f"{v:.10g}"] = f"{ruta.name}:{c}"
    return out


def valores_en_textos(raices: list[Path], sensibles: dict[str, str]) -> list[dict]:
    """Archivos de código o documentos que contienen un número de una fila suprimida (H-09)."""
    hallazgos = []
    for raiz in raices:
        for ruta in sorted(Path(raiz).rglob("*")):
            if not ruta.is_file() or ruta.suffix.lower() not in EXTENSIONES_TEXTO or "__pycache__" in ruta.parts:
                continue
            texto = ruta.read_text(encoding="utf8", errors="replace")
            for m in RE_NUMERO.finditer(texto):
                clave = f"{float(m.group()):.10g}"
                if clave in sensibles:
                    hallazgos.append({"tipo": "valor real en texto", "archivo": str(ruta), "valor": m.group(), "origen": sensibles[clave]})
    return hallazgos


def clave_hallazgo(h: dict) -> str:
    """Identificador estable de un hallazgo, para compararlo con los ya declarados."""
    partes = [h["tipo"], Path(h["archivo"]).as_posix()] + [str(h.get(k, "")) for k in ("contexto", "columna", "variable", "cod_hospital", "con", "fila")]
    return " | ".join(partes)


def declarados(ruta: Path | str) -> dict[str, str]:
    """Hallazgos conocidos y declarados en un incidente (columnas «clave» e «incidente»)."""
    ruta = Path(ruta)
    if not ruta.exists():
        return {}
    d = pd.read_csv(ruta, dtype=str)
    return dict(zip(d["clave"], d["incidente"]))


def verificar(salidas: list[Path], textos: list[Path], minimo: int = 10, ruta_declarados: Path | str | None = None) -> dict:
    """Puerta completa: reconstrucciones en todas las carpetas de salidas y valores sensibles en pruebas y documentos.
    Los hallazgos ya declarados en un incidente se informan aparte; la puerta falla solo con hallazgos nuevos."""
    rec = []
    for s in salidas:
        if Path(s).exists():
            rec += [{**h, "archivo": f"{Path(s).name}/{h['archivo']}"} for h in reconstrucciones(s, minimo)]
    sens = {}
    for s in salidas:
        if Path(s).exists():
            sens.update(valores_sensibles(s))
    txt = valores_en_textos([t for t in textos if Path(t).exists()], sens)
    dec = declarados(ruta_declarados) if ruta_declarados else {}
    nuevos = [h for h in rec + txt if clave_hallazgo({**h, "archivo": h["archivo"]}) not in dec]
    ya = [h for h in rec + txt if clave_hallazgo(h) in dec]
    return {"reconstrucciones": rec, "valores_en_textos": txt, "nuevos": nuevos, "declarados": ya, "ok": not nuevos}
