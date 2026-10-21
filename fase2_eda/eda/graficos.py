"""Infraestructura gráfica: estilo apto para daltonismo, pie con fuente e índice de figuras.

Reglas (requisito de la Fase 2):
- Backend Agg, paleta Okabe-Ito, PNG a 150 dpi sin metadatos variables: la salida es idéntica
  byte a byte entre corridas.
- Títulos y ejes en español; pie con la fuente (extracto o completo), los años y la nota de supresión.
- Ningún gráfico sin CSV de respaldo: las funciones reciben tablas leídas desde ``salidas/.../tablas``
  y ``guardar_figura`` falla si el CSV indicado no existe.
- Las figuras por hospital llevan aviso de uso interno y usan códigos, no nombres (decisión 10).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.colorbar import Colorbar  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from .manifiesto import sha256_archivo  # noqa: E402

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#F0E442", "#000000"]
GRIS_SUPRIMIDO = "#D9D9D9"
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list("okabe_div", ["#0072B2", "#FFFFFF", "#D55E00"])
CMAP_SECUENCIAL = "viridis"
DPI = 150
TAMANO = (8.0, 4.8)
RESERVA_PIE = 0.07  # fracción inferior reservada para el pie


class FiguraSinCSV(FileNotFoundError):
    """La figura no tiene un CSV de respaldo en la carpeta de tablas."""


def configurar_estilo() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.figsize": TAMANO,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.prop_cycle": cycler(color=OKABE_ITO),
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
        }
    )


configurar_estilo()


def nueva_figura(figsize: tuple[float, float] = TAMANO, **kw) -> tuple[Figure, plt.Axes]:
    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(rect=(0.0, RESERVA_PIE, 1.0, 1.0 - RESERVA_PIE))
    ax = fig.add_subplot(1, 1, 1, **kw)
    return fig, ax


def nueva_figura_multiple(nfil: int, ncol: int, figsize: tuple[float, float] = TAMANO, **kw):
    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(rect=(0.0, RESERVA_PIE, 1.0, 1.0 - RESERVA_PIE))
    ejes = fig.subplots(nfil, ncol, **kw)
    return fig, ejes


def pie_figura(fuente: str, anios: str, uso_interno: bool = False, extra: str = "") -> str:
    partes = [f"Fuente: egresos GRD de FONASA ({fuente}). Años: {anios}.", "Celdas con 1 a 9 casos suprimidas."]
    if extra:
        partes.append(extra)
    if uso_interno:
        partes.append("Uso interno: códigos de hospital sin nombres (decisión 10 abierta).")
    return " ".join(partes)


# ----------------------------------------------------------------------------- registro
@dataclass
class _Entrada:
    id_figura: str
    tarea: str
    titulo: str
    descripcion_alternativa: str
    csv: str
    fuente: str
    decision_relacionada: str
    archivo_png: str
    uso_interno: str
    sha256_png: str
    sha256_csv: str


ESQUEMA_INDICE = {
    "id_figura": pl.String,
    "tarea": pl.String,
    "titulo": pl.String,
    "descripcion_alternativa": pl.String,
    "csv": pl.String,
    "fuente": pl.String,
    "decision_relacionada": pl.String,
    "archivo_png": pl.String,
    "uso_interno": pl.String,
    "sha256_png": pl.String,
    "sha256_csv": pl.String,
}


class RegistroFiguras:
    def __init__(self, dir_figuras: Path, dir_tablas: Path, fuente: str) -> None:
        self.dir_figuras = Path(dir_figuras)
        self.dir_tablas = Path(dir_tablas)
        self.fuente = fuente
        self.dir_figuras.mkdir(parents=True, exist_ok=True)
        self._entradas: dict[str, _Entrada] = {}

    def guardar_figura(
        self,
        fig: Figure,
        *,
        id_figura: str,
        tarea: str | int,
        titulo: str,
        descripcion_alt: str,
        csv: str,
        anios: str,
        decision: str = "",
        uso_interno: bool = False,
        extra_pie: str = "",
        antes_de_guardar: Callable[[Figure], None] | None = None,
    ) -> Path:
        """Guarda la figura (PNG a 150 dpi) y la registra en el índice.

        ``antes_de_guardar`` es opcional: si se pasa, se llama UNA vez con la figura después de poner el título y el pie, de reservar el espacio final del pie y de un
        ``fig.canvas.draw()`` que aplica el diseño, justo antes de ``savefig``. Sirve para colocar etiquetas (``etiquetas_sin_solape``) sobre el diseño final, porque el título
        y el pie se agregan aquí y mueven los ejes. Sin el parámetro no se hace ningún ``draw`` adicional y el comportamiento es el de siempre.
        """
        csv_ruta = self.dir_tablas / csv
        if not csv_ruta.is_file():
            plt.close(fig)
            raise FiguraSinCSV(f"la figura {id_figura} no tiene CSV de respaldo: {csv}")
        if not descripcion_alt.strip():
            plt.close(fig)
            raise ValueError(f"la figura {id_figura} necesita una descripción alternativa")
        fig.suptitle(titulo, x=0.01, ha="left", fontsize=11, fontweight="bold")
        pie = fig.text(
            0.01,
            0.012,
            pie_figura(self.fuente, anios, uso_interno, extra_pie),
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="#333333",
            wrap=True,
        )
        # La reserva inferior crece si el pie, ya ajustado al ancho, ocupa más de dos líneas.
        reserva = max(RESERVA_PIE, pie.get_window_extent().height / fig.bbox.height + 0.02)
        motor = fig.get_layout_engine()
        if motor is not None and reserva > RESERVA_PIE:
            motor.set(rect=(0.0, reserva, 1.0, 1.0 - reserva))
        if antes_de_guardar is not None:
            fig.canvas.draw()                      # aplica el diseño final (con título y pie) para que el gancho mida sobre él
            antes_de_guardar(fig)
        destino = self.dir_figuras / f"{id_figura}.png"
        fig.savefig(destino, format="png", dpi=DPI, metadata={"Software": None})
        plt.close(fig)
        self._entradas[id_figura] = _Entrada(
            id_figura=id_figura,
            tarea=str(tarea),
            titulo=titulo,
            descripcion_alternativa=descripcion_alt.strip(),
            csv=csv,
            fuente=self.fuente,
            decision_relacionada=decision,
            archivo_png=destino.name,
            uso_interno="si" if uso_interno else "no",
            sha256_png=sha256_archivo(destino),
            sha256_csv=sha256_archivo(csv_ruta),
        )
        return destino

    def indice(self) -> pl.DataFrame:
        if not self._entradas:
            return pl.DataFrame(schema=ESQUEMA_INDICE)
        filas = [e.__dict__ for _, e in sorted(self._entradas.items())]
        return pl.DataFrame(filas, schema=ESQUEMA_INDICE)

    def escribir_indice(self, destino: Path) -> Path:
        self.indice().write_csv(destino)
        return destino

    def ids(self) -> list[str]:
        return sorted(self._entradas)


# --------------------------------------------------------------------------- utilidades
def _a_array(valores) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in valores], dtype=float)


def barras_verticales(
    ax: plt.Axes,
    categorias: Sequence[str],
    valores: Sequence[float | None],
    *,
    etiqueta_y: str,
    etiqueta_x: str = "",
    color: str = OKABE_ITO[0],
    anotar: bool = True,
    fmt: str = "{:,.0f}",
) -> None:
    """Barras simples; los valores ``None`` (suprimidos) se muestran como marca gris con ``n<10``."""
    x = np.arange(len(categorias))
    v = _a_array(valores)
    ok = ~np.isnan(v)
    ax.bar(x[ok], v[ok], color=color, width=0.7)
    if (~ok).any():
        ax.bar(x[~ok], np.zeros((~ok).sum()), color=GRIS_SUPRIMIDO)
        for xi in x[~ok]:
            ax.text(xi, 0, "n<10", ha="center", va="bottom", fontsize=7, color="#555555")
    if anotar:
        for xi, vi in zip(x[ok], v[ok]):
            ax.text(xi, vi, fmt.format(vi).replace(",", "."), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(list(categorias), rotation=0)
    ax.set_ylabel(etiqueta_y)
    if etiqueta_x:
        ax.set_xlabel(etiqueta_x)
    ax.grid(axis="x", visible=False)
    formato_miles(ax, "y")


def formato_miles(ax: plt.Axes, eje: str = "y") -> None:
    """Separador de miles con punto (estilo chileno) en el eje indicado."""
    from matplotlib.ticker import FuncFormatter

    f = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(f)


def formato_decimal(ax: plt.Axes, eje: str = "y") -> None:
    """Números sin notación científica y con coma decimal (útil en ejes logarítmicos)."""
    from matplotlib.ticker import FuncFormatter

    f = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(f)


def mapa_calor(
    fig: Figure,
    ax: plt.Axes,
    matriz: np.ndarray,
    filas: Sequence[str],
    columnas: Sequence[str],
    *,
    cmap=CMAP_SECUENCIAL,
    vmin: float | None = None,
    vmax: float | None = None,
    etiqueta_barra: str = "",
    anotar: bool = True,
    fmt: str = "{:.2f}",
    rotar_x: int = 0,
    ocultar_diagonal: bool = False,
    coma_decimal: bool = True,
) -> Colorbar:
    """Mapa de calor; las celdas NaN (suprimidas o no evaluables) se muestran en gris. Los decimales anotados usan coma.

    Devuelve la barra de color que crea (``formato_barra_decimal`` le pone la coma decimal). Es el valor de retorno explícito y no depende del orden
    en que se agregaron los ejes de la figura; las llamadas que lo ignoran no cambian.
    """
    m = np.ma.masked_invalid(np.asarray(matriz, dtype=float))
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cm = cm.with_extremes(bad=GRIS_SUPRIMIDO)
    im = ax.imshow(m, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(columnas)))
    ax.set_xticklabels(list(columnas), rotation=rotar_x, ha="right" if rotar_x else "center")
    ax.set_yticks(np.arange(len(filas)))
    ax.set_yticklabels(list(filas))
    ax.grid(False)
    ax.spines[:].set_visible(False)
    barra = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    if etiqueta_barra:
        barra.set_label(etiqueta_barra)
    if anotar and m.size <= 400:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.ma.is_masked(m[i, j]):
                    if not (ocultar_diagonal and i == j):
                        ax.text(j, i, "n/d", ha="center", va="center", fontsize=6, color="#555555")
                    continue
                r, g, b, _ = im.cmap(im.norm(m[i, j]))
                luminancia = 0.299 * r + 0.587 * g + 0.114 * b
                texto = fmt.format(m[i, j])
                if texto.startswith("-") and not texto.strip("-0.,"):
                    texto = texto[1:]                                                       # un negativo que se redondea a cero se anota sin signo ("0,00", no "-0,00")
                ax.text(
                    j,
                    i,
                    texto.replace(".", ",") if coma_decimal else texto,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if luminancia < 0.5 else "black",
                )
    return barra


def formato_barra_decimal(barra: Colorbar) -> None:
    """Coma decimal en las marcas de la barra de color que entrega ``mapa_calor``.

    Se fija en ``barra.formatter`` y no directamente en el eje: ``update_ticks`` (y ``set_ticks``) restituyen el formateador de la barra.
    """
    from matplotlib.ticker import FuncFormatter

    barra.formatter = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    barra.update_ticks()


def recuadros_de_grupo(ax: plt.Axes, bloques: Sequence[tuple[int, int]], *, color: str = "white", ancho: float = 1.0) -> list:
    """Dibuja el contorno de bloques diagonales ``(inicio, tamano)`` sobre un mapa de calor cuadrado (la celda ``i`` se centra en la coordenada ``i``).

    Devuelve los rectángulos creados, uno por bloque. Un bloque de tamaño menor que 1 no se dibuja.
    """
    from matplotlib.patches import Rectangle

    creados = []
    for inicio, tamano in bloques:
        if tamano < 1:
            continue
        r = Rectangle((inicio - 0.5, inicio - 0.5), tamano, tamano, fill=False, edgecolor=color, linewidth=ancho, clip_on=True)
        ax.add_patch(r)
        creados.append(r)
    return creados


def linea_gruesa(ax: plt.Axes, posicion: float, *, color: str = "black", ancho: float = 1.6) -> list:
    """Línea horizontal y vertical entre la celda ``posicion - 1`` y la celda ``posicion`` de un mapa de calor (separa la referencia de las demás variables).

    Devuelve las dos líneas.
    """
    corte = posicion - 0.5
    return [ax.axhline(corte, color=color, linewidth=ancho), ax.axvline(corte, color=color, linewidth=ancho)]


def altura_filas(n: int, por_fila: float = 0.17, minimo: float = 4.0, extra: float = 1.8) -> float:
    """Alto de figura (pulgadas) para un gráfico de ``n`` filas con etiquetas legibles: ``n * por_fila + extra``, nunca menor que ``minimo``."""
    return max(float(minimo), float(n) * por_fila + extra)


_DIRECCIONES_ETIQUETA = (
    (0.0, "left", "center"),
    (45.0, "left", "bottom"),
    (90.0, "center", "bottom"),
    (135.0, "right", "bottom"),
    (180.0, "right", "center"),
    (225.0, "right", "top"),
    (270.0, "center", "top"),
    (315.0, "left", "top"),
)
"""Las ocho direcciones que prueba ``etiquetas_sin_solape`` (ángulo en grados, alineación horizontal y vertical), en orden: derecha, arriba a la derecha, arriba..."""
_RADIO_MARCADOR_PX = 4.0
"""Mitad del lado (en píxeles) del cuadro que ocupa un punto: una etiqueta no debe tapar ningún punto."""


def _area_interseccion(a: Bbox, b: Bbox) -> float:
    ancho = min(a.x1, b.x1) - max(a.x0, b.x0)
    alto = min(a.y1, b.y1) - max(a.y0, b.y0)
    return float(ancho * alto) if ancho > 0 and alto > 0 else 0.0


def etiquetas_sin_solape(
    ax: plt.Axes, xs: Sequence[float], ys: Sequence[float], textos: Sequence[str], *, tamano: float = 6.5, radios: Sequence[float] = (6, 12, 20)
) -> list:
    """Coloca una etiqueta junto a cada punto sin que se solape con otra etiqueta ni con ningún punto. Devuelve las anotaciones, en el orden de los puntos.

    Llama primero a ``ax.figure.canvas.draw()`` para que el diseño ``constrained`` ya esté aplicado al medir. Para cada punto, en el orden que da quien llama (el más
    importante primero), prueba ocho direcciones por cada radio (en puntos, del menor al mayor), mide el cuadro de la anotación con ``get_window_extent`` sobre el renderer
    Agg y se queda con el primer candidato cuyo cuadro no cruza una etiqueta ya puesta ni un punto (y cabe en la figura); si ninguno sirve, con el de menor solape (el primero
    de ellos en un empate). Cada anotación se crea con ``set_in_layout(False)`` para no cambiar el diseño ya medido. Determinista.

    Debe llamarse con el diseño final de la figura: úsela dentro del gancho ``antes_de_guardar`` de ``RegistroFiguras.guardar_figura``, porque el título y el pie se agregan
    después de que la función de la figura termina y mueven los ejes.
    """
    xs_a, ys_a = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    textos = [str(t) for t in textos]
    if not (len(xs_a) == len(ys_a) == len(textos)):
        raise ValueError(f"etiquetas_sin_solape: 'xs', 'ys' y 'textos' tienen largos distintos ({len(xs_a)}, {len(ys_a)} y {len(textos)})")
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    centros = ax.transData.transform(np.column_stack([xs_a, ys_a])) if len(xs_a) else np.zeros((0, 2))
    r = _RADIO_MARCADOR_PX
    puntos = [Bbox.from_extents(cx - r, cy - r, cx + r, cy + r) for cx, cy in centros]
    lienzo = Bbox.from_extents(0.0, 0.0, float(fig.bbox.width), float(fig.bbox.height))
    puestas: list[Bbox] = []
    anotaciones = []
    for i, texto in enumerate(textos):
        ann = ax.annotate(texto, xy=(xs_a[i], ys_a[i]), xytext=(0.0, 0.0), textcoords="offset points", fontsize=tamano, annotation_clip=False)
        ann.set_in_layout(False)
        mejor: tuple[float, float, str, str, Bbox] | None = None
        mejor_costo = float("inf")
        for radio in radios:
            for angulo, ha, va in _DIRECCIONES_ETIQUETA:
                dx, dy = float(radio) * float(np.cos(np.radians(angulo))), float(radio) * float(np.sin(np.radians(angulo)))
                ann.xyann = (dx, dy)
                ann.set_ha(ha)
                ann.set_va(va)
                caja = ann.get_window_extent(renderer)
                costo = sum(_area_interseccion(caja, c) for c in puestas) + sum(_area_interseccion(caja, p) for p in puntos)
                costo += 10.0 * (caja.width * caja.height - _area_interseccion(caja, lienzo))        # lo que queda fuera de la figura pesa más que un solape
                if costo < mejor_costo - 1e-9:
                    mejor, mejor_costo = (dx, dy, ha, va, caja), costo
                if mejor_costo <= 1e-9:
                    break
            if mejor_costo <= 1e-9:
                break
        assert mejor is not None
        dx, dy, ha, va, caja = mejor
        ann.xyann = (dx, dy)
        ann.set_ha(ha)
        ann.set_va(va)
        puestas.append(caja)
        anotaciones.append(ann)
    return anotaciones


def tabla_a_matriz(
    df: pl.DataFrame, fila: str, columna: str, valor: str, orden_filas: Sequence[str] | None = None,
    orden_columnas: Sequence[str] | None = None
) -> tuple[np.ndarray, list[str], list[str]]:
    """Pivota una tabla larga a matriz (NaN donde no hay dato o está suprimido)."""
    filas = list(orden_filas) if orden_filas is not None else sorted(set(df[fila].cast(pl.String).to_list()))
    cols = list(orden_columnas) if orden_columnas is not None else sorted(set(df[columna].cast(pl.String).to_list()))
    pos_f = {f: i for i, f in enumerate(filas)}
    pos_c = {c: j for j, c in enumerate(cols)}
    m = np.full((len(filas), len(cols)), np.nan)
    for f, c, v in zip(df[fila].cast(pl.String).to_list(), df[columna].cast(pl.String).to_list(), df[valor].to_list()):
        if f in pos_f and c in pos_c and v is not None:
            m[pos_f[f], pos_c[c]] = float(v)
    return m, filas, cols
