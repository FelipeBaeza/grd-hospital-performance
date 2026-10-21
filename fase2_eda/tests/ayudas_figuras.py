"""Ayudas de prueba para figuras con etiquetas (Tareas 10 y 11): detectan solapes sobre el diseño final, el que ve ``savefig``."""

from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.transforms import Bbox


def _area(a: Bbox, b: Bbox) -> float:
    ancho = min(a.x1, b.x1) - max(a.x0, b.x0)
    alto = min(a.y1, b.y1) - max(a.y0, b.y0)
    return float(ancho * alto) if ancho > 0 and alto > 0 else 0.0


def pares_solapados(fig: Figure, tolerancia_px2: float = 1e-6) -> list[tuple[str, str, float]]:
    """Pares de textos de los ejes (``ax.texts``) cuyas cajas se solapan, con el área del solape en píxeles cuadrados.

    Dibuja la figura antes de medir, de modo que mide sobre el diseño final.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cajas = [(t.get_text(), t.get_window_extent(renderer)) for ax in fig.axes for t in ax.texts if t.get_visible() and t.get_text()]
    solapes = []
    for i in range(len(cajas)):
        for j in range(i + 1, len(cajas)):
            a = _area(cajas[i][1], cajas[j][1])
            if a > tolerancia_px2:
                solapes.append((cajas[i][0], cajas[j][0], a))
    return solapes


def vigilar_savefig(monkeypatch) -> list[list[tuple[str, str, float]]]:
    """Reemplaza ``Figure.savefig`` por una función que, justo antes de guardar, mide los solapes de ``ax.texts`` con el diseño final.

    Devuelve la lista en la que se acumula el resultado de cada guardado (una lista de pares solapados por figura guardada): un solape en el PNG final se detecta sin mirar la
    imagen. La función original sigue guardando el archivo.
    """
    original = Figure.savefig
    resultados: list[list[tuple[str, str, float]]] = []

    def guardar(self, *args, **kwargs):
        resultados.append(pares_solapados(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", guardar)
    return resultados
