"""Identidades de cada tabla publicada para la supresión complementaria (cambio 1.6 del protocolo; ver ``proteccion.py``).

Cada entrada dice qué sumas conoce el lector de la tabla: los totales por año (la Tarea 3 publica las filas leídas de cada año), los
totales de cada cohorte (la Tarea 7 publica la cascada), los períodos agregados que suman sus años (2019-2022, 2019-2023 y el total de
las reglas), y las igualdades dentro de una fila (filas leídas = nulos + informados). Ante la duda se declara la suma como conocida: es
el lado seguro, porque solo agrega celdas ocultas.

``FAMILIAS`` une tablas que publican las mismas celdas (por ejemplo, la distribución de TIPOALTA de todos los egresos aparece en la
Tarea 4 y en la Tarea 7): la tabla que se escribe después oculta también las celdas que la primera ocultó, para que una no destape lo
que la otra protege.
"""

from __future__ import annotations

from .proteccion import Fila, Grupo, Proteccion, donde

ANIOS = ("2019", "2020", "2021", "2022", "2023", "2024")
A19_22 = ("2019", "2020", "2021", "2022")
A19_23 = ("2019", "2020", "2021", "2022", "2023")


def _agregados(col: str, por: tuple[str, ...]) -> tuple[Grupo, ...]:
    """Los períodos 2019-2022 y 2019-2023 suman sus años: una identidad por período y por cada combinación de ``por``.

    La tercera identidad es la resta de las dos: 2019-2023 menos 2019-2022 es 2023. Sin ella, un 2023 oculto con los dos períodos a la vista
    se despejaría aunque cada suma por separado tenga dos celdas ocultas.
    """
    return (Grupo(por, donde=donde(**{col: ("2019-2022", *A19_22)})), Grupo(por, donde=donde(**{col: ("2019-2023", *A19_23)})),
            Grupo(por, donde=donde(**{col: ("2019-2022", "2019-2023", "2023")})))


def _solo_anios(col: str, por: tuple[str, ...], **kw) -> Grupo:
    return Grupo((col, *por), donde=donde(**{col: ANIOS}), **kw)


def _tipoalta_como_t23(col_anio: str, col_clave: str) -> tuple[Grupo, ...]:
    """Las identidades que ``t23_tipoalta.csv`` declara para TIPOALTA, para las tablas de la Tarea 4 que publican las mismas celdas.

    La Tarea 7 publica los períodos 2019-2022 y 2019-2023 de cada etiqueta (que suman sus años) y el total de cada rol por año (en la
    cascada). Declararlas también aquí hace que la Tarea 4 oculte las mismas celdas que después ocultará la Tarea 7.
    """
    t = donde(columna=("TIPOALTA",))
    return (
        Grupo(("columna", col_clave), donde=(*t, *donde(**{col_anio: A19_22}))),
        Grupo(("columna", col_clave), donde=(*t, *donde(**{col_anio: A19_23}))),
        Grupo(("columna", col_anio, "_rol"), donde=(*t, *donde(**{col_anio: A19_23}))),
    )


REGISTRO: dict[str, Proteccion] = {
    # ------------------------------------------------------------------ Tarea 3
    "t21_deriva_formato.csv": Proteccion(
        grupos=(Grupo(("anio", "columna"), total="n_valores"),),
        filas=(Fila(("n_valores", "n_clase"), resto_de="n_valores"),),
        dependientes=("prop_clase",),
    ),
    # ------------------------------------------------------------------ Tarea 4
    "t22_perfil_columnas.csv": Proteccion(
        grupos=(Grupo(("columna",), donde=donde(anio=ANIOS), columnas=("n_nulos", "n_centinela", "n_informados")),),
        filas=(
            Fila(("n_filas", "n_nulos", "n_informados")),
            Fila(("n_informados", "n_modal"), resto_de="n_informados"),
            Fila(("n_informados", "n_patron_moda"), resto_de="n_informados"),
            Fila(("n_informados", "n_centinela"), resto_de="n_informados"),
            # los egresos con el valor modal tienen también el patrón modal: la diferencia es un conteo (con el patrón y otro valor)
            Fila(("n_patron_moda", "n_modal"), resto_de="n_patron_moda"),
        ),
        dependientes=("pct_nulos", "pct_centinela", "pct_faltante", "prop_modal", "prop_patron_moda", "chao1"),
        dependencias=(("pct_nulos", ("n_nulos",)), ("pct_centinela", ("n_centinela",)), ("pct_faltante", ("n_nulos", "n_centinela")),
                      ("prop_modal", ("n_modal", "n_informados")), ("prop_patron_moda", ("n_patron_moda", "n_informados")), ("chao1", ("n_informados",))),
    ),
    "t22_top_valores.csv": Proteccion(
        grupos=(Grupo(("columna", "anio"), total="_total"), *_tipoalta_como_t23("anio", "_clave")),
        dependientes=("prop",),
    ),
    "t22_deriva_categorias.csv": Proteccion(
        grupos=(Grupo(("columna", "anio")), *_tipoalta_como_t23("anio", "_clave")),
        dependientes=("prop",),
    ),
    "t22_hist_edad.csv": Proteccion(grupos=(Grupo(("anio",)),)),
    "t22_hist_estancia.csv": Proteccion(grupos=(Grupo(("anio",)),)),
    # n_fecha y n_servicio también salen en la conciliación con el diccionario (filas con dato de cada columna): el complemento se busca
    # primero entre las celdas propias de esta tabla
    "t22_pares_traslado.csv": Proteccion(
        filas=(Fila(("n_fecha", "n_servicio", "n_solo_fecha", "n_solo_servicio"), preferir=("n_solo_fecha", "n_solo_servicio")),), dependientes=("diferencia",)),
    "t22_bloque_rn.csv": Proteccion(filas=(Fila(("n_informados", "n_modal"), resto_de="n_informados"),), dependientes=("prop_modal",)),
    "t22_conciliacion.csv": Proteccion(
        filas=(Fila(("n_filas_med", "n_con_med", "n_sin_med")), Fila(("n_con_med", "n_centinela_med"), resto_de="n_con_med"),
               Fila(("con_datos_doc", "sin_datos_doc"))),
        dependientes=("pct_sin_doc", "pct_sin_med", "dif_pp", "z"),
        conteos_extra=("con_datos_doc", "sin_datos_doc"),
        dependencias=(("pct_sin_doc", ("con_datos_doc", "sin_datos_doc")), ("pct_sin_med", ("n_con_med", "n_sin_med")),
                      ("dif_pp", ("con_datos_doc", "sin_datos_doc", "n_con_med", "n_sin_med")), ("z", ("con_datos_doc", "sin_datos_doc", "n_con_med", "n_sin_med"))),
    ),
    "diccionario_medido.csv": Proteccion(
        filas=(Fila(("n_filas_med", "n_con_med", "n_sin_med")), Fila(("n_con_med", "n_centinela_med"), resto_de="n_con_med"),
               Fila(("con_datos_doc", "sin_datos_doc"))),
        dependientes=("pct_sin_doc", "pct_sin_med", "dif_pp"),
        conteos_extra=("con_datos_doc", "sin_datos_doc"),
        dependencias=(("pct_sin_doc", ("con_datos_doc", "sin_datos_doc")), ("pct_sin_med", ("n_con_med", "n_sin_med")),
                      ("dif_pp", ("con_datos_doc", "sin_datos_doc", "n_con_med", "n_sin_med")), ("z", ("con_datos_doc", "sin_datos_doc", "n_con_med", "n_sin_med"))),
    ),
    # ------------------------------------------------------------------ Tarea 5 y 6 (calidad)
    "t24_reglas.csv": Proteccion(
        grupos=(Grupo(("regla_id",), donde=donde(ambito=(*ANIOS, "total"))),
                Grupo(("regla_id",), donde=donde(ambito=("2019", "2020", "régimen 2019 a 2020"))),
                Grupo(("regla_id",), donde=donde(ambito=("2021", "2022", "2023", "2024", "régimen 2021 a 2024"))),
                Grupo(("regla_id",), donde=donde(ambito=("total", "régimen 2019 a 2020", "régimen 2021 a 2024")))),
        filas=(Fila(("n_evaluados", "n_violaciones"), resto_de="n_evaluados"),),
        dependientes=("prop_violaciones",),
    ),
    "t24_consistencia_cruzada.csv": Proteccion(
        grupos=(Grupo(("regla_id",), donde=donde(ambito=(*ANIOS, "total"))),),
        filas=(Fila(("n_evaluados", "n_violaciones"), resto_de="n_evaluados"),),
        dependientes=("prop_violaciones",),
    ),
    "t24_rut_por_columna.csv": Proteccion(grupos=(Grupo(("anio",)),)),
    "t24_id_largo.csv": Proteccion(grupos=(Grupo(("anio",)),), dependientes=("prop",)),
    "t24_capitulos_cie10.csv": Proteccion(grupos=(Grupo(("anio",)),), dependientes=("prop",)),
    "t24_validez_codigos.csv": Proteccion(grupos=(Grupo(("anio", "familia"), total="n_total_familia"),), dependientes=("prop",)),
    "t24_grd_tipo_actividad.csv": Proteccion(
        grupos=(Grupo(("anio",)), Grupo(("anio", "tipo_actividad")), Grupo(("anio", "tipo_grd")), Grupo(("anio", "grupo_grd")),
                Grupo(("anio", "grupo_grd", "tipo_actividad"))),
    ),
    # ------------------------------------------------------------------ Tarea 7 (cohorte)
    "t23_tipoalta.csv": Proteccion(
        grupos=(Grupo(("periodo",)), *_agregados("periodo", ("etiqueta",)), Grupo(("periodo", "rol"), donde=donde(periodo=A19_23)),
                Grupo(("periodo", "codigo"), donde=donde(periodo=("2019-2022", "2019-2023")))),
        dependientes=("prop",),
    ),
    # cada variante es la anterior menos un paso de la cascada: la diferencia entre dos variantes vecinas es un conteo (por ejemplo, los
    # fallecidos en actividad ambulatoria = B menos C). _hijo_* es el valor de la variante siguiente (lo agrega cohorte._escribir).
    # la cascada es una cadena de restas (TOTAL - EX01 - EX02 - EX03 = A, A - EX04 = B, ...) y cada período agregado suma sus años; en los
    # extractos una exclusión pequeña (EX01) se despejaba con la cadena
    "t23_cascada.csv": Proteccion(
        grupos=(*(Grupo(("periodo",), donde=donde(paso=pasos)) for pasos in (("TOTAL", "EX01", "EX02", "EX03", "A"), ("A", "EX04", "B"), ("B", "EX05", "C"),
                                                                              ("C", "EX06", "D"), ("D", "EX07", "EX08", "E"))),
                *_agregados("periodo", ("paso",)), Grupo(("paso",), donde=donde(periodo=("2019-2024", *ANIOS))),
                Grupo(("paso",), donde=donde(periodo=("2019-2023", "2019-2024", "2024")))),
        dependientes=("prop_del_total",),
    ),
    "t23_cascada_pipeline.csv": Proteccion(
        grupos=(Grupo(("periodo",), donde=donde(paso=("TOTAL", "EX01", "EX02", "EX03_PIPELINE", "COHORTE_DURA_PIPELINE"))),
                *_agregados("periodo", ("paso",)), Grupo(("paso",), donde=donde(periodo=("2019-2024", *ANIOS))),
                Grupo(("paso",), donde=donde(periodo=("2019-2023", "2019-2024", "2024")))),
        dependientes=("prop_del_total",),
    ),
    "t23_cascada_variantes.csv": Proteccion(
        grupos=(Grupo(("variante",), donde=donde(periodo=("2019-2023", *A19_23))),),
        # cada variante reparte su n entre los cuatro desenlaces (n = fallecidos + no evento + censurados + sin dato): sin esta identidad, la
        # celda oculta para proteger la resta de dos variantes se despejaba restando las otras de su fila
        filas=(*(Fila((c, f"_hijo_{c}"), resto_de=c) for c in ("n", "n_fallecidos", "n_no_evento", "n_censurados", "n_sin_dato")),
               Fila(("n", "n_fallecidos", "n_no_evento", "n_censurados", "n_sin_dato"))),
        dependientes=("prevalencia", "ic_bajo", "ic_alto", "prevalencia_conocida", "ic_bajo_conocida", "ic_alto_conocida", "dif_prevalencia_pp"),
    ),
    "t23_estancia_cero.csv": Proteccion(
        grupos=(Grupo(("periodo",)), Grupo(("periodo", "actividad")), Grupo(("periodo", "rol")), *_agregados("periodo", ("actividad", "rol"))),
        filas=(Fila(("n", "n_cero"), resto_de="n"), Fila(("n", "n_estancia_valida"), resto_de="n")),
    ),
    "t23_resultado_estratos.csv": Proteccion(
        grupos=(Grupo(("periodo", "variable"), columnas=("n",)), *_agregados("periodo", ("variable", "categoria")),
                # cada variable reparte la misma base, así que su suma es el total de la base. En los períodos donde los fallecidos de la base
                # (adultos con desenlace conocido) menos los de la cohorte D (públicos) dejan un conteo pequeño (la actividad ambulatoria), ese
                # total es secreto y ninguna variable puede quedar con todas sus celdas a la vista; _T_oculto marca esos períodos
                Grupo(("periodo", "variable"), total_oculto="_T_oculto", columnas=("n_fallecidos",))),
        filas=(Fila(("n", "n_fallecidos"), resto_de="n"),),
        dependientes=("participacion", "mortalidad", "ic_bajo", "ic_alto", "mortalidad_resto", "razon_vs_resto", "razon_bajo", "razon_alto", "bajo_umbral"),
    ),
    # el lector conoce el total de cada período (la cascada publica la cohorte A y sus censurados), pero no los totales por edad ni por sexo
    # de esta cohorte: ninguna otra tabla los publica, y declararlos ocultaba casi toda la tabla. Los sobrevivientes con estadía válida
    # (n_estancia) no se publican: n menos fallecidos menos n_estancia daba cuántos sobrevivientes no tenían estadía, un conteo de 1 a 9.
    "t23_resultado_edad_sexo.csv": Proteccion(
        grupos=(Grupo(("periodo",)), *_agregados("periodo", ("grupo_edad", "sexo"))),
        filas=(Fila(("n", "n_fallecidos"), resto_de="n"),),
        dependientes=("mortalidad", "ic_bajo", "ic_alto", "estancia_media"),
    ),
    # ------------------------------------------------------------------ Tarea 8 (comorbilidad)
    "t23_deriva_codificacion.csv": Proteccion(
        grupos=(Grupo(("anio",)),),
        dependientes=("suma", "media", "desv", "ic_bajo", "ic_alto", "cambio_relativo", "cambio_bajo", "cambio_alto"),
    ),
    # ------------------------------------------------------------------ Tarea 9 (hospitales)
    "t23_hospitales_anio.csv": Proteccion(
        grupos=(Grupo(("anio",)), Grupo(("cod_hospital",), columnas=("n_egresos",))),
        filas=(Fila(("n_egresos", "n_adultos", "n_menores", "n_edad_desconocida")),),
        dependientes=("prop_menores",),
    ),
    # cada traslado es inválido por identificador, elegible o queda fuera por otra razón (ventana incompleta): ese resto implícito era de
    # 1 a 9 en algunos años; las métricas y el resumen leen la tabla protegida
    "t23_enlaces_traslado.csv": Proteccion(
        grupos=_agregados("periodo", ("direccion", "ventana_dias")),
        filas=(Fila(("n_total", "n_id_invalido", "n_elegibles"), resto_de="n_total", preferir=("n_elegibles",)),
               Fila(("n_elegibles", "n_enlazados"), resto_de="n_elegibles")),
        dependientes=("tasa_enlace", "ic_bajo", "ic_alto"),
    ),
    "t23_procedencia_autorref.csv": Proteccion(
        grupos=(Grupo(("anio",)),),
        filas=(Fila(("n_episodios", "n_con_hospproc"), resto_de="n_episodios"), Fila(("n_con_hospproc", "n_evaluables"), resto_de="n_con_hospproc"),
               Fila(("n_evaluables", "n_autorref_exacta"), resto_de="n_evaluables"), Fila(("n_evaluables", "n_autorref_aprox"), resto_de="n_evaluables"),
               Fila(("n_autorref_aprox", "n_autorref_exacta"), resto_de="n_autorref_aprox")),
        dependientes=("prop_con_hospproc", "prop_autorref_exacta", "prop_autorref_aprox"),
    ),
    "t23_hospitales_volumen.csv": Proteccion(
        grupos=(Grupo(("anio",)), Grupo(("cod_hospital",))),
        dependientes=("esperados_D", "cumple_egresos", "cumple_muertes", "cumple_volumen"),
    ),
    "t23_embudo.csv": Proteccion(
        grupos=(_solo_anios("periodo", ()), Grupo(("cod_hospital",), donde=donde(periodo=("2019-2023", *A19_23)))),
        filas=(Fila(("n", "k"), resto_de="n"),),
        dependientes=("p", "z", "z_ajustado", "lim95_bajo", "lim95_alto", "lim998_bajo", "lim998_alto", "fuera_95", "fuera_998"),
    ),
    # ------------------------------------------------------------------ Tareas 10 y 11 (panel)
    "t23_panel_niveles.csv": Proteccion(grupos=(Grupo(("variable", "ventana")),), dependientes=("prop",)),
    # ------------------------------------------------------------------ análisis posteriores del profesor (R)
    "r1_resultado_anio_diagnostico.csv": Proteccion(
        grupos=(Grupo(("anio", "nivel"), total="_total_d", columnas=("n_d",)), Grupo(("anio", "nivel"), total="_total_k", columnas=("n_fallecidos",)),
                Grupo(("anio", "nivel"), total="_total_e", columnas=("n_e",)), Grupo(("nivel", "grupo")),
                # un capítulo publicado = sus códigos frecuentes publicados + el resto de sus códigos
                Grupo(("anio", "_cap"), donde=donde(nivel=("codigo",)), total="_tcap_d", columnas=("n_d",)),
                Grupo(("anio", "_cap"), donde=donde(nivel=("codigo",)), total="_tcap_k", columnas=("n_fallecidos",)),
                Grupo(("anio", "_cap"), donde=donde(nivel=("codigo",)), total="_tcap_e", columnas=("n_e",))),
        filas=(Fila(("n_d", "n_fallecidos"), resto_de="n_d"),),
        dependientes=("mortalidad", "ic_bajo", "ic_alto", "estancia_media", "estancia_media_tope"),
    ),
    "r2_faltantes_variables.csv": Proteccion(filas=(Fila(("n", "n_faltante"), resto_de="n"),), dependientes=("prop_faltante",)),
    "r4_traslados_tipo_hospital.csv": Proteccion(
        grupos=(Grupo(("periodo",), columnas=("n_c", "n_entrantes", "n_salientes", "n_hosp_domiciliaria", "n_d_entrantes", "n_fallecidos_entrantes",
                                             "n_d_no_entrantes", "n_fallecidos_no_entrantes")),),
        filas=(Fila(("n_c", "n_entrantes"), resto_de="n_c"), Fila(("n_c", "n_salientes"), resto_de="n_c"), Fila(("n_c", "n_hosp_domiciliaria"), resto_de="n_c"),
               Fila(("n_d_entrantes", "n_fallecidos_entrantes"), resto_de="n_d_entrantes"), Fila(("n_d_no_entrantes", "n_fallecidos_no_entrantes"), resto_de="n_d_no_entrantes")),
        dependientes=("prop_entrantes", "parte_de_todos_los_entrantes", "prop_salientes", "prop_hosp_domiciliaria", "mortalidad_entrantes", "mortalidad_no_entrantes",
                      "razon_mortalidad_entrantes"),
    ),
}

# Tablas que publican las mismas celdas: familia, columnas que forman la llave común (en el mismo orden en todas las tablas de la familia) y
# columnas de conteo con su nombre común. La tabla que se escribe después hereda como ocultas las celdas que ya ocultó la primera. La llave se
# compara como texto. Las columnas que empiezan con «_» son auxiliares: las agrega quien escribe la tabla y no se publican.
FAMILIAS: dict[str, tuple[str, tuple[str, ...], dict[str, str]]] = {
    "t22_perfil_columnas.csv": ("categorias_anio", ("columna", "anio", "valor_modal"), {"n_modal": "n"}),
    "t22_deriva_categorias.csv": ("categorias_anio", ("columna", "anio", "_clave"), {"n": "n"}),
    "t22_top_valores.csv": ("categorias_anio", ("columna", "anio", "_clave"), {"n": "n"}),
    "t23_tipoalta.csv": ("categorias_anio", ("_columna", "periodo", "_clave"), {"n": "n"}),
    "t23_procedencia_autorref.csv": ("categorias_anio", ("_columna", "anio", "_clave"), {"n_episodios": "n"}),
    "t23_hospitales_volumen.csv": ("volumen_d", ("anio", "cod_hospital"), {"n_D": "n"}),
    "t23_cascada.csv": ("cascada", ("periodo", "paso"), {"n": "n"}),
    "t23_cascada_pipeline.csv": ("cascada", ("periodo", "paso"), {"n": "n"}),
    "t23_cascada_variantes.csv": ("cascada", ("periodo", "variante"), {"n": "n"}),
    "t23_embudo.csv": ("volumen_d", ("periodo", "cod_hospital"), {"n": "n"}),
    "t22_conciliacion.csv": ("conciliacion", ("columna",), {c: c for c in ("n_con_med", "n_sin_med", "n_centinela_med", "con_datos_doc", "sin_datos_doc")}),
    "diccionario_medido.csv": ("conciliacion", ("columna",), {c: c for c in ("n_con_med", "n_sin_med", "n_centinela_med", "con_datos_doc", "sin_datos_doc")}),
}
