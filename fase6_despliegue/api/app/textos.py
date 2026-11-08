"""Textos fijos del sistema: una sola fuente para la API y la web. Resumen de lo que mide el indicador y de sus límites (Fase 1, secciones 13 y 17; Fase 5, informe de cierre)."""

QUE_ES = ("El indicador compara, para cada hospital, las muertes (o los días de estadía) que ocurrieron con las que un modelo esperaba según el perfil de sus pacientes. "
          "Un O/E de 1 significa que ocurrió lo esperado. Sirve para decidir dónde conviene mirar con más detalle; no es un dictamen de calidad ni un ranking.")

NO_ES = [
    "No mide la calidad asistencial por sí solo ni prueba causalidad.",
    "No permite decir que un hospital es mejor o peor que otro: los hospitales se muestran en un gráfico por volumen y siempre en orden de código, no en una lista ordenada.",
    "No considera infraestructura, presupuesto ni dotación de personal.",
    "No corrige del todo el alta anticipada ni los traslados; por eso se muestran junto al O/E.",
    "Solo incluye adultos (18 años o más) hospitalizados en hospitales con GRD.",
]

ADVERTENCIAS = [
    {"id": "nivel_global", "titulo": "Nivel global levemente por debajo de 1",
     "texto": "En 2024 los modelos esperaban algo más de muertes (4 %) y de días de estadía (4 %) de los que ocurrieron. Los dos O/E globales (0,962 y 0,956) cumplen el rango de aceptación, pero su intervalo de confianza llega bajo 0,95."},
    {"id": "sobredispersion", "titulo": "Entre el 14,5 % y el 15,9 % de los hospitales queda fuera del límite del 95 %",
     "texto": "Los hospitales difieren más de lo que explican los datos disponibles: complejidad, forma de registrar, derivaciones o calidad. El indicador no separa esas causas; orienta una revisión."},
    {"id": "estadia_mayores", "titulo": "La estadía de los mayores se sobreestima",
     "texto": "En personas de 90 años o más el O/E de estadía es cercano a 0,91."},
    {"id": "traslados", "titulo": "Traslados entrantes y salientes",
     "texto": "Los hospitales de derivación reciben más pacientes trasladados (hasta 50 %) y algunos derivan a una proporción mayor de sus pacientes. Su O/E se lee con ese contexto."},
    {"id": "codificacion", "titulo": "La codificación de diagnósticos sigue creciendo",
     "texto": "En 2024 se registraron 3,7 % más diagnósticos de los esperados respecto de 2023. Si crece más en unos hospitales que en otros, el O/E se distorsiona; por eso se muestra el índice de exhaustividad."},
]

DEMO_AVISOS = [
    "El perfil es inventado: esta demo no es una herramienta clínica y no debe usarse para decidir sobre una persona.",
    "Las contribuciones son una explicación matemática del modelo (SHAP), no una causa biológica.",
    "El perfil no se guarda en ninguna parte ni se escribe en los registros del servidor.",
]

LIMITES_DATOS = ("Datos: egresos hospitalarios de FONASA (Ley 20.285) de 2019 a 2024; modelos entrenados con 2019–2022, recalibrados con 2023 y evaluados en 2024. Los resultados mostrados son los de la enmienda 03 del protocolo (cohorte de la Fase 3 v7, modelos de la Fase 4 v6); el año 2024 se abrió dos veces, con autorización, y nunca se usó para ajustar nada. "
                 "Los egresos se redondean a la decena y no se muestran cifras basadas en menos de 10 casos ni hospitales con menos de 500 egresos analizables. No se publican las muertes ni los días observados o esperados por hospital, ni los límites del gráfico de embudo. Los hospitales se identifican solo por su código.")
