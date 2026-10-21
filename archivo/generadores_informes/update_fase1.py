import docx
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def update_fase1_doc(path):
    doc = docx.Document(path)
    
    # 1. Portada / Versión (P5)
    for p in doc.paragraphs[:10]:
        if "Versión 2" in p.text:
            p.text = "Versión 3: Actualizada y Validada con Evidencia Empírica de la Fase 2 (EDA)"
            if len(p.runs) > 0:
                p.runs[0].bold = True
            break
            
    # 2. Sección 0: Qué cambió en esta versión (P12)
    for i, p in enumerate(doc.paragraphs):
        if p.text.startswith("Esta versión reemplaza al informe de Fase 1 de septiembre"):
            p.text = (
                "Esta versión 3 consolida la comprensión del dominio mediante la incorporación sistemática "
                "de los hallazgos empíricos y análisis estadísticos obtenidos en la Fase 2 (Comprensión de los Datos). "
                "El informe está dirigido a la dirección del profesor guía, estructurado desde la disciplina de "
                "la ingeniería informática y delimitado formalmente por la restricción temporal estricta de 17 semanas "
                "del proyecto de titulación. No se recurre a ejemplos analógicos, sino a la exposición directa "
                "de las reglas de ingeniería, las limitaciones computacionales y los hallazgos medidos en los datos."
            )
            break

    # 3. Table 1: Control de cambios de la versión
    t1 = doc.tables[1]
    # Agregar filas de Fase 2
    fase2_changes = [
        ("Fuga causal (R6 y D14)", 
         "Se asumía que la severidad y mortalidad del GRD eran variables ajustables.", 
         "La Fase 2 midió que IR_29301_MORTALIDAD (AUC 0,860-0,871) e IR_29301_SEVERIDAD (AUC 0,842-0,858) superan el umbral de fuga de 0,85. La primera se prohíbe de forma absoluta (R6) y la segunda solo se reserva para análisis de sensibilidad aislado (D14).", 
         "5"),
        ("Cascada de exclusión", 
         "Estimaciones iniciales de exclusiones basadas en extractos preliminares (75, 7.760, 22.303).", 
         "Se documentó que los estimados iniciales provinieron de extractos muestrales con un factor de escala de 19,36x. En la población completa se proyectan ~5.073 GRD no agrupables, ~146.928 recién nacidos y ~431.832 obstétricos.", 
         "9.3"),
        ("Catálogo hospitalario", 
         "Se declaraba un universo fijo de 68 hospitales.", 
         "La suite identificó empíricamente al menos 72 recintos hospitalarios en el histórico 2019-2024 (68 bajo esquema activo de financiamiento GRD en 2023).", 
         "9.1"),
        ("Complicaciones intrahospitalarias", 
         "Lista de 5 comorbilidades sospechosas definida de forma teórica.", 
         "El índice empírico de complicación probable fijó como prioritarias a ELIX_24, ELIX_04, ELIX_25, ELIX_30 y ELIX_27 para el análisis de sensibilidad.", 
         "8.2"),
        ("Colinealidad exacta de Elixhauser", 
         "Se contemplaba usar simultáneamente índices agregados y categorías individuales.", 
         "Se comprobó que N_COMORB_ELIX y el puntaje de van Walraven generan colinealidad exacta (VIF infinito) con las 31 categorías, restringiéndose exclusivamente a líneas base simples.", 
         "8.3"),
        ("Servicio de ingreso", 
         "Sospecha de que SERVICIOINGRESO reflejaba la estancia y no el ingreso.", 
         "Se comprobó que solo coincide en 15,2 % en egresos con traslado interno, validando que captura la admisión. Se descarta la exclusión por causalidad pero se vigila por proxy.", 
         "5.3"),
        ("Truncamiento de estadías largas", 
         "Regla de corte de cola de estadía pendiente de definición.", 
         "Se confirmó empíricamente la regla del percentil 99 (corte en 51 días en la cohorte de sobrevivientes), afectando a menos del 1 % de los episodios.", 
         "10.2"),
        ("Etiquetas de tipo de alta", 
         "Asunción de unión exacta con el catálogo de tipo de alta.", 
         "Se constató que dos etiquetas llegan truncadas en los archivos de FONASA; se implementó regla de homologación por prefijo léxico para evitar perder el 0,72 % de los registros.", 
         "9.2"),
        ("Prevalencia de mortalidad", 
         "Se reportaba una tasa de mortalidad global de 2,78 %.", 
         "Al aplicar los filtros de cohorte dura y hospitalización adulta, la prevalencia real observada asciende de forma escalonada entre 3,41 % y 5,11 %.", 
         "9.5"),
        ("Restricción de 17 semanas", 
         "Cronograma descriptivo estándar.", 
         "Se fundamentan todas las decisiones de diseño arquitectónico y modelado bajo el límite no negociable de 17 semanas del semestre de titulación.", 
         "15, 18, 19")
    ]
    for tema, pasaba, hizo, sec in fase2_changes:
        row = t1.add_row()
        row.cells[0].text = tema
        row.cells[1].text = pasaba
        row.cells[2].text = hizo
        row.cells[3].text = sec

    # 4. Table 5: Regla R6
    t5 = doc.tables[5]
    for row in t5.rows:
        if row.cells[0].text.strip() == "R6":
            row.cells[2].text = (
                "La columna IR_29301_MORTALIDAD está prohibida como variable predictora bajo cualquier circunstancia, "
                "incluyendo los análisis de sensibilidad. En la Fase 2 se comprobó matemáticamente que su capacidad predictiva "
                "supera el umbral crítico de fuga causal (AUC de 0,860 en ranking y 0,871 en confirmación, sobre el límite de 0,85). "
                "Asimismo, IR_29301_SEVERIDAD (AUC de 0,842 a 0,858) queda restringida exclusivamente a un análisis de sensibilidad aislado (Decisión 14)."
            )
            row.cells[3].text = (
                "Introduciría fuga de información hacia atrás (data leakage) y circularidad algorítmica, copiando un resultado "
                "precalculado por el agrupador administrativo en lugar de aprender el riesgo clínico basal del paciente."
            )

    # 5. Sección 5.3: Servicio de ingreso
    for i, p in enumerate(doc.paragraphs):
        if "Variables condicionales" in p.text or "Servicio de ingreso (UCI/UTI)" in p.text:
            pass
        if "Si el servicio de ingreso registra la unidad donde estuvo el paciente durante la hospitalización, se descarta" in p.text:
            p.text = (
                "Variables condicionales (Servicio de ingreso). En la Fase 2 se contrastó empíricamente la concordancia entre "
                "SERVICIOINGRESO y SERVICIOALTA. En pacientes sin traslado interno la coincidencia es del 99,2 % (esperable por no existir "
                "movimiento de cama), pero en pacientes con traslado interno la coincidencia desciende drásticamente al 15,2 % "
                "(intervalo [14,8 %; 15,7 %]). Esta divergencia valida empíricamente que la variable registra efectivamente la condición "
                "al momento de la admisión y no el servicio de egreso. Por ende, no se activa la regla de descarte por fuga causal, "
                "aunque la variable se mantiene clasificada como condicional y bajo vigilancia de proxy hospitalario."
            )

    # 6. Sección 8.2: Comorbilidades sospechosas
    for i, p in enumerate(doc.paragraphs):
        if "Cinco son las más sospechosas: trastornos de líquidos y electrolitos" in p.text:
            p.text = (
                "Como FONASA no provee el marcador de diagnóstico presente al ingreso (Present on Admission - POA), ciertas categorías "
                "podrían constituir complicaciones desarrolladas durante la estadía. A partir del cálculo empírico del índice de complicación "
                "probable desarrollado en la Fase 2, se identificó el ordenamiento de las cinco categorías más propensas a ser complicaciones: "
                "ELIX_24 (trastornos de líquidos y electrolitos), ELIX_04 (trastornos de la circulación pulmonar), ELIX_25 (anemia por pérdida de sangre), "
                "ELIX_30 (pérdida de peso / desnutrición) y ELIX_27 (coagulopatía). Tres de ellas coinciden con la hipótesis inicial y dos con los antecedentes "
                "de la literatura internacional. La sensibilidad oficial del modelo se ejecutará suprimiendo este quinteto específico."
            )
        if "Mortalidad. Regresión logística con edad, sexo, grupo clínico y las 31 categorías." in p.text:
            p.text = (
                "Mortalidad. Regresión logística con edad, sexo, grupo clínico y las 31 categorías de Elixhauser (mapeo completo de Quan a cuatro caracteres). "
                "Advertencia de multicolinealidad: el análisis estadístico de la Fase 2 demostró que N_COMORB_ELIX, PUNTAJE_VAN_WALRAVEN y "
                "PUNTAJE_VAN_WALRAVEN_SIN_5 son combinaciones lineales exactas de las 31 categorías individuales (VIF infinito). "
                "Por consiguiente, los índices agregados se reservan únicamente para modelos basales univariados y queda técnicamente prohibido "
                "incorporarlos de forma simultánea junto a las 31 categorías individuales dentro del mismo modelo."
            )

    # 7. Sección 9.1: Hospitales y Tipo de Alta
    for i, p in enumerate(doc.paragraphs):
        if "comparar hospitales con los 68 de los informes oficiales" in p.text or "68 de los informes oficiales" in p.text:
            p.text = p.text.replace("68 de los informes oficiales", "al menos 72 hospitales identificados en los registros históricos de 2019 a 2024 (68 bajo esquema activo de financiamiento GRD en 2023)")
        if "9.2 Qué hacemos con cada tipo de alta" in p.text:
            pass

    # Agregar nota sobre etiquetas truncadas después de P96
    for i, p in enumerate(doc.paragraphs):
        if p.text.startswith("9.2 Qué hacemos con cada tipo de alta"):
            p_next = doc.paragraphs[i+1]
            p_next.text = (
                p_next.text + " En la auditoría de catálogos de la Fase 2 se constató que dos de las 11 etiquetas del campo TIPOALTA "
                "(específicamente DERIV_HOSP_RED y DERIV_PRIVADA_COMPRA) llegan truncadas en los archivos fuente originales. "
                "Para evitar que una unión relacional estricta descarte el 0,72 % de los episodios sin asignar rol, el pipeline de preprocesamiento "
                "implementa una asignación determinista mediante prefijos léxicos estandarizados."
            )
            break

    # 8. Table 12: Cascada de exclusiones
    t12 = doc.tables[12]
    # Modificar textos de volumen estimado
    for row in t12.rows:
        cat = row.cells[0].text.strip()
        if "GRD no asignable" in cat:
            row.cells[1].text = "215 en muestra de desarrollo; proyectado a ~5.073 en población histórica completa."
            row.cells[2].text = "Sin GRD válido no es factible clasificar la casuística clínica del egreso."
        elif "Recién nacidos" in cat:
            row.cells[1].text = "7.760 en muestra de desarrollo; proyectado a ~146.928 en población histórica completa."
            row.cells[2].text = "Corresponde a neonatología fisiológica postparto y no a morbilidad aguda. Distorsionaría artificialmente el O/E de recintos materno-infantiles."
        elif "Partos normales" in cat:
            row.cells[1].text = "19.899 en muestra de desarrollo (2019-2023); proyectado a ~431.832 en población histórica completa."
            row.cells[2].text = "Eventos obstétricos sin complicación con mortalidad basal nula y estadía protocolizada."
        elif "Traslados salientes" in cat:
            row.cells[1].text = "~89.438 casos históricos (censurados)."
            row.cells[2].text = "Desenlace intrahospitalario desconocido; su exclusión previene el sesgo de alta anticipada."

    # 9. Sección 9.5: Prevalencia
    for i, p in enumerate(doc.paragraphs):
        if "El 2,78 % de mortalidad corresponde a la base completa" in p.text:
            p.text = (
                "Prevalencia de la cohorte final: La tasa bruta global de mortalidad en los registros crudos es de 2,78 %. "
                "Sin embargo, la aplicación de los filtros sucesivos de cohorte incrementa la prevalencia de forma consistente: "
                "Cohorte A (cohorte dura sin neonatos sanos ni partos no complicados): 3,41 %; "
                "Cohorte B (adultos >= 18 años): 4,02 %; "
                "Cohorte C (hospitalización adulta en camas básicas/críticas): 4,72 %; "
                "Cohorte D (episodios con desenlace conocido para el modelo de mortalidad): 5,11 %. "
                "Esta prevalencia del 5,11 % constituye la base de referencia estricta para evaluar la ganancia del PR AUC en el modelado."
            )
            break

    # 10. Sección 10.1: Modelo L1 y Estadías Largas
    for i, p in enumerate(doc.paragraphs):
        if "Se comparan tres algoritmos: regresión logística (la línea base), LightGBM y XGBoost" in p.text:
            p.text = (
                p.text + " Advertencia técnica sobre penalización L1: En las exploraciones de la Fase 2, un modelo logístico con penalización L1 "
                "alcanzó un AUROC aparente de 0,925 en el año de confirmación 2023. No obstante, se comprobó empíricamente que dicho desempeño está "
                "sobreestimado por la presencia de variables con correlación contextual o proxy con el hospital (tales como NIVEL_CUIDADO_INGRESO, "
                "TIPO_INGRESO, PROCEDENCIA_AGR, ESPECIALIDAD_MEDICA y SERVICIOINGRESO). Dicha cifra no representa una capacidad predictiva clínicamente "
                "esperable y el modelo definitivo deberá evaluarse prescindiendo de estas covariables contaminantes."
            )
        if "Estadías muy largas. La regla para tratarlas se define en la Fase 2" in p.text:
            p.text = (
                "Estadías muy largas: La regla de truncamiento de colas prolongadas se fijó empíricamente en el percentil 99 de la distribución de estadía "
                "(corte en 51 días, p99 = 51) calculado sobre la cohorte de sobrevivientes (Cohorte E) en la ventana histórica 2019-2022. "
                "Dicho corte fue confirmado en la cohorte 2023, afectando a solo el 0,97 % de los episodios en entrenamiento y al 0,78 % en confirmación, "
                "preservando la estabilidad del denominador en el cálculo del O/E de estancia."
            )

    # 11. Sección 15, 18, 19: Justificación formal del límite de 17 semanas
    for i, p in enumerate(doc.paragraphs):
        if p.text.startswith("Se mantiene el orden y la duración del cronograma del seminario, que sumaba 17 semanas"):
            p.text = (
                "El proyecto se rige por un cronograma estricto de 17 semanas para su ejecución completa. Desde la óptica de la ingeniería informática, "
                "esta restricción temporal impone decisiones de diseño fundamentales que salvaguardan la viabilidad técnica y la calidad del producto:\n"
                "1. Adopción de la metodología CRISP-DM apoyada en una suite reproducible de pruebas automatizadas (1.747 pruebas unitarias e integración) "
                "que previene regresiones y garantiza determinismo de datos sin consumir tiempo en depuraciones manuales.\n"
                "2. Desarrollo y calibración algorítmica sobre extractos muestrales rigurosos de 50.000 registros por año, asegurando que la arquitectura "
                "y el cortafuegos estén completamente probados antes de someter la infraestructura a la carga masiva de los 5,8 millones de registros.\n"
                "3. Enfoque exclusivo en modelos de ensamble de árboles tabulares (LightGBM y XGBoost) y funciones nativas Poisson/Tweedie, descartando "
                "arquitecturas de aprendizaje profundo (Deep Learning) que requerirían semanas adicionales de calibración sin ventajas en datos tabulares.\n"
                "4. Delimitación del alcance administrativo: se descarta la vinculación nominal externa con el Registro Civil para mortalidad a 30 días post-alta, "
                "declarando el sesgo de alta anticipada como una limitación estructural transparente de los datos administrativos disponibles."
            )
            break

    # 12. Table 19: Decisiones metodológicas (completar las 17 con evidencia de Fase 2)
    t19 = doc.tables[19]
    # Limpiar filas existentes salvo encabezado
    while len(t19.rows) > 1:
        t19._tbl.remove(t19.rows[-1]._tr)
        
    decisiones_fase2 = [
        ("1", "Historial de internación previa", "Variables de hospitalización previa en 12 meses y días desde el último egreso.", "Gris / Descartado temporalmente: el supuesto S01 se rechazó al constatar 1,82 % de fechas de nacimiento inconsistentes en el identificador encriptado. Requiere datos maestros del Registro Civil."),
        ("2", "Variables condicionales (servicio/especialidad)", "Evaluar si registran gravedad basal o capacidad/proceso hospitalario.", "Amarillo / Evaluado: SERVICIOINGRESO refleja efectivamente la admisión (solo 15,2 % de coincidencia en traslados internos). Se vigila proxy hospitalario en especialidad médica."),
        ("3", "Variables de equidad y territorio", "Reservar previsión, nacionalidad, provincia y comuna fuera del modelo.", "Verde / Aprobado: se preservan en módulo de equidad y auditoría; no ingresan a los modelos basales."),
        ("4", "Etnia", "Almacenar variable en tabla de equidad sin modelar en esta tesis.", "Gris / Aprobado institucionalmente: se preserva en repositorio sin intervenir en la estimación de riesgo clínico."),
        ("5", "Población de estudio", "Restringir el alcance exclusivamente a adultos (>= 18 años).", "Amarillo / Aprobado: evita distorsiones de la mortalidad pediátrica, focalizando la calibración en una tasa observada de 4,02 % en adultos."),
        ("6", "Volumen mínimo institucional", "Umbral de 500 egresos y 10 muertes esperadas para publicación hospitalaria.", "Gris / Condicional: protege la estabilidad del funnel plot y evita reidentificación de pacientes en celdas pequeñas con baja casuística."),
        ("7", "Tratamiento de colas de estadía", "Fijar regla de truncamiento con datos históricos de 2019 a 2022.", "Verde / Aprobado: regla de percentil 99 (51 días en sobrevivientes) confirmada empíricamente en 2023 (<1 % de afectación)."),
        ("8", "Sensibilidad de comorbilidades Elixhauser", "Análisis de sensibilidad suprimiendo categorías sospechosas de complicación.", "Amarillo / Aprobado: quinteto priorizado empíricamente mediante el índice de complicación probable (ELIX_24, 04, 25, 30, 27)."),
        ("9", "Banda de calibración de estadía", "Razón O/E global de días de estancia contenida en rango [0,95; 1,05].", "Amarillo / Aprobado: función objetivo Poisson/Tweedie en LightGBM/XGBoost modela la media directa sin subestimación geométrica."),
        ("10", "Identificación visual de hospitales", "Exhibición de nombres institucionales o anonimización en prototipo.", "Gris / Institucional: visualización mediante códigos hospitalarios y entornos académicos protegidos según el marco ético."),
        ("11", "Tratamiento de valores estructurados como RUT", "Canalización de hallazgos de calidad hacia FONASA.", "Amarillo / Verificado: cero violaciones de privacidad en salidas agregadas; protocolo de resguardo activo."),
        ("12", "Validación clínica de reglas", "Mecanismo de gobernanza médica de las definiciones adoptadas.", "Gris / Institucional: supeditado a contraparte de auditoría médica hospitalaria."),
        ("13", "Rastreo de traslados salientes", "Reconstrucción de trayectoria clínica longitudinal entre recintos.", "Gris / Descartado temporalmente: inviable técnicamente sin consistencia determinista del CIP en el tiempo; se excluyen de la cohorte activa."),
        ("14", "Severidad del GRD", "Exclusión del modelo principal y restricción a sensibilidad.", "Verde / Aprobado: comprobación matemática de fuga causal (AUC 0,842-0,858); prohibida en el modelo central."),
        ("15", "Estadías de cero días", "Tratamiento de episodios ambulatorios en mortalidad.", "Amarillo / Aprobado: conservados en mortalidad intrahospitalaria por capturar defunciones tempranas; excluidos del modelo de estancia."),
        ("16", "Exclusión de traslados entrantes", "Tratamiento de pacientes recepcionados desde otros centros.", "Gris / Aprobado: excluidos de la evaluación central para evitar doble contabilidad y autorreferencia detectada (30,6 % en HOSPPROCEDENCIA)."),
        ("17", "Ventana de entrenamiento", "Inclusión o exclusión de período pandémico COVID-19 (2020-2021).", "Amarillo / Aprobado: entrenamiento con partición temporal 2019-2022 con análisis de sensibilidad obligatorio excluyendo 2020-2021.")
    ]
    for n, tema, prop, resol in decisiones_fase2:
        row = t19.add_row()
        row.cells[0].text = n
        row.cells[1].text = tema
        row.cells[2].text = prop
        row.cells[3].text = resol

    doc.save(path)
    print(f'Successfully updated {path}')

if __name__ == '__main__':
    update_fase1_doc('Informe_Fase1_v2_corregido (2).docx')
