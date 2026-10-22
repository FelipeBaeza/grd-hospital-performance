# Limitaciones y respuestas al comité (plan de corrección 5.2 y 5.3)

Versión del 22-10-2026. Usa la evidencia de las enmiendas 01 y 02, que se corrieron una vez cada una con el código
registrado antes (`docs/fase5_evaluacion/Anexo_Enmienda01_resultados.docx` y `Anexo_Enmienda02_resultados.docx`). Las cifras de 2024 son del diseño original, rotulado
«ya abierto». Las de 2023 son del diseño anidado, que no usa 2024.

**Regla de lectura.** Una decisión «resiste» si al menos el 80 % de los hospitales conserva su zona en el embudo y la
correlación de rangos (ρ de Spearman) del O/E por hospital es de al menos 0,90.

---

## Parte 1. Capítulo de limitaciones

Cada limitación dice qué es, cuánto pesa según la evidencia y qué se hizo o queda como trabajo futuro.

### 1.1 Datos

| Limitación | Qué implica | Evidencia | Qué se hizo / trabajo futuro |
|---|---|---|---|
| **Sin marcador de diagnóstico presente al ingreso (POA)** | No se puede distinguir una comorbilidad previa de una complicación del hospital | Las 5 categorías sospechosas tienen OR de 3,6 a 5,8 (Fase 2) | Sin ellas, el indicador resiste: 91–100 % de los hospitales en la misma zona, ρ 0,93–0,99 (análisis 2.1). Queda pedir POA |
| **Sin variables fisiológicas ni órdenes de limitación del esfuerzo terapéutico** | El ajuste no ve la gravedad al ingreso más allá de los diagnósticos | — | Declararlo. Es el techo del modelo («el techo lo fijan las 40 variables», Fase 4) |
| **Mortalidad solo intrahospitalaria** | Un hospital que da el alta antes, o deriva, «exporta» muertes | Los traslados salientes y la hospitalización domiciliaria quedan censurados | Mortalidad a 30 días con el Registro Civil (trabajo futuro) y tasas de contexto (enmienda 02, E2.6a) |
| **Deriva de codificación** | Más diagnósticos secundarios con los años: de 4,16 a 5,89 por egreso entre 2019 y 2024 | Índice de exhaustividad por hospital de 0,65 a 1,50 | Sin ninguna comorbilidad, el indicador resiste en 2023 (ρ 0,909) pero no en 2024 (ρ 0,879; 8 hospitales cambian de zona, análisis 2.4). La codificación sí influye en el O/E de mortalidad |
| **Paradoja del diagnóstico secundario** | En los fallecidos se registran menos enfermedades crónicas | HTA no complicada OR 0,89; hipotiroidismo OR 0,91 (Fase 2) | Declararlo |
| **Sin contraparte clínica** | Nadie validó clínicamente las reglas de cohorte ni las variables de proceso | Decisión 12 de la Fase 1, pendiente | La enmienda 02 adopta la opción conservadora para EX03 (MDC 14 completo fuera) |
| **Solo adultos; sin pediatría ni neonatología** | El indicador no cubre esos servicios | Cohorte B (18 años o más) y EX02 | Declararlo |
| **Población obstétrica arbitraria (EX03)** | «MDC 14 sin Elixhauser» no equivale a parto sin complicación | Hallazgo H-07 | Excluir el MDC 14 completo no cambia el indicador: en 2023 ningún hospital cambia de zona, en 2024 uno (E2.2). Salen 46 muertes obstétricas en 2019–2023. Se puede adoptar en una versión 2 |
| **Alta voluntaria y fuga en la cohorte de estadía** | Estadías cortadas por decisión del paciente | Cerca del 2 % de E (H-08) | Sacarlas no cambia el indicador: ningún hospital cambia de zona, ρ 0,998 (E2.1) |

### 1.2 Modelo

| Limitación | Qué implica | Evidencia | Qué se hizo / trabajo futuro |
|---|---|---|---|
| **Variables condicionales** (tipo de ingreso, procedencia, nivel de cuidado al ingreso) | Pueden medir capacidad del hospital además de riesgo del paciente | **El O/E de mortalidad no resiste quitarlas**: ρ 0,891 en 2023 y 0,802 en 2024, con 5 y 6 hospitales que cambian de zona (2.3). En estadía resiste (ρ 0,96) | Es la limitación más importante para la lectura por hospital. Se informa que el indicador es sensible a esta decisión |
| **Recalibración con un solo año** | La deriva siguió después de recalibrar | O/E global 0,979 (mortalidad) y 0,957 (estadía) en 2024; 0,905 y 0,946 en el diseño anidado | Re-basar cada año en el despliegue. En 2024 no cambia la proporción de hospitales fuera (análisis 3.3) |
| **Estadía topada en 53 días** | El tope oculta el 10,7 % de los días | ρ 0,995 del O/E con y sin tope (Fase 4) | Indicador complementario de cola (trabajo futuro) |
| **Sesgo de supervivencia en estadía** | Un hospital con más muertes saca estadías largas del denominador | Correlación O/E mortalidad–estadía +0,25 (auditoría) | Declararlo; leer los dos indicadores juntos |
| **Ajuste de estadía modesto** | R² ≈ 0,40 | RMSE 7,01 | Declararlo |
| **Detención temprana sobre el pliegue de validación** | Optimismo leve a favor de los modelos de árboles | H-14 | Con la logística, el indicador casi no cambia: 94–100 % en la misma zona, ρ 0,98 (2.5) |

### 1.3 Indicador

| Limitación | Qué implica | Evidencia | Qué se hizo / trabajo futuro |
|---|---|---|---|
| **El embudo winsorizado deja fuera a hospitales «normales»** | El 15,9 % fuera no es por sí solo señal de mal ajuste | Con la variación real, el procedimiento deja fuera 11,6–13,2 % de hospitales sin diferencias; sin winsorizar, cerca del 5 % (3.4c) | Informar el embudo sin winsorizar como sensibilidad (3.4a: 8,7 % fuera en mortalidad 2024). Modelo jerárquico como trabajo futuro |
| **El O/E de estadía depende del tamaño del hospital** | Los hospitales pequeños y nuevos salen «mejor de lo esperado» | O/E 0,85–0,86 en el tercil de menos egresos; 0,79–0,83 en los hospitales con menos de 2 años en GRD (3.6). Con grupos por volumen, los hospitales fuera bajan del 15,9 % al 5,8 % (3.1b); con grupos DEIS, solo al 11,6 % (E2.4a) | Presentar la estadía por grupo de pares de volumen, o junto al volumen |
| **Institutos monográficos en el embudo general** | En 2024 los 3 quedan fuera en mortalidad (2 en señal baja) | 3.1a: sacarlos no mueve al resto (97–100 %) | Informarlos aparte |
| **Un único año de prueba, y no ciego** | 2024 se evaluó antes con un prototipo (I-01), y el profesor ratificó la apertura después (I-10) | Incidentes I-01 a I-10 | Enmienda 01 con diseño anidado (2023). Validar en 2025 con sello criptográfico (plan 4.5) |
| **Estabilidad entre años moderada** | El orden de los hospitales cambia algo de un año a otro | ρ 0,821 (mortalidad) y 0,882 (estadía); 8 de 11 y 6 de 9 alertas repetidas (3.5) | Leer las alertas como persistentes cuando se repiten |
| **Los códigos no anonimizan** | `Hospitales.csv` y el DEIS traducen los códigos a nombres | H-18 | La protección es no rankear y presentar el indicador como cribado interno |

### 1.4 Proceso

| Limitación | Evidencia | Estado |
|---|---|---|
| Exposiciones de 2024 antes de la apertura y cambios de código posteriores | I-01 a I-07 | Declaradas |
| La enmienda 01 no es ciega: sus análisis se exploraron antes | I-08; las 40 comparaciones son idénticas | Declarado; se informaron todos los resultados |
| Valores reales en el repositorio público | I-09 (prueba y tablas del prototipo) e I-11 (conteos nacionales por complemento) | I-11 corregido (tablas regeneradas); historial público pendiente (I-09) |

---

## Parte 2. Las 15 preguntas del comité, con la evidencia nueva

**1. ¿Cómo sabe que el O/E mide desempeño y no tipo de hospital?**
No del todo, y por eso no lo presento como medida de calidad.

- **Institutos monográficos.** Los 3 quedan fuera en mortalidad 2024. Sacarlos no cambia la zona del resto (97–100 %
  en la misma zona, análisis 3.1a).
- **Tamaño del hospital en estadía.** El O/E de estadía sí depende del tamaño: es 0,85–0,86 en el tercil más chico frente
  a 0,96–0,98 en los otros (3.6). Con grupos de pares por volumen, los hospitales fuera de los límites bajan del
  15,9 % al 5,8 % (3.1b).
- **Persistencia.** Las alertas se repiten entre años (ρ 0,82 y 0,88). Eso indica diferencias sistemáticas, no
  necesariamente de calidad.

Propongo leer la estadía por grupo de pares de volumen. Con los grupos de complejidad del DEIS (alta, mediana e
institutos), los hospitales de estadía fuera de los límites solo bajan al 11,6 % (enmienda 02, E2.4a): el tamaño pesa
más que la complejidad.

**2. Sin marcador POA, ¿por qué usar coagulopatía o trastornos hidroelectrolíticos?**
Es el estándar de los indicadores tipo HSMR. Medí qué pasa sin las 5 categorías sospechosas: el PR AUC baja 0,028,
pero el indicador casi no cambia (91–100 % de los hospitales en la misma zona, ρ 0,93–0,99; análisis 2.1).

Sin ninguna comorbilidad, el indicador resiste en 2023 (ρ 0,909), pero no en 2024 (ρ 0,879; análisis 2.4). La
codificación importa, y lo declaro.

**3. ¿2024 fue realmente un año no visto?**
No. Un prototipo anterior se evaluó en 2024 el 27-09-2026 (I-01), y el profesor guía ratificó la apertura después de
conocer los resultados (I-10). El pipeline definitivo sí leyó 2024 una sola vez, después de pre-registrar el
protocolo. Todas las exposiciones están en el registro de incidentes.

Por eso los análisis de robustez usan como resultado el diseño anidado, que prueba en 2023. La validación ciega
queda para 2025, con el desenlace cifrado con la clave del profesor.

**4. ¿Por qué cambió la decisión sobre los traslados entrantes, y cuándo?**
La Fase 3 los incluyó por un error de alineamiento. Lo detecté en la Fase 5 y la versión 4 de la Fase 1 es una
enmienda posterior a la apertura. Sin traslados entrantes, el indicador resiste: 93,8–100 % en la misma zona, ρ
0,97–0,99 (análisis 2.6).

**5. ¿Por qué no hizo la sensibilidad sin 2020–2021 que prometió?**
Ya está hecha (análisis 2.2). El indicador no cambia: ρ de 0,989 a 0,998. En el diseño anidado solo queda 2019 para
entrenar y el O/E global baja a 0,830: la deriva crece, pero el orden de los hospitales se mantiene.

**6. Con 15,9 % de hospitales fuera, ¿el ajuste es malo?**
No necesariamente. Simulé el procedimiento con la variación real entre hospitales: aunque ningún hospital sea
anómalo, deja fuera entre el 11,6 y el 13,2 % (análisis 3.4c). Lo demás son diferencias persistentes (ρ 0,82 y 0,88
entre años) y casuística no comparable: institutos y hospitales pequeños.

**7. ¿Por qué Spiegelhalter aditivo y winsorizado, y no un modelo jerárquico como el de CMS?**
Por transparencia y porque tiene precedente (CQC y SHMI). Sin winsorizar, quedan fuera el 8,7 % en mortalidad 2024;
con truncamiento al estilo SHMI, el 21,7 %. En los dos casos, más del 91 % de los hospitales conserva su zona
(3.4a y 3.4b). El porcentaje fuera depende del estimador, y por eso lo informo junto a la validación. El modelo
jerárquico queda como trabajo futuro.

**8. LightGBM mejora solo 0,033 de PR AUC: ¿vale la complejidad?**
El indicador no depende del algoritmo. Con la logística, el 94–100 % de los hospitales queda en la misma zona, con ρ
0,98 en las dos tareas (análisis 2.5). La regla de parsimonia, escrita antes, eligió LightGBM por su mejor
predicción. Para el indicador, cualquiera de los dos sirve.

**9. ¿Por qué recalibrar con un solo año y no re-basar cada año?**
El protocolo prohibía ajustar con 2024. Re-basar no cambia la conclusión: 92–99 % de los hospitales en la misma zona
(análisis 3.3), y en 2024 la proporción fuera incluso sube un poco en mortalidad. En el despliegue propongo re-basar
cada año.

**10. ¿El tope de 53 días no oculta justo las estadías que importan?**
Oculta el 10,7 % de los días, pero el O/E con y sin tope tiene ρ 0,995. Propongo un indicador complementario de cola
por hospital.

**11. ¿Qué pasa con los pacientes derivados o enviados a hospitalización domiciliaria?**
Quedan censurados. Los hospitales con alerta baja de estadía derivan el doble que el resto (auditoría, H-06). La
enmienda 02 publica, junto al O/E, la tasa de hospitalización domiciliaria y de alta voluntaria por hospital
(E2.6a). Los hospitales con más altas a domicilio tienen un O/E de estadía algo más bajo (ρ −0,14 a −0,26). Sacar
el alta voluntaria y la fuga de la estadía no cambia el indicador (E2.1). La mortalidad a 30 días con el Registro
Civil es trabajo futuro.

**12. ¿Se puede reidentificar a alguien con lo que publica?**
No hay registros individuales, y las celdas de 1 a 9 casos se suprimen. Encontré y declaré tres problemas:

- la prueba de privacidad usaba valores reales (I-09, corregido);
- había tablas de un prototipo con celdas pequeñas en el historial público (I-09);
- había conteos nacionales reconstruibles por complemento en las tablas por subgrupo (I-11, corregido: los grupos
  pequeños ahora se combinan).

El escáner nuevo intenta reconstruir conteos combinando tablas, y hoy no encuentra ninguno en lo publicado. Los
códigos de hospital no anonimizan: la protección es no rankear.

**13. ¿Por qué incluir el nivel de cuidado al ingreso si depende de las camas UCI?**
Es la pregunta más difícil, y la evidencia no me favorece. Sin las 4 variables condicionales, el O/E de mortalidad
no resiste: ρ 0,891 en 2023 y 0,802 en 2024 (análisis 2.3). Esas variables explican parte de las diferencias entre
hospitales, y no puedo separar cuánto es riesgo del paciente y cuánto es capacidad del hospital.

Lo presento como limitación principal. Una versión 2 del indicador debería mostrar las dos variantes, o justificar
clínicamente cada variable.

**14. ¿Quién validó clínicamente las reglas?**
Nadie todavía, y es una limitación. Las reglas se basan en literatura y datos. Para EX03 probé la opción
conservadora, que excluye el MDC 14 completo, y el indicador no cambia: en 2023 ningún hospital cambia de zona y en
2024 uno solo (enmienda 02, E2.2). Pedí revisión de la procedencia y del nivel de cuidado.

**15. ¿Qué pasa si un hospital codifica más paliativos (Z51.5)?**
Su esperado sube y su O/E baja. Cerca del 3 % de las muertes de 2019 a 2023 tiene Z51.5 en algún diagnóstico. Lo
medí en la enmienda 02:

- la tasa de Z51.5 por hospital no se relaciona con el O/E (|ρ| de 0,02 o menos, E2.6a);
- sin esos episodios, el indicador resiste: 96,9 y 95,7 % de los hospitales en la misma zona (E2.6b).

Con estos datos no hay señal de que la codificación de paliativos explique las diferencias.

---

## Nota sobre Hospitales.csv (plan 5.4)

Las versiones v3 y v4 de las Fases 1 y 2 decían que `Hospitales.csv` traía el nivel de complejidad. Solo trae código
y nombre. Los informes v5 ya no lo afirman, y la Fase 1 v5 advierte que el archivo traduce los códigos a nombres.
El nivel de complejidad ahora sale del catálogo DEIS (`fase3_preparacion/config/tipo_establecimiento.csv`): 52
hospitales generales de alta complejidad, 14 de mediana, 3 institutos y 3 pediátricos.
