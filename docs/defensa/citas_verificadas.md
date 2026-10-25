# Citas del informe de seminario verificadas (etapa 5)

Revisión del 22-10-2026 de las tres citas que la auditoría marcó como mal atribuidas (informe de seminario,
`docs/00_propuesta_seminario/informe.pdf`, §2.1, §2.2, Tabla 1 y referencias). Cada fuente se comparó con la
página del editor o con un registro bibliográfico. Los informes v5 no citan estos tres trabajos. La corrección
importa para el capítulo de estado del arte de la tesis, que parte del informe.

## Resumen

| Cita | Qué dice el informe | Qué dice la fuente | Problema |
| --- | --- | --- | --- |
| Chen y cols. (2023) | Que los indicadores GRD «permiten una comparación más ajustada por casuística» (§2.1) | Metaanálisis del **efecto del pago por GRD** en la calidad de la atención hospitalaria | Atribución equivocada. Además, la referencia tiene **mal los autores** y el título |
| Zeleke y cols. (2023) | Que los árboles son sólidos para **predecir la mortalidad** (§2.1 y §2.2) | Predice **estadía prolongada** (más de 6 días) y **excluye a los fallecidos** | Atribución equivocada para mortalidad. Además, Gradient Boosting y la regresión logística rinden casi igual |
| Guzmán-Muñoz y cols. (2025) | Que los árboles son sólidos para predecir la mortalidad (§2.1); en la Tabla 1, «estadística tradicional» | Kaplan–Meier y regresión de Cox; ningún modelo de aprendizaje automático | El texto contradice la Tabla 1. La referencia tiene **mal el DOI**, el número y los autores desde el tercero |

## 1. Chen y cols. (2023)

**Referencia correcta:**

Chen, Y.-J., Zhang, X.-Y., Yan, J.-Q., Tang, X., Qian, M.-C. y Ying, X.-H. (2023). Impact of Diagnosis-Related
Groups on inpatient quality of health care: A systematic review and meta-analysis. *INQUIRY: The Journal of
Health Care Organization, Provision, and Financing*, 60, 1–16. https://doi.org/10.1177/00469580231167011

**En el informe** los autores son «Chen, Y., Cryan, C., Yu, K., Bhatt, D., y Bhatt, J.» y el título dice
«Diagnosis-Related Group–Based payment». El DOI, la revista, el volumen y el año son correctos. Los autores
del informe no aparecen en ninguna de las dos fuentes consultadas.

**Contenido.** El estudio revisa 29 estudios con 36.214.219 pacientes y mide el efecto de **pagar por GRD**. Con
ese pago, la estadía baja (DME agrupada −0,25; IC 95 % −0,37 a −0,12). No hay efecto significativo en la
mortalidad (RR 0,91; IC 95 % 0,72 a 1,15) ni en la readmisión a 30 días (RR 0,79; IC 95 % 0,62 a 1,01). No
estudia el uso de los GRD para comparar hospitales.

**La Tabla 1 es correcta** («Meta-análisis GRD; baja estancia, riesgo estable; impacto de implementación»).
Lo equivocado es la frase de §2.1.

**Redacción propuesta para §2.1:**

> Los GRD se usan como sistema de pago y de clasificación de la casuística. Chen y cols. (2023), con 29 estudios
> y más de 36 millones de pacientes, muestran que el pago por GRD acorta la estadía sin un efecto significativo
> en la mortalidad ni en la readmisión. Ese efecto en la estadía es una razón más para ajustar por riesgo antes
> de comparar hospitales: las diferencias de estadía pueden reflejar incentivos de pago, no solo la gravedad.

Para respaldar que los indicadores GRD sirven para comparar hospitales queda Zapata (2018), ya citada.

## 2. Zeleke y cols. (2023)

**Referencia (correcta en el informe):**

Zeleke, A. J., Palumbo, P., Tubertini, P., Miglio, R. y Chiari, L. (2023). Machine learning-based prediction of
hospital prolonged length of stay admission at emergency department: A Gradient Boosting algorithm analysis.
*Frontiers in Artificial Intelligence*, 6, 1179226. https://doi.org/10.3389/frai.2023.1179226

Conviene agregar el número de artículo (1179226), que falta en el informe.

**Contenido.** El estudio incluye 12.858 adultos ingresados por urgencia en un hospital universitario de
Bolonia en 2022. Define estadía prolongada como más de 6 días y **excluye a los pacientes que fallecieron**.
Gradient Boosting obtiene un AUC de 0,754 y la regresión logística, uno de 0,752.

**Problemas:**

- §2.1 lo cita para la mortalidad. El estudio no predice la mortalidad.
- §2.2 lo cita junto a Mekhaldi y cols. (2020) para la «estadía prolongada». Ese uso es correcto, pero la
  diferencia con la regresión logística es de 0,002 de AUC. El estudio no muestra que los árboles superen a la
  logística, y eso coincide con lo que encontró este proyecto: la logística deja al 94–100 % de los hospitales
  en la misma zona que LightGBM, con ρ 0,98 (enmienda 01, análisis 2.5).

**Redacción propuesta.** En §2.1, sacar a Zeleke de la frase sobre mortalidad y llevarlo a la frase sobre
estadía:

> … para la estimación de días de estadía, estos algoritmos han mostrado buena capacidad predictiva (Mekhaldi y
> cols., 2020; Almeida y cols., 2024), aunque con ventajas pequeñas sobre la regresión logística cuando la
> estadía se trata como un desenlace binario (Zeleke y cols., 2023).

## 3. Guzmán-Muñoz y cols. (2025)

**Referencia correcta:**

Guzmán-Muñoz, E., Concha-Cisternas, Y., Vásquez-Muñoz, M., Yáñez-Sepúlveda, R., Núñez-Espinosa, C., Bittelman
Saporte, S., Nemtala Urquiza, F. y Morales Araneda, R. (2025). In-hospital mortality among 40,253 older adults
with hip fracture: Survival outcomes and multivariate analysis in a Chilean cohort. *Journal of Clinical
Medicine*, 14(21), 7717. https://doi.org/10.3390/jcm14217717

**En el informe:**

- El DOI es `10.3390/jcm14247717`; el correcto es `10.3390/jcm14217717`, número 21.
- El título omite «and Multivariate Analysis».
- Los autores desde el tercero («Lira, C., y Valdés-Badilla, P.») no coinciden con la página del editor.

La lista completa de autores solo se pudo confirmar en la página del editor. Conviene revisarla en el PDF antes
de citarla.

**Contenido.** El estudio analiza 40.253 adultos mayores con fractura de cadera en Chile. Usa curvas de
Kaplan–Meier, pruebas log-rank y regresión de Cox. **No usa aprendizaje automático.**

**Problemas:**

- La Tabla 1 lo clasifica bien («estadística tradicional»).
- §2.1 lo usa para respaldar los ensambles de árboles en mortalidad, y eso es incorrecto.

**Redacción propuesta.** En §2.1, sacarlo de la frase sobre árboles. Se puede citar como antecedente chileno
de ajuste por riesgo de la mortalidad intrahospitalaria con datos administrativos:

> En Chile, Guzmán-Muñoz y cols. (2025) estudiaron la mortalidad intrahospitalaria de 40.253 adultos mayores con
> fractura de cadera con modelos de supervivencia (Kaplan–Meier y Cox).

**Si se quiere una cita chilena de aprendizaje automático para mortalidad,** el mismo grupo publicó después un
estudio con modelos supervisados: Guzmán-Muñoz y cols. (2026), *Supervised machine learning-based prediction of
in-hospital mortality following hip fracture in older adults*, *Diagnostics*, 16(4), 612,
https://doi.org/10.3390/diagnostics16040612. Antes de citarlo hay que leerlo, para confirmar qué modelos usa y
cómo se comparan con la logística. Ese trabajo no se revisó aquí.

## 4. La frase de §2.1 que queda sin respaldo

Con los cambios anteriores, la frase «para predecir la mortalidad intrahospitalaria […] la literatura confirma
su solidez al modelar el riesgo clínico» se queda sin citas. Se puede respaldar con Rajkomar y cols. (2018),
ya citado, aunque usa aprendizaje profundo y no árboles. Otra opción es reformularla según lo que encontró el
proyecto:

> Para la mortalidad intrahospitalaria, los ensambles de árboles se usan cada vez más, pero la regresión
> logística sigue siendo una referencia difícil de superar en calibración. En este proyecto, LightGBM superó a
> la logística en PR AUC (+0,032), pero las dos dejaron al 94–100 % de los hospitales en la misma zona del embudo
> (enmienda 01, análisis 2.5).

## 5. DIPRES y la Comisión Nacional de Productividad (revisión del profesor guía, 24-10-2026)

El profesor guía pidió citar las métricas de eficiencia de la DIPRES y de la Comisión Nacional de Productividad
que menciona la Fase 1. Se verificaron las dos referencias del informe de seminario:

| Cita del seminario | Qué se encontró | Qué se hace |
| --- | --- | --- |
| Dirección de Presupuestos (2021). *Evaluación de la eficiencia hospitalaria: indicadores de mortalidad y estancia por GRD* | No se encontró un documento con ese título. Lo que sí existe es Urbina y Sugg (2021), *Análisis de la eficiencia de los hospitales públicos de alta complejidad en Chile entre 2014 y 2018* (DIPRES, junio de 2021): mide la **eficiencia de costos** de 65 hospitales con una frontera estocástica, no la mortalidad ni la estadía | La Fase 1 v6 cita Urbina y Sugg (2021) y describe lo que mide. El título del seminario debe corregirse en el informe final |
| Comisión Nacional de Productividad (2020). *Uso eficiente de quirófanos electivos y gestión de lista de espera quirúrgica no GES* | El estudio existe (resumen ejecutivo de enero de 2020, *Uso eficiente de quirófanos electivos y gestión de lista de espera quirúrgica*). Compara el uso de los pabellones entre hospitales públicos | Se cita tal cual en la Fase 1 v6 |

## Fuentes consultadas

- Chen y cols. (2023): [registro DOAJ](https://doaj.org/article/eb7bee0de8b4433e99ecb003a37980f2),
  [PDF del artículo](https://hitdept.mums.ac.ir/images/hitdept/OlfatiM544/Journal/farvardin1403/10.1177_00469580231167011.pdf)
  y [registro del catálogo UPI](https://oalib-perpustakaan.upi.edu/Record/doaj_eb7bee0de8b4433e99ecb003a37980f2).
- Zeleke y cols. (2023): [Frontiers in Artificial Intelligence](https://www.frontiersin.org/articles/10.3389/frai.2023.1179226/full).
- Guzmán-Muñoz y cols. (2025): [MDPI, Journal of Clinical Medicine](https://www.mdpi.com/2077-0383/14/21/7717).
- Guzmán-Muñoz y cols. (2026): [registro de la Universidad Europea](https://abacus.universidadeuropea.com/signposting/describedby/2c6ed882-5b06-451f-9e81-2822a159be37).
- Urbina y Sugg (2021): [Diario Financiero, 5 de julio de 2021](https://www.df.cl/economia-y-politica/macro/informe-de-la-dipres-mide-que-tan-eficientes-son-los-gastos-de-los).
- Comisión Nacional de Productividad (2020): [resumen ejecutivo](https://cnep.cl/wp-content/uploads/2025/05/Uso-eficiente-de-quirofanos-resumen-ejecutivo.pdf).
