#!/usr/bin/env bash
# ordenar_proyecto.sh — Ordena el repositorio de la tesis de forma reversible con git.
#
#   bash _orden_propuesto/ordenar_proyecto.sh             simulación: muestra lo que haría, no cambia nada
#   bash _orden_propuesto/ordenar_proyecto.sh --aplicar   ejecuta
#
# Qué hace, en orden:
#   1. Congela el estado actual en un commit y la etiqueta v1-congelado (hoy las fases 2 a 6 no están en git).
#   2. Guarda el manifiesto SHA-256 de ese estado en docs/historial/MANIFIESTO_v1-congelado.sha256.
#   3. Agrega los informes v5, la guía y los README; mueve versiones antiguas a docs/historial/ y artefactos de
#      proceso a archivo/; borra copias idénticas, cachés y bloqueos de LibreOffice; y hace un segundo commit.
# No toca el código de las fases, los catálogos de la raíz, data seminario/ ni venv/. No hace push.
#
# Para deshacer el orden:   git reset --hard v1-congelado
set -euo pipefail

APLICAR=0
[ "${1:-}" = "--aplicar" ] && APLICAR=1
ENTREGA="_orden_propuesto"
NUEVOS="$ENTREGA/nuevos"

run() { printf '  + %s\n' "$*"; if [ "$APLICAR" = 1 ]; then "$@"; fi; }
titulo() { printf '\n== %s\n' "$*"; }
falla() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- comprobaciones
[ -d .git ] && [ -d fase3_preparacion ] && [ -f fase4_modelado/modelos/manifiesto.json ] \
  || falla "ejecuta el script desde la raíz del proyecto (la carpeta Tesis)."
[ -d "$NUEVOS" ] || falla "no encuentro $NUEVOS: copia la carpeta _orden_propuesto completa a la raíz."
command -v git >/dev/null || falla "git no está instalado."
git config user.name >/dev/null && git config user.email >/dev/null \
  || falla "git no tiene nombre y correo configurados (git config --global user.name ... / user.email ...)."
if git rev-parse -q --verify refs/tags/v1-congelado >/dev/null; then
  falla "la etiqueta v1-congelado ya existe: el orden ya se aplicó. Para deshacerlo: git reset --hard v1-congelado"
fi
if [ -e .git/MERGE_HEAD ] || [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; then falla "hay un merge o rebase en curso."; fi

# Archivos de docs/ que el código de la Fase 2 lee: no se mueven y deben seguir existiendo.
PROTEGIDOS=(
  "docs/fase1_comprension_dominio/decisiones_pendientes.csv"
  "docs/fase1_comprension_dominio/reglas_variables.csv"
  "docs/fase1_comprension_dominio/tipoalta_roles.csv"
  "docs/fase1_comprension_dominio/matriz_trazabilidad.csv"
  "docs/fase1_comprension_dominio/config/fase1_spec.json"
  "docs/fase1_comprension_dominio/Fase_1__Comprensión_del_dominio_-_Categorias_de_Selección.csv"
  "docs/fase1_comprension_dominio/Fase_1__Comprensión_del_dominio_-_Decisiones_Metodológicas.csv"
  "docs/fase1_comprension_dominio/Fase_1__Comprensión_del_dominio_-_Reglas_del_Modelo.csv"
  "docs/fase1_comprension_dominio/Fase_1__Comprensión_del_dominio_-_Requisitos_del_Modelo.csv"
  "docs/fase2_comprension_datos/Fase_2__Diccionario_de_Base_de_Datos.csv"
  "docs/fase2_comprension_datos/Fase_2__Problemas_de_Calidad_de_Datos.csv"
  "docs/fase2_comprension_datos/Fase_2__Variables_Derivadas.csv"
)
for f in "${PROTEGIDOS[@]}"; do [ -f "$f" ] || falla "falta un archivo que lee la Fase 2: $f"; done

if [ "$APLICAR" = 0 ]; then
  echo "SIMULACIÓN: no se cambiará nada. Para aplicar: bash $ENTREGA/ordenar_proyecto.sh --aplicar"
fi

# ---------------------------------------------------------------- paso 1: congelar
titulo "Paso 1. Congelar el estado actual (commit + etiqueta v1-congelado)"
run cp "$ENTREGA/gitignore" .gitignore
if [ "$APLICAR" = 1 ]; then
  git add -A
  echo "  archivos en el commit de congelamiento: $(git diff --cached --name-only | wc -l)"
  git commit -q -m "Congela el estado del proyecto antes de ordenarlo (fases 2 a 6, informes v4)" \
                -m "Incluye todo lo que hoy no estaba en git. Registra también el retiro de la versión antigua (src/, web/, reports/...)."
  git tag -a v1-congelado -m "Estado del proyecto antes de ordenarlo (21-10-2026)"
else
  cp "$ENTREGA/gitignore" "$ENTREGA/.gitignore.simulacion"
  n=$(git -c core.excludesFile="$ENTREGA/.gitignore.simulacion" add -A --dry-run | wc -l)
  rm -f "$ENTREGA/.gitignore.simulacion"
  echo "  se registrarían ~$n cambios (archivos nuevos y el retiro de la versión antigua)"
fi

# ---------------------------------------------------------------- paso 2: manifiesto
titulo "Paso 2. Manifiesto SHA-256 del estado congelado"
run mkdir -p docs/historial
if [ "$APLICAR" = 1 ]; then
  git ls-files -z | xargs -0 sha256sum > docs/historial/MANIFIESTO_v1-congelado.sha256
  echo "  $(wc -l < docs/historial/MANIFIESTO_v1-congelado.sha256) archivos con huella"
else
  echo "  + git ls-files -z | xargs -0 sha256sum > docs/historial/MANIFIESTO_v1-congelado.sha256"
fi

# ---------------------------------------------------------------- paso 3: ordenar
titulo "Paso 3a. Informes v5, guía y README nuevos"
while IFS= read -r -d '' f; do
  rel="${f#"$NUEVOS"/}"
  run mkdir -p "$(dirname "$rel")"
  run cp "$f" "$rel"
done < <(find "$NUEVOS" -type f -print0 | sort -z)

mover() {  # mover <origen> <carpeta destino>
  if [ -e "$1" ]; then run mkdir -p "$2"; run git mv "$1" "$2/"; else echo "  (no existe, se omite) $1"; fi
}
borrar_si_igual() {  # borrar_si_igual <copia> <original que se conserva>
  if [ -e "$1" ] && [ -e "$2" ] && cmp -s "$1" "$2"; then run git rm -q "$1"
  elif [ -e "$1" ]; then echo "  (distinto del original, se archiva) $1"; mover "$1" "archivo/duplicados_revisar"; fi
}

titulo "Paso 3b. Versiones anteriores de los informes → docs/historial/"
mover docs/fase1_comprension_dominio/Informe_Fase1_v3_actualizado.docx docs/historial/fase1
mover docs/fase1_comprension_dominio/Informe_Fase1_v4_consenso.docx docs/historial/fase1
for f in Fase2_Comprension_Datos_Analisis_Revisado.pdf Fase2_Comprension_Datos_Base_Completa.docx \
         Fase2_Comprension_Datos_Base_Completa.pdf Fase2_Comprension_Datos_Base_Completa_v4_consenso.docx; do
  mover "docs/fase2_comprension_datos/$f" docs/historial/fase2
done
mover docs/fase3_preparacion_datos/Informe_Fase3_Preparacion_Datos.docx docs/historial/fase3
mover docs/fase3_preparacion_datos/Informe_Fase3_Preparacion_Datos_v4_consenso.docx docs/historial/fase3
mover docs/fase4_modelado/Informe_Fase4_Modelado.docx docs/historial/fase4
mover docs/fase4_modelado/Informe_Fase4_Modelado_v4_consenso.docx docs/historial/fase4
mover docs/fase5_evaluacion/Informe_Fase5_Evaluacion.docx docs/historial/fase5
mover docs/fase5_evaluacion/Informe_Fase5_Preparacion.docx docs/historial/fase5
mover docs/informe.pdf docs/00_propuesta_seminario

titulo "Paso 3c. Artefactos del proceso → archivo/"
borrar_si_igual tasks/t10t11_design_review.md  tasks/t10t11_design_review_r3_aceptada.md
borrar_si_igual tasks/t10t11_design_verdict.json tasks/t10t11_design_verdict_r3.json
borrar_si_igual tasks/t10t11_diseno.md          tasks/t10t11_diseno_v3_aprobado.md
for f in fase2_brief_t10_t11.txt fase2_brief_t12_t13.txt t10t11_design_review_r3_aceptada.md \
         t10t11_design_verdict_r3.json t10t11_diseno_v3_aprobado.md t10t11_enmiendas.txt; do
  mover "tasks/$f" archivo/tareas_fase2
done
if [ -d fase2_eda/.agents/tasks ]; then
  while IFS= read -r -d '' f; do mover "$f" archivo/revisiones_fase2; done < <(find fase2_eda/.agents/tasks -type f -print0 | sort -z)
fi
mover fase2_eda/cambios_baja_memoria.patch archivo/parches_fase2_aplicados
mover fase2_eda/cambios_todos_los_datos.patch archivo/parches_fase2_aplicados
mover docs/fase1_comprension_dominio/update_fase1.py archivo/generadores_informes
mover docs/fase2_comprension_datos/generate_fase2_docx.py archivo/generadores_informes

titulo "Paso 3d. Copias idénticas (se conserva la que lee el código)"
for p in decisiones_pendientes matriz_trazabilidad reglas_variables tipoalta_roles; do
  borrar_si_igual "docs/fase1_comprension_dominio/config/$p.csv" "docs/fase1_comprension_dominio/$p.csv"
done
borrar_si_igual docs/fase2_comprension_datos/tipoalta_roles.csv docs/fase1_comprension_dominio/tipoalta_roles.csv

titulo "Paso 3e. Cachés de Python, bloqueos de LibreOffice y carpetas vacías"
while IFS= read -r -d '' d; do run rm -rf "$d"; done < <(find . -path ./venv -prune -o -path ./.git -prune -o -type d -name __pycache__ -print0)
while IFS= read -r -d '' f; do run rm -f "$f"; done < <(find . -path ./venv -prune -o -path ./.git -prune -o -type f -name '.~lock.*#' -print0)
for d in fase2_eda/.agents/tasks fase2_eda/.agents tasks .agents; do
  if [ -d "$d" ]; then run rmdir "$d" 2>/dev/null || echo "  (no vacía, se deja) $d"; fi
done

titulo "Paso 3f. Las guías de cada fase apuntan a los informes v5"
run sed -i 's#docs/fase3_preparacion_datos/Informe_Fase3_Preparacion_Datos.docx#docs/fase3_preparacion_datos/Informe_Fase3_v5.docx#' fase3_preparacion/GUIA_FASE3.md
run sed -i 's#docs/fase4_modelado/Informe_Fase4_Modelado.docx#docs/fase4_modelado/Informe_Fase4_v5.docx#' fase4_modelado/GUIA_FASE4.md
if [ "$APLICAR" = 1 ]; then
  grep -q "Informe_Fase5_v5" fase5_evaluacion/GUIA_FASE5.md || printf '\nInforme explicado: `docs/fase5_evaluacion/Informe_Fase5_v5.docx`.\n' >> fase5_evaluacion/GUIA_FASE5.md
  grep -q "Informe_Fase2_v5" fase2_eda/GUIA_BASE_COMPLETA.md || printf '\nInforme explicado: `docs/fase2_comprension_datos/Informe_Fase2_v5.docx`.\n' >> fase2_eda/GUIA_BASE_COMPLETA.md
else
  echo "  + añadir la referencia al informe v5 en GUIA_FASE5.md y GUIA_BASE_COMPLETA.md"
fi

titulo "Paso 3g. Guardar el script en archivo/ y hacer el commit"
run cp "$ENTREGA/ordenar_proyecto.sh" archivo/ordenar_proyecto.sh
for f in "${PROTEGIDOS[@]}"; do [ -f "$f" ] || falla "se perdió un archivo que lee la Fase 2: $f (deshacer con git reset --hard v1-congelado)"; done
if [ "$APLICAR" = 1 ]; then
  git add -A
  git commit -q -m "Ordena el proyecto: informes v5, guía, historial de versiones y archivo de proceso" \
                -m "Sin cambios en el código de las fases ni en sus salidas. Deshacer: git reset --hard v1-congelado"
  rm -rf "$ENTREGA"
  titulo "Listo"
  git log --oneline -3
  echo
  echo "Revisa la estructura (README.md y docs/README.md). Para deshacer todo: git reset --hard v1-congelado"
  echo "Nada se subió a GitHub. Antes de hacer push, revisa el plan de corrección (docs/auditoria/), etapa 4.1."
else
  titulo "Fin de la simulación. Nada cambió."
fi
