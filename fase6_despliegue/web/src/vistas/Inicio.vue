<script setup>
import { ref, onMounted } from 'vue'
import { api, fmtEntero } from '../api.js'
defineEmits(['ir'])
const m = ref(null), crit = ref(null), error = ref('')
onMounted(async () => {
  try { [m.value, crit.value] = await Promise.all([api.metodologia(), api.criterios()]) } catch (e) { error.value = e.message }
})
// Ilustración con datos simulados: un hospital cuyo O/E se sale de las curvas
const ej = [{ x: 30, y: 1.45, c: '#dd690a' }, { x: 90, y: 0.9, c: '#2e5597' }, { x: 250, y: 1.06, c: '#2e5597' }, { x: 600, y: 1.2, c: '#dd690a' }, { x: 180, y: 0.62, c: '#9f3b38' }]
const sx = (v) => 40 + (Math.log(v) - Math.log(20)) / (Math.log(900) - Math.log(20)) * 520
const sy = (v) => 150 - (v - 0.4) / (1.8 - 0.4) * 130
const cur = (z, s) => Array.from({ length: 30 }, (_, i) => { const x = Math.exp(Math.log(20) + i / 29 * (Math.log(900) - Math.log(20))); return `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(1 + s * z * Math.sqrt(1 / x + 0.016)).toFixed(1)}` }).join(' ')
</script>

<template>
  <h1>¿Se alejan los hospitales de lo esperado?</h1>
  <p v-if="m">{{ m.que_es }}</p>
  <p v-if="error" class="caja peligro">No se pudo cargar la información: {{ error }}</p>

  <div v-if="crit && m" class="grid g4">
    <div class="kpi"><b>{{ crit.criterios.filter((c) => c.veredicto !== 'no cumple').length }} de {{ crit.criterios.length }}</b><span>criterios de aceptación cumplidos por los modelos en {{ m.anio_evaluado }}</span></div>
    <div class="kpi"><b>{{ fmtEntero(m.cohortes.n_mortalidad) }}</b><span>egresos de adultos en el análisis de mortalidad</span></div>
    <div class="kpi"><b>{{ fmtEntero(m.cohortes.n_estadia) }}</b><span>egresos en el análisis de estadía</span></div>
    <div class="kpi"><b>≥ {{ m.volumen.min_egresos }}</b><span>egresos por hospital para mostrarlo; solo códigos, sin nombres</span></div>
  </div>

  <div class="caja ok">
    <strong class="t">Cómo leer un gráfico de embudo</strong>
    <p>Cada punto es un hospital. Hacia la derecha, más pacientes. La línea azul (O/E = 1) es «ocurrió lo esperado». Las curvas marcan hasta dónde llega una diferencia que el azar puede explicar: con pocos pacientes, el O/E se mueve mucho sin que signifique nada; con muchos, una diferencia pequeña ya es llamativa.</p>
    <p><b>Un punto fuera de las curvas no es un veredicto</b>: es un motivo para revisar con más detalle qué pasa en ese hospital.</p>
  </div>
  <div class="tarjeta">
    <svg viewBox="0 0 600 190" width="100%" role="img" aria-label="Ejemplo ilustrativo de gráfico de embudo con datos simulados" style="max-width: 640px">
      <rect x="40" y="20" width="520" height="130" fill="#fff" stroke="#d8dee9" />
      <line x1="40" x2="560" :y1="sy(1)" :y2="sy(1)" stroke="#1f3864" stroke-width="1.5" />
      <path :d="cur(1.96, 1)" fill="none" stroke="#dd690a" stroke-dasharray="7 5" stroke-width="2" /><path :d="cur(1.96, -1)" fill="none" stroke="#dd690a" stroke-dasharray="7 5" stroke-width="2" />
      <path :d="cur(3.09, 1)" fill="none" stroke="#dd690a" stroke-dasharray="2 4" stroke-width="1.8" /><path :d="cur(3.09, -1)" fill="none" stroke="#dd690a" stroke-dasharray="2 4" stroke-width="1.8" />
      <circle v-for="(p, i) in ej" :key="i" :cx="sx(p.x)" :cy="sy(p.y)" r="6" :fill="p.c" stroke="#fff" />
      <text x="300" y="176" text-anchor="middle" font-size="11" fill="#5b6472">más pacientes →</text>
      <text x="44" y="14" font-size="11" fill="#9f3b38" font-style="italic">DATOS SIMULADOS: solo ilustran cómo se lee el gráfico</text>
    </svg>
  </div>

  <div class="grid g2">
    <div class="caja aviso" v-if="m">
      <strong class="t">Lo que este indicador NO es</strong>
      <ul style="margin: 4px 0 0 18px; padding: 0"><li v-for="t in m.no_es" :key="t">{{ t }}</li></ul>
    </div>
    <div class="tarjeta">
      <h3>¿Por dónde seguir?</h3>
      <p><button class="btn" @click="$emit('ir', 'indicadores')">Ver los gráficos de embudo</button></p>
      <p><button class="btn sec" @click="$emit('ir', 'hospital')">Consultar la ficha de un hospital</button></p>
      <p><button class="btn sec" @click="$emit('ir', 'demo')">Probar la demo pedagógica del modelo</button></p>
    </div>
  </div>
</template>
