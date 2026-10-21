<script setup>
import { ref, onMounted } from 'vue'
import { api, fmt } from '../api.js'
import Advertencias from '../componentes/Advertencias.vue'
const m = ref(null), crit = ref(null), error = ref('')
onMounted(async () => { try { [m.value, crit.value] = await Promise.all([api.metodologia(), api.criterios()]) } catch (e) { error.value = e.message } })
const V = { cumple: 'cumple', 'no cumple': 'no cumple', 'cumple con reservas': 'cumple con reservas' }
</script>

<template>
  <h1>Metodología y límites</h1>
  <p v-if="error" class="caja peligro">{{ error }}</p>
  <template v-if="m">
    <div class="tarjeta"><h3>Qué es</h3><p>{{ m.que_es }}</p></div>
    <div class="caja aviso"><strong class="t">Qué no es</strong><ul style="margin: 4px 0 0 18px; padding: 0"><li v-for="t in m.no_es" :key="t">{{ t }}</li></ul></div>
    <div class="tarjeta"><h3>Datos y reglas de publicación</h3><p>{{ m.limites_datos }}</p>
      <p class="peq">Volumen mínimo: {{ m.volumen.min_egresos }} egresos y {{ m.volumen.min_esperado }} eventos esperados. Celda mínima: {{ m.volumen.celda_minima }} casos.</p></div>
    <div class="tarjeta" v-if="crit">
      <h3>Desempeño de los modelos en {{ m.anio_evaluado }} (evaluación única)</h3>
      <div class="scroll"><table><thead><tr><th>Tarea</th><th>Criterio</th><th>Valor</th><th>Umbral</th><th>Resultado</th></tr></thead>
        <tbody><tr v-for="c in crit.criterios" :key="c.tarea + c.criterio"><td>{{ c.tarea }}</td><td>{{ c.criterio }}</td><td>{{ fmt(c.valor, 3) }} <span class="nota" v-if="c.ic_inf !== null">({{ fmt(c.ic_inf, 3) }} a {{ fmt(c.ic_sup, 3) }})</span></td><td>{{ c.umbral }}</td><td>{{ V[c.veredicto] || c.veredicto }}</td></tr></tbody></table></div>
    </div>
    <Advertencias :lista="m.advertencias" />
  </template>
</template>
