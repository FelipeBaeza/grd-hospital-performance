<script setup>
import { ref, computed, onMounted } from 'vue'
import { api, fmt, pct } from '../api.js'

const cat = ref(null), res = ref(null), error = ref(''), enviando = ref(false), errores = ref([])
const perfil = ref({ sexo: '', edad: 65, tipo_ingreso: '', tipo_procedencia: '', diagnostico_principal: '', ingreso_fin_de_semana: false, cuidado_intensivo_al_ingreso: false, comorbilidades: [] })
const busca = ref('')
onMounted(async () => {
  try {
    cat.value = await api.demoCatalogo()
    const c = cat.value
    perfil.value.sexo = c.sexo[0]; perfil.value.tipo_ingreso = c.tipo_ingreso.find((s) => s.startsWith('URGENCIA')) || c.tipo_ingreso[0]; perfil.value.tipo_procedencia = c.tipo_procedencia[0].valor; perfil.value.diagnostico_principal = c.diagnosticos[0].codigo
  } catch (e) { error.value = e.message }
})
const diagFiltrados = computed(() => {
  if (!cat.value) return []
  const q = busca.value.trim().toLowerCase()
  const l = q ? cat.value.diagnosticos.filter((d) => d.codigo.toLowerCase().includes(q) || d.nombre.toLowerCase().includes(q)) : cat.value.diagnosticos
  return l.slice(0, 300)
})
const calcular = async () => {
  enviando.value = true; error.value = ''; errores.value = []; res.value = null
  try { res.value = await api.demoRiesgo({ ...perfil.value, edad: Number(perfil.value.edad) }) }
  catch (e) { error.value = e.message; errores.value = e.detalle || [] }
  finally { enviando.value = false }
}
const top = (t) => (res.value ? res.value[t].contribuciones.filter((c) => Math.abs(c.efecto) > 1e-6).slice(0, 8) : [])
const maxAbs = (t) => Math.max(0.0001, ...top(t).map((c) => Math.abs(c.efecto)))
const ancho = (t, c) => `${(Math.abs(c.efecto) / maxAbs(t)) * 100}%`
const NOM_PROC = { mortalidad: 'Riesgo de morir en el hospital', estadia: 'Días de estadía esperados' }
</script>

<template>
  <h1>Demo pedagógica: cómo «piensa» el modelo</h1>
  <div class="caja aviso" v-if="cat">
    <strong class="t">Perfil inventado, uso solo educativo</strong>
    <ul style="margin: 4px 0 0 18px; padding: 0"><li v-for="a in cat.avisos" :key="a">{{ a }}</li></ul>
  </div>
  <p v-if="error && !errores.length" class="caja peligro">{{ error }}</p>

  <form v-if="cat" class="tarjeta" @submit.prevent="calcular">
    <div class="grid g3">
      <div><label for="sx">Sexo</label><select id="sx" v-model="perfil.sexo"><option v-for="s in cat.sexo" :key="s">{{ s }}</option></select></div>
      <div><label for="ed">Edad (18 a 110)</label><input id="ed" type="number" min="18" max="110" v-model="perfil.edad" /></div>
      <div><label for="ti">Tipo de ingreso</label><select id="ti" v-model="perfil.tipo_ingreso"><option v-for="s in cat.tipo_ingreso" :key="s">{{ s }}</option></select></div>
      <div><label for="tp">Procedencia</label><select id="tp" v-model="perfil.tipo_procedencia"><option v-for="s in cat.tipo_procedencia" :key="s.valor" :value="s.valor">{{ s.valor }}</option></select></div>
      <div style="grid-column: span 2"><label for="bq">Diagnóstico principal (CIE-10, 3 caracteres)</label>
        <input id="bq" v-model="busca" placeholder="Buscar por código o nombre…" style="margin-bottom: 4px" />
        <select id="dg" v-model="perfil.diagnostico_principal" aria-label="Diagnóstico principal"><option v-for="d in diagFiltrados" :key="d.codigo" :value="d.codigo">{{ d.codigo }} · {{ d.nombre }}</option></select></div>
    </div>
    <p><label class="chk"><input type="checkbox" v-model="perfil.ingreso_fin_de_semana" /> Ingreso en fin de semana</label>
      <label class="chk"><input type="checkbox" v-model="perfil.cuidado_intensivo_al_ingreso" /> Cuidado intensivo al ingreso</label></p>
    <details>
      <summary>Comorbilidades (Elixhauser) · {{ perfil.comorbilidades.length }} seleccionadas</summary>
      <div class="grid g3"><label v-for="c in cat.comorbilidades" :key="c.id" class="chk"><input type="checkbox" :value="c.id" v-model="perfil.comorbilidades" /> {{ c.nombre }}</label></div>
    </details>
    <p><button class="btn" type="submit" :disabled="enviando">{{ enviando ? 'Calculando…' : 'Calcular' }}</button></p>
    <ul v-if="errores.length" class="caja peligro" style="margin-left: 0"><li v-for="e in errores" :key="e.campo">{{ e.campo }}: {{ e.mensaje }}</li></ul>
  </form>

  <template v-if="res">
    <div class="grid g2">
      <div class="kpi"><b>{{ pct(res.mortalidad.probabilidad) }}</b><span>probabilidad de morir en el hospital (promedio de los pacientes: {{ pct(res.mortalidad.probabilidad_base) }})</span></div>
      <div class="kpi"><b>{{ fmt(res.estadia.dias_esperados, 1) }} días</b><span>estadía esperada si sobrevive (promedio: {{ fmt(res.estadia.dias_base, 1) }})</span></div>
    </div>
    <div class="grid g2">
      <div v-for="t in ['mortalidad', 'estadia']" :key="t" class="tarjeta">
        <h3>¿Qué empuja el resultado? · {{ NOM_PROC[t] }}</h3>
        <p class="nota">Escala {{ res[t].escala }}. Rojo: aumenta el valor. Azul: lo reduce.</p>
        <div v-for="c in top(t)" :key="c.variable" class="barra-fila">
          <div class="peq"><b>{{ c.etiqueta }}</b> <span class="nota">{{ c.valor }}</span></div>
          <div class="pista"><div class="relleno" :class="c.efecto > 0 ? 'mas' : 'menos'" :style="{ width: ancho(t, c) }"></div><span class="v">{{ c.efecto > 0 ? '+' : '−' }}{{ fmt(Math.abs(c.efecto), 2) }}</span></div>
        </div>
      </div>
    </div>
    <p class="nota">{{ res.lectura }}</p>
  </template>
</template>

<style scoped>
.chk { display: inline-flex; gap: 6px; align-items: center; margin-right: 16px; font-weight: 400 }
.barra-fila { margin-bottom: 8px }
.pista { display: flex; align-items: center; gap: 8px }
.relleno { height: 14px; border-radius: 3px; min-width: 2px }
.relleno.mas { background: #9f3b38 }
.relleno.menos { background: #2e5597 }
.v { font-size: 0.8rem; color: #5b6472 }
details { margin: 8px 0 }
summary { cursor: pointer; font-weight: 600 }
</style>
