<script setup>
import { ref, watch, onMounted } from 'vue'
import { api, fmt, fmtEntero, pct } from '../api.js'
import Advertencias from '../componentes/Advertencias.vue'
import ChipZona from '../componentes/ChipZona.vue'

const props = defineProps({ codigo: { type: String, default: '' } })
const lista = ref([]), ficha = ref(null), error = ref(''), elegido = ref(props.codigo)
const NOMBRE = { mortalidad: 'Mortalidad en el hospital', estadia: 'Estadía de los sobrevivientes' }
const cargar = async () => {
  ficha.value = null; error.value = ''
  if (!elegido.value) return
  try { ficha.value = await api.ficha(elegido.value) } catch (e) { error.value = e.message }
}
onMounted(async () => { try { lista.value = (await api.hospitales()).hospitales } catch (e) { error.value = e.message } await cargar() })
watch(() => props.codigo, (v) => { elegido.value = v; cargar() })
const ir = () => { location.hash = elegido.value ? `#/hospital/${elegido.value}` : '#/hospital' }
</script>

<template>
  <h1>Ficha de hospital</h1>
  <div class="tarjeta">
    <label for="h">Código del hospital</label>
    <select id="h" v-model="elegido" @change="ir" style="max-width: 320px"><option value="">— elija un código —</option><option v-for="h in lista" :key="h" :value="h">{{ h }}</option></select>
    <p class="nota">Solo aparecen los hospitales con al menos 500 egresos analizables en 2024. Los hospitales se identifican por código.</p>
  </div>
  <p v-if="error" class="caja peligro">{{ error }}</p>

  <template v-if="ficha">
    <h2>Hospital {{ ficha.cod_hospital }} · {{ ficha.anio }}</h2>
    <div class="grid g2">
      <div v-for="(f, t) in ficha.indicadores" :key="t" class="tarjeta">
        <h3>{{ NOMBRE[t] }}</h3>
        <p><span style="font-size: 1.8rem; color: var(--azul); font-weight: 700">{{ f.oe === null ? 'no se publica' : fmt(f.oe) }}</span> <span class="nota">O/E</span></p>
        <p class="peq" v-if="f.oe !== null">Intervalo de confianza del 95 %: {{ fmt(f.ic_inf) }} a {{ fmt(f.ic_sup) }}</p>
        <p class="peq" v-else>Este hospital tiene menos de 10 eventos observados: el conteo y el O/E no se publican para proteger la privacidad.</p>
        <p><ChipZona :zona="f.zona" /></p>
        <table><tbody>
          <tr><td>Egresos analizados</td><td>{{ fmtEntero(f.n_egresos) }}</td></tr>
          <tr><td>{{ t === 'mortalidad' ? 'Muertes' : 'Días' }} observados</td><td>{{ f.observado_pub === '<10' ? 'menos de 10' : fmtEntero(Number(f.observado_pub)) }}</td></tr>
          <tr><td>{{ t === 'mortalidad' ? 'Muertes' : 'Días' }} esperados</td><td>{{ fmt(f.esperado, 1) }}</td></tr>
          <tr><td>Límites del 95 % para este volumen</td><td>{{ fmt(f.lim_inf_95) }} a {{ fmt(f.lim_sup_95) }}</td></tr>
        </tbody></table>
      </div>
    </div>
    <div class="caja">
      <strong class="t">Cómo leer esto</strong>
      {{ ficha.lectura }} Una zona «dentro de lo esperado» significa que la diferencia con 1 puede deberse al azar; una zona fuera de los límites invita a revisar procesos, registro de diagnósticos y derivaciones, no a sacar conclusiones sobre el hospital.
    </div>

    <div class="tarjeta">
      <h3>Contexto que ayuda a interpretar el O/E</h3>
      <div class="grid g4">
        <div class="kpi"><b>{{ fmt(ficha.contexto.indice_vs_2023) }}</b><span>índice de exhaustividad de la codificación (1 = lo esperado según 2023; más alto = más diagnósticos registrados)</span></div>
        <div class="kpi"><b>{{ fmt(ficha.contexto.indice_relativo_al_anio) }}</b><span>el mismo índice respecto del promedio de los hospitales de 2024</span></div>
        <div class="kpi"><b>{{ pct(ficha.contexto.tasa_traslado_saliente) }}</b><span>de sus egresos son traslados a otro hospital (alta anticipada posible)</span></div>
        <div class="kpi"><b>{{ pct(ficha.contexto.proporcion_traslado_entrante) }}</b><span>de sus pacientes llegaron trasladados desde otro hospital</span></div>
      </div>
    </div>
    <Advertencias :lista="ficha.advertencias" />
  </template>
</template>
