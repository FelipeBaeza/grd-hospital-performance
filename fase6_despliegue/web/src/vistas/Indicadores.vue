<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { api, fmt, fmtEntero, pct } from '../api.js'
import Embudo from '../componentes/Embudo.vue'
import Advertencias from '../componentes/Advertencias.vue'
import ChipZona from '../componentes/ChipZona.vue'

const tarea = ref('mortalidad')
const adv = ref([])
const datos = ref(null), error = ref(''), filtro = ref('todos'), cargando = ref(false)
const cargar = async () => { cargando.value = true; error.value = ''; try { datos.value = await api.indicadores(tarea.value) } catch (e) { error.value = e.message } finally { cargando.value = false } }
onMounted(async () => { try { adv.value = (await api.metodologia()).advertencias } catch (e) { /* se muestra sin advertencias extra */ } await cargar() }); watch(tarea, cargar)

const unidad = computed(() => (tarea.value === 'mortalidad' ? 'muertes' : 'días'))
const filas = computed(() => (datos.value ? datos.value.puntos.filter((p) => filtro.value === 'todos' || p.zona !== 'dentro de lo esperado') : []))  // siempre en orden de código
const elegir = (cod) => { location.hash = `#/hospital/${cod}` }
</script>

<template>
  <h1>Indicadores por hospital · {{ datos?.anio || 2024 }}</h1>
  <div class="pestanas" role="tablist" aria-label="Tarea">
    <button class="btn sec" :class="{ act: tarea === 'mortalidad' }" role="tab" :aria-selected="tarea === 'mortalidad'" @click="tarea = 'mortalidad'">Mortalidad en el hospital</button>
    <button class="btn sec" :class="{ act: tarea === 'estadia' }" role="tab" :aria-selected="tarea === 'estadia'" @click="tarea = 'estadia'">Estadía de los sobrevivientes</button>
  </div>
  <p class="nota">{{ tarea === 'mortalidad' ? 'O/E = muertes observadas ÷ muertes esperadas según el perfil de los pacientes.' : 'O/E = días de estadía observados ÷ días esperados, en pacientes que sobreviven (con tope de 53 días).' }}</p>
  <p v-if="error" class="caja peligro">No se pudo cargar: {{ error }}</p>
  <p v-if="cargando">Cargando…</p>

  <template v-if="datos">
    <div class="tarjeta">
      <Embudo :datos="datos" :etiqueta-x="tarea === 'mortalidad' ? 'muertes esperadas del hospital (escala log)' : 'días esperados del hospital (escala log)'" @elegir="elegir" />
      <p class="peq"><b>{{ datos.resumen.n_elegibles }}</b> hospitales con volumen suficiente; <b>{{ pct(datos.resumen.pct_fuera_95) }}</b> queda fuera del límite del 95 % (por azar se esperaría cerca de 5 %): los hospitales se distinguen más de lo que explican los datos disponibles. <span class="nota">Haga clic en un punto para ver la ficha.</span></p>
    </div>
    <Advertencias :lista="adv" />

    <div class="tarjeta">
      <h2>Tabla por código de hospital</h2>
      <p class="nota">Orden por código. No hay orden por desempeño ni numeración de posiciones: no es un ranking.</p>
      <label for="f">Mostrar</label>
      <select id="f" v-model="filtro" style="max-width: 320px"><option value="todos">todos los hospitales</option><option value="fuera">solo los que quedan fuera del límite del 95 %</option></select>
      <div class="scroll">
        <table>
          <thead><tr><th>Código</th><th>Egresos</th><th>{{ unidad === 'muertes' ? 'Muertes' : 'Días' }} observados</th><th>Esperados</th><th>O/E</th><th>Intervalo 95 %</th><th>Zona</th><th></th></tr></thead>
          <tbody>
            <tr v-for="p in filas" :key="p.cod_hospital">
              <td><b>{{ p.cod_hospital }}</b></td><td>{{ fmtEntero(p.n_egresos) }}</td><td>{{ p.observado_pub === '<10' ? 'menos de 10' : fmtEntero(Number(p.observado_pub)) }}</td><td>{{ fmt(p.esperado, 1) }}</td>
              <td>{{ p.oe === null ? 'no se publica' : fmt(p.oe) }}</td><td>{{ p.oe === null ? '—' : `${fmt(p.ic_inf)} a ${fmt(p.ic_sup)}` }}</td><td><ChipZona :zona="p.zona" /></td>
              <td><a :href="`#/hospital/${p.cod_hospital}`">ficha</a></td>
            </tr>
          </tbody>
        </table>
      </div>
      <p class="nota" v-if="datos.puntos.some((p) => p.oe === null)">Cuando un hospital tiene menos de 10 eventos observados, no se publica el conteo ni el O/E (revelarían la cifra); se informa solo su zona.</p>
    </div>
  </template>
</template>
