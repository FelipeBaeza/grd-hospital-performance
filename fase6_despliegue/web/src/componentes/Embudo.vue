<script setup>
// Gráfico de embudo: cada punto es un hospital; el eje horizontal es el volumen esperado (escala log) y el vertical el O/E.
// Las curvas marcan hasta dónde llega una diferencia explicable por el azar y la sobredispersión. No ordena ni compara hospitales.
import { computed, ref } from 'vue'
import { fmt, fmtEntero, NOMBRE_ZONA } from '../api.js'

const props = defineProps({ datos: { type: Object, required: true }, etiquetaX: { type: String, default: 'volumen esperado del hospital (escala log)' }, seleccionado: { type: String, default: '' } })
const emit = defineEmits(['elegir'])

const W = 900, H = 470, M = { l: 56, r: 18, t: 16, b: 52 }
const c = computed(() => props.datos.curva)
const visibles = computed(() => props.datos.puntos.filter((p) => p.oe !== null))
const ocultos = computed(() => props.datos.puntos.filter((p) => p.oe === null).length)
const xmin = computed(() => Math.min(...c.value.esperado)), xmax = computed(() => Math.max(...c.value.esperado))
const ymax = computed(() => Math.max(1.6, ...visibles.value.map((p) => p.oe)) * 1.04)
const ymin = computed(() => Math.min(0.2, ...visibles.value.map((p) => p.oe)) * 0.95)
const sx = (v) => M.l + ((Math.log(v) - Math.log(xmin.value)) / (Math.log(xmax.value) - Math.log(xmin.value))) * (W - M.l - M.r)
const sy = (v) => M.t + (1 - (v - ymin.value) / (ymax.value - ymin.value)) * (H - M.t - M.b)
const ruta = (xs, ys) => xs.map((x, i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(' ')
const curvas = computed(() => [
  { d: ruta(c.value.esperado, c.value.inf_95), cls: 'c95' }, { d: ruta(c.value.esperado, c.value.sup_95), cls: 'c95' },
  { d: ruta(c.value.esperado, c.value.inf_998), cls: 'c998' }, { d: ruta(c.value.esperado, c.value.sup_998), cls: 'c998' },
])
const ticksY = computed(() => { const t = []; for (let v = Math.ceil(ymin.value * 5) / 5; v <= ymax.value; v += 0.2) t.push(+v.toFixed(2)); return t })
const ticksX = computed(() => { const t = []; for (let e = Math.floor(Math.log10(xmin.value)); e <= Math.ceil(Math.log10(xmax.value)); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= xmin.value && v <= xmax.value) t.push(v) } return t })
const color = (z) => (z === 'dentro de lo esperado' ? '#2e5597' : z.startsWith('señal') ? '#9f3b38' : '#dd690a')
const activo = ref(null)
const tip = computed(() => (activo.value ? visibles.value.find((p) => p.cod_hospital === activo.value) : null))
</script>

<template>
  <div>
    <svg :viewBox="`0 0 ${W} ${H}`" width="100%" role="img" :aria-label="`Gráfico de embudo con ${visibles.length} hospitales`" style="max-width: 100%; height: auto">
      <rect :x="M.l" :y="M.t" :width="W - M.l - M.r" :height="H - M.t - M.b" fill="#fff" stroke="#d8dee9" />
      <g v-for="v in ticksY" :key="'y' + v">
        <line :x1="M.l" :x2="W - M.r" :y1="sy(v)" :y2="sy(v)" stroke="#eceff4" />
        <text :x="M.l - 8" :y="sy(v) + 4" text-anchor="end" font-size="11" fill="#5b6472">{{ fmt(v, 1) }}</text>
      </g>
      <g v-for="v in ticksX" :key="'x' + v">
        <line :x1="sx(v)" :x2="sx(v)" :y1="M.t" :y2="H - M.b" stroke="#f1f3f7" />
        <text :x="sx(v)" :y="H - M.b + 16" text-anchor="middle" font-size="11" fill="#5b6472">{{ fmtEntero(v) }}</text>
      </g>
      <line :x1="M.l" :x2="W - M.r" :y1="sy(1)" :y2="sy(1)" stroke="#1f3864" stroke-width="1.5" />
      <path v-for="(k, i) in curvas" :key="i" :d="k.d" fill="none" :stroke="k.cls === 'c95' ? '#dd690a' : '#dd690a'" :stroke-width="k.cls === 'c95' ? 2 : 1.8" :stroke-dasharray="k.cls === 'c95' ? '7 5' : '2 4'" />
      <g>
        <circle v-for="p in visibles" :key="p.cod_hospital" :cx="sx(p.esperado)" :cy="sy(p.oe)" :r="p.cod_hospital === seleccionado ? 8 : 5.5" :fill="color(p.zona)" :stroke="p.cod_hospital === seleccionado ? '#1f3864' : '#fff'" :stroke-width="p.cod_hospital === seleccionado ? 3 : 1.2"
          tabindex="0" role="button" :aria-label="`Hospital ${p.cod_hospital}: O/E ${fmt(p.oe)}, ${NOMBRE_ZONA[p.zona]}`" style="cursor: pointer"
          @mouseenter="activo = p.cod_hospital" @mouseleave="activo = null" @focus="activo = p.cod_hospital" @blur="activo = null" @click="emit('elegir', p.cod_hospital)" @keydown.enter="emit('elegir', p.cod_hospital)" />
      </g>
      <text :x="(M.l + W - M.r) / 2" :y="H - 8" text-anchor="middle" font-size="12" fill="#1d2433">{{ etiquetaX }}</text>
      <text :x="14" :y="(M.t + H - M.b) / 2" text-anchor="middle" font-size="12" fill="#1d2433" :transform="`rotate(-90 14 ${(M.t + H - M.b) / 2})`">O/E (observado / esperado)</text>
      <g v-if="tip" pointer-events="none">
        <rect :x="Math.min(sx(tip.esperado) + 10, W - 190)" :y="Math.max(sy(tip.oe) - 54, M.t + 2)" width="180" height="48" rx="6" fill="#1f3864" opacity="0.95" />
        <text :x="Math.min(sx(tip.esperado) + 10, W - 190) + 10" :y="Math.max(sy(tip.oe) - 54, M.t + 2) + 19" font-size="12.5" fill="#fff" font-weight="700">Hospital {{ tip.cod_hospital }}</text>
        <text :x="Math.min(sx(tip.esperado) + 10, W - 190) + 10" :y="Math.max(sy(tip.oe) - 54, M.t + 2) + 37" font-size="11.5" fill="#fff">O/E {{ fmt(tip.oe) }} · {{ NOMBRE_ZONA[tip.zona] }}</text>
      </g>
    </svg>
    <div class="leyenda">
      <span><i style="background: #2e5597"></i>dentro de lo esperado</span>
      <span><i style="background: #dd690a"></i>fuera del límite del 95 %</span>
      <span><i style="background: #9f3b38"></i>fuera del límite del 99,8 %</span>
      <span>línea azul: O/E = 1 · curva discontinua: límite 95 % · curva punteada: límite 99,8 %</span>
    </div>
    <p v-if="ocultos" class="nota">{{ ocultos }} hospital(es) con menos de 10 eventos observados no se grafican por privacidad; su zona se informa en la tabla.</p>
  </div>
</template>
