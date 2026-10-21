<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import Inicio from './vistas/Inicio.vue'
import Indicadores from './vistas/Indicadores.vue'
import Ficha from './vistas/Ficha.vue'
import Demo from './vistas/Demo.vue'
import Metodologia from './vistas/Metodologia.vue'

const RUTAS = [
  { id: 'inicio', texto: 'Inicio y cómo leer' },
  { id: 'indicadores', texto: 'Indicadores' },
  { id: 'hospital', texto: 'Ficha de hospital' },
  { id: 'demo', texto: 'Demo pedagógica' },
  { id: 'metodologia', texto: 'Metodología y límites' },
]
const hash = ref(location.hash || '#/inicio')
const actualizar = () => { hash.value = location.hash || '#/inicio'; window.scrollTo(0, 0) }
onMounted(() => window.addEventListener('hashchange', actualizar))
onBeforeUnmount(() => window.removeEventListener('hashchange', actualizar))

const partes = computed(() => hash.value.replace(/^#\//, '').split('/'))
const vista = computed(() => (RUTAS.some((r) => r.id === partes.value[0]) ? partes.value[0] : 'inicio'))
const codigo = computed(() => partes.value[1] || '')
const ir = (id) => { location.hash = `#/${id}` }
</script>

<template>
  <header class="cabecera">
    <div class="contenedor">
      <div class="marca">
        <b>Indicadores hospitalarios GRD · observado / esperado</b>
        <span>Prototipo académico · datos de 2024 · hospitales identificados solo por código</span>
      </div>
      <nav aria-label="Secciones">
        <button v-for="r in RUTAS" :key="r.id" :class="{ act: vista === r.id }" :aria-current="vista === r.id ? 'page' : undefined" @click="ir(r.id)">{{ r.texto }}</button>
      </nav>
    </div>
  </header>
  <main class="contenedor">
    <Inicio v-if="vista === 'inicio'" @ir="ir" />
    <Indicadores v-else-if="vista === 'indicadores'" />
    <Ficha v-else-if="vista === 'hospital'" :codigo="codigo" />
    <Demo v-else-if="vista === 'demo'" />
    <Metodologia v-else />
  </main>
</template>
