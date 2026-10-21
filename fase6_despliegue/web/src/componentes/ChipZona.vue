<script setup>
import { computed } from 'vue'
import { NOMBRE_ZONA } from '../api.js'
const props = defineProps({ zona: String })
const clase = computed(() => (props.zona === 'dentro de lo esperado' ? 'ok' : props.zona?.startsWith('señal') ? 'mal' : 'warn'))
</script>
<template><span class="chip" :class="clase">{{ NOMBRE_ZONA[zona] || zona }}</span></template>
