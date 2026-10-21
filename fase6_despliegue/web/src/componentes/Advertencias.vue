<script setup>
defineProps({ lista: { type: Array, default: () => [] }, titulo: { type: String, default: 'Advertencias que acompañan a este indicador' } })
</script>
<template>
  <div v-if="lista.length" class="caja aviso" role="note">
    <strong class="t">{{ titulo }}</strong>
    <ul style="margin: 4px 0 0 18px; padding: 0">
      <li v-for="a in lista" :key="a.id"><b>{{ a.titulo }}.</b> {{ a.texto }}</li>
    </ul>
  </div>
</template>
