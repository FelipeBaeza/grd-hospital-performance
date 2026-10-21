// Cliente mínimo de la API. En producción /api lo reenvía nginx; en desarrollo, Vite.
const BASE = '/api'

async function pedir(ruta, opciones) {
  const r = await fetch(BASE + ruta, opciones)
  const cuerpo = await r.json().catch(() => ({}))
  if (!r.ok) {
    const e = new Error(cuerpo.detail || 'Error al consultar la API')
    e.detalle = cuerpo.detalle
    throw e
  }
  return cuerpo
}

export const api = {
  salud: () => pedir('/salud'),
  metodologia: () => pedir('/metodologia'),
  criterios: () => pedir('/criterios'),
  hospitales: () => pedir('/hospitales'),
  indicadores: (tarea) => pedir(`/indicadores/${tarea}`),
  ficha: (codigo) => pedir(`/hospitales/${codigo}`),
  subgrupos: (tarea) => pedir(`/subgrupos/${tarea}`),
  calibracion: (tarea) => pedir(`/calibracion/${tarea}`),
  demoCatalogo: () => pedir('/demo/catalogo'),
  demoRiesgo: (perfil) => pedir('/demo/riesgo', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(perfil) }),
}

export const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? '—' : Number(x).toLocaleString('es-CL', { minimumFractionDigits: d, maximumFractionDigits: d }))
export const fmtEntero = (x) => (x === null || x === undefined ? '—' : Number(x).toLocaleString('es-CL'))
export const pct = (x, d = 1) => (x === null || x === undefined ? '—' : (x * 100).toLocaleString('es-CL', { minimumFractionDigits: d, maximumFractionDigits: d }) + ' %')

export const NOMBRE_ZONA = {
  'dentro de lo esperado': 'dentro de lo esperado',
  'alerta alta': 'sobre el límite del 95 %',
  'alerta baja': 'bajo el límite del 95 %',
  'señal alta': 'sobre el límite del 99,8 %',
  'señal baja': 'bajo el límite del 99,8 %',
}
