// JavaScript para la aplicación web de Desempeño Hospitalario GRD

let globalBenchmarks = [];
let globalSummary = {};

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  loadSummary();
  loadBenchmarks(2024);
  loadComparison();
  loadModels();
  initCalculator();
});

// Navegación de pestañas
function initTabs() {
  const btns = document.querySelectorAll(".tab-btn");
  btns.forEach(btn => {
    btn.addEventListener("click", () => {
      btns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");

      if (targetId === "tab-dashboard") {
        renderQuadrantCanvas(globalBenchmarks);
      }
    });
  });
}

// Cargar resumen de datos y KPIs
async function loadSummary() {
  try {
    const res = await fetch("/api/summary");
    const data = await res.json();
    globalSummary = data;

    document.getElementById("kpi-total-egresos").textContent = data.poblacion_inicial.toLocaleString();
    document.getElementById("kpi-cohorte-mortalidad").textContent = data.cohorte_dura_mortalidad.toLocaleString();
    document.getElementById("kpi-cohorte-estancia").textContent = data.cohorte_estancia.toLocaleString();
  } catch (err) {
    console.error("Error cargando summary:", err);
  }
}

// Cargar Benchmarks Hospitalarios
async function loadBenchmarks(year = 2024) {
  try {
    const res = await fetch(`/api/benchmarks?year=${year}`);
    const data = await res.json();
    globalBenchmarks = data;

    renderHospitalTable(data);
    renderQuadrantCanvas(data);
    populateServiceFilter(data);
  } catch (err) {
    console.error("Error cargando benchmarks:", err);
  }
}

// Llenar selector de Servicios de Salud
function populateServiceFilter(data) {
  const select = document.getElementById("filter-service");
  if (!select) return;
  const services = [...new Set(data.map(d => d.SERVICIO_SALUD))].sort();
  select.innerHTML = '<option value="">Todos los Servicios de Salud</option>';
  services.forEach(s => {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    select.appendChild(opt);
  });
}

// Renderizar tabla de hospitales
function renderHospitalTable(data) {
  const tbody = document.getElementById("hospitals-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  data.forEach(h => {
    const tr = document.createElement("tr");

    const badgeClass = `badge-${(h.clasificacion_desempeno || "PROMEDIO_ESPERADO").toLowerCase()}`;
    const badgeText = (h.clasificacion_desempeno || "PROMEDIO").replace(/_/g, " ");

    tr.innerHTML = `
      <td><strong>${h.COD_HOSPITAL}</strong></td>
      <td>${h.SERVICIO_SALUD}</td>
      <td>${h.n_episodios_totales.toLocaleString()}</td>
      <td><strong>${h.HSMR !== null ? h.HSMR.toFixed(1) : '-'}</strong> <span style="font-size:0.75rem; color:#64748b;">[${h.HSMR_IC95_INF || '-'}-${h.HSMR_IC95_SUP || '-'}]</span></td>
      <td><strong>${h.IEMC_ML !== null ? h.IEMC_ML.toFixed(3) : '-'}</strong></td>
      <td>${h.IEMC_FONASA !== null ? h.IEMC_FONASA.toFixed(3) : '-'}</td>
      <td><span class="badge ${badgeClass}">${badgeText}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// Filtros interactivos
document.addEventListener("input", e => {
  if (e.target.id === "filter-search" || e.target.id === "filter-service" || e.target.id === "filter-category") {
    applyFilters();
  }
});

function applyFilters() {
  const search = (document.getElementById("filter-search").value || "").toLowerCase();
  const service = document.getElementById("filter-service").value;
  const category = document.getElementById("filter-category").value;

  const filtered = globalBenchmarks.filter(h => {
    const matchSearch = h.COD_HOSPITAL.toString().includes(search) || h.SERVICIO_SALUD.toLowerCase().includes(search);
    const matchService = !service || h.SERVICIO_SALUD === service;
    const matchCat = !category || h.clasificacion_desempeno === category;
    return matchSearch && matchService && matchCat;
  });

  renderHospitalTable(filtered);
  renderQuadrantCanvas(filtered);
}

// Dibujar Cuadrante de Desempeño (Scatter Canvas)
function renderQuadrantCanvas(data) {
  const canvas = document.getElementById("quadrantChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  // Ajustar resolución
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = 420;

  const w = canvas.width;
  const h = canvas.height;
  const pad = 50;

  ctx.clearRect(0, 0, w, h);

  // Ejes: X = IEMC (0.6 a 1.4), Y = HSMR (40 a 160)
  const minX = 0.6, maxX = 1.4;
  const minY = 40, maxY = 150;

  function toX(val) {
    return pad + ((val - minX) / (maxX - minX)) * (w - 2 * pad);
  }
  function toY(val) {
    return h - pad - ((val - minY) / (maxY - minY)) * (h - 2 * pad);
  }

  // Líneas de corte de referencia (IEMC = 1.0, HSMR = 100)
  const refX = toX(1.0);
  const refY = toY(100.0);

  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);

  // Línea vertical
  ctx.beginPath();
  ctx.moveTo(refX, pad);
  ctx.lineTo(refX, h - pad);
  ctx.stroke();

  // Línea horizontal
  ctx.beginPath();
  ctx.moveTo(pad, refY);
  ctx.lineTo(w - pad, refY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Etiquetas de cuadrantes
  ctx.font = "10px Inter, sans-serif";
  ctx.fillStyle = "rgba(52, 211, 153, 0.4)";
  ctx.fillText("SOBRESALIENTE (Alta Supervivencia & Eficiencia)", pad + 10, h - pad - 10);

  ctx.fillStyle = "rgba(248, 113, 113, 0.4)";
  ctx.fillText("ALERTA CRÍTICA (Mayor Mortalidad & Estancia)", w - pad - 240, pad + 20);

  // Ejes labels
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText("IEMC Ajustado por ML (Eficiencia de Estancia) →", w / 2 - 120, h - 15);

  ctx.save();
  ctx.translate(15, h / 2 + 60);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("HSMR Ajustado (Mortalidad Estandarizada) →", 0, 0);
  ctx.restore();

  // Puntos de hospitales
  data.forEach(hosp => {
    if (hosp.IEMC_ML === null || hosp.HSMR === null) return;
    const px = toX(Math.max(minX, Math.min(maxX, hosp.IEMC_ML)));
    const py = toY(Math.max(minY, Math.min(maxY, hosp.HSMR)));

    let color = "#94a3b8";
    if (hosp.clasificacion_desempeno === "SOBRESALIENTE") color = "#10b981";
    else if (hosp.clasificacion_desempeno === "ALTA_EFICIENCIA") color = "#3b82f6";
    else if (hosp.clasificacion_desempeno === "ALERTA_ESTANCIA") color = "#f59e0b";
    else if (hosp.clasificacion_desempeno === "ALERTA_CRITICA") color = "#ef4444";
    else if (hosp.clasificacion_desempeno === "ALERTA_MORTALIDAD") color = "#f43f5e";

    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

// Cargar comparativa ML vs FONASA
async function loadComparison() {
  try {
    const res = await fetch("/api/comparison");
    const data = await res.json();
    const tbody = document.getElementById("comparison-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    data.slice(0, 15).forEach(row => {
      const tr = document.createElement("tr");
      const diff = row.diferencia_IEMC_ML_FONASA;
      const diffColor = diff > 0 ? "#f87171" : "#34d399";
      tr.innerHTML = `
        <td><strong>${row.COD_HOSPITAL}</strong></td>
        <td>${row.SERVICIO_SALUD}</td>
        <td>${row.n_episodios_totales.toLocaleString()}</td>
        <td><strong>${row.IEMC_ML.toFixed(3)}</strong></td>
        <td>${row.IEMC_FONASA.toFixed(3)}</td>
        <td style="color:${diffColor}; font-weight:600;">${diff > 0 ? "+" : ""}${diff.toFixed(3)}</td>
        <td><span class="badge badge-${(row.clasificacion_desempeno || 'promedio').toLowerCase()}">${row.clasificacion_desempeno}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error cargando comparativa:", err);
  }
}

// Cargar modelos y auditoría
async function loadModels() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();

    if (data.mortality && data.mortality.metricas_test_2024) {
      const m = data.mortality.metricas_test_2024;
      document.getElementById("kpi-auc").textContent = m.roc_auc.toFixed(4);
      document.getElementById("model-auc-val").textContent = m.roc_auc.toFixed(4);
      document.getElementById("model-pr-val").textContent = m.pr_auc.toFixed(4);
      document.getElementById("model-brier-val").textContent = m.brier_score.toFixed(5);
      document.getElementById("model-oe-val").textContent = m.ratio_O_E_global.toFixed(4);
    }

    if (data.los && data.los.metricas_test_2024) {
      const l = data.los.metricas_test_2024;
      document.getElementById("kpi-mae").textContent = `${l.mae_dias.toFixed(2)}d`;
      document.getElementById("los-mae-val").textContent = `${l.mae_dias.toFixed(2)} días`;
      document.getElementById("los-medae-val").textContent = `${l.medae_dias.toFixed(2)} días`;
      document.getElementById("los-rho-val").textContent = l.spearman_rho.toFixed(4);
      document.getElementById("los-oe-val").textContent = l.ratio_O_E_global.toFixed(4);
    }

    // Renderizar importancia de variables
    if (data.feature_importance_mortality) {
      const tbodyImp = document.getElementById("importance-tbody");
      if (tbodyImp) {
        tbodyImp.innerHTML = "";
        data.feature_importance_mortality.slice(0, 10).forEach(f => {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td><code>${f.feature}</code></td><td>${Math.round(f.importance_gain).toLocaleString()}</td>`;
          tbodyImp.appendChild(tr);
        });
      }
    }
  } catch (err) {
    console.error("Error cargando models:", err);
  }
}

// Calculadora interactiva al ingreso
function initCalculator() {
  const form = document.getElementById("patient-calc-form");
  if (!form) return;

  form.addEventListener("submit", async e => {
    e.preventDefault();

    const checkedElix = Array.from(document.querySelectorAll(".chip-checkbox:checked")).map(c => c.value);

    const payload = {
      edad: parseInt(document.getElementById("calc-edad").value, 10),
      sexo: document.getElementById("calc-sexo").value,
      prevision: document.getElementById("calc-prevision").value,
      tipo_ingreso: document.getElementById("calc-tipo-ingreso").value,
      procedencia_agr: document.getElementById("calc-procedencia").value,
      diagnostico1_cie10: document.getElementById("calc-cie10").value,
      n_egresos_12m: parseInt(document.getElementById("calc-egresos12m").value, 10),
      dias_desde_egreso_previo: document.getElementById("calc-dias-previo").value ? parseFloat(document.getElementById("calc-dias-previo").value) : null,
      comorbilidades_elix: checkedElix,
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      document.getElementById("result-mort-pct").textContent = `${data.porcentaje_mortalidad}%`;
      document.getElementById("result-los-days").textContent = `${data.estancia_esperada_dias} días`;
      document.getElementById("result-risk-level").textContent = `Riesgo ${data.nivel_riesgo}`;
      document.getElementById("result-vanwalraven").textContent = data.score_van_walraven;
      document.getElementById("result-grupo-clinico").textContent = data.grupo_clinico_asignado;

      const gauge = document.getElementById("risk-gauge");
      gauge.className = `gauge-circle risk-${data.nivel_riesgo.toLowerCase()}`;
    } catch (err) {
      alert("Error al predecir: " + err.message);
    }
  });
}
