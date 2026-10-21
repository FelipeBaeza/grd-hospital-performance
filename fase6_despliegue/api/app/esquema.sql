-- Esquema de la base: solo catálogo de hospitales y resultados agregados. Ninguna tabla guarda registros de pacientes ni perfiles de la demo.
DROP TABLE IF EXISTS calibracion_decil, subgrupo, contexto_hospital, indicador_hospital, criterio_global, hospital, metadatos CASCADE;

CREATE TABLE hospital (
    cod_hospital TEXT PRIMARY KEY
);

CREATE TABLE criterio_global (
    tarea TEXT NOT NULL,
    criterio TEXT NOT NULL,
    valor DOUBLE PRECISION,
    ic_inf DOUBLE PRECISION,
    ic_sup DOUBLE PRECISION,
    umbral TEXT,
    veredicto TEXT NOT NULL,
    es_criterio BOOLEAN NOT NULL,
    orden INTEGER NOT NULL,
    PRIMARY KEY (tarea, criterio)
);

CREATE TABLE indicador_hospital (
    tarea TEXT NOT NULL,
    cod_hospital TEXT NOT NULL REFERENCES hospital (cod_hospital),
    n_egresos INTEGER NOT NULL,
    observado_pub TEXT,              -- texto: puede ser "<10" cuando el conteo es pequeño
    esperado DOUBLE PRECISION,
    oe DOUBLE PRECISION,             -- nulo cuando el conteo es pequeño (revelaría el observado)
    ic_inf DOUBLE PRECISION,
    ic_sup DOUBLE PRECISION,
    lim_inf_95 DOUBLE PRECISION,
    lim_sup_95 DOUBLE PRECISION,
    lim_inf_998 DOUBLE PRECISION,
    lim_sup_998 DOUBLE PRECISION,
    zona TEXT NOT NULL,
    PRIMARY KEY (tarea, cod_hospital)
);

CREATE TABLE contexto_hospital (
    cod_hospital TEXT PRIMARY KEY REFERENCES hospital (cod_hospital),
    indice_vs_2023 DOUBLE PRECISION,
    indice_relativo_al_anio DOUBLE PRECISION,
    tasa_traslado_saliente DOUBLE PRECISION,
    proporcion_traslado_entrante DOUBLE PRECISION
);

CREATE TABLE subgrupo (
    tarea TEXT NOT NULL,
    variable TEXT NOT NULL,
    grupo TEXT NOT NULL,
    n INTEGER NOT NULL,
    observado_pub TEXT,
    esperado DOUBLE PRECISION,
    oe DOUBLE PRECISION,
    PRIMARY KEY (tarea, variable, grupo)
);

CREATE TABLE calibracion_decil (
    tarea TEXT NOT NULL,
    decil INTEGER NOT NULL,
    n INTEGER NOT NULL,
    observado DOUBLE PRECISION,
    predicho DOUBLE PRECISION,
    razon_oe DOUBLE PRECISION,
    PRIMARY KEY (tarea, decil)
);

CREATE TABLE metadatos (
    clave TEXT PRIMARY KEY,
    valor JSONB NOT NULL
);
