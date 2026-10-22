# Imagen para reproducir las Fases 2 a 5 con el entorno exacto de la reproducción independiente.
# Los datos de FONASA y las capas viven fuera de la imagen y se montan como volúmenes:
#
#   docker build -t grd-tesis .
#   docker run --rm -it -v "<data seminario_full>:/datos:ro" -v "<carpeta de trabajo>:/trabajo" grd-tesis
#   # dentro del contenedor, por ejemplo:
#   cd fase3_preparacion && python run_fase3.py --data-root /datos --capas /trabajo/capas
#
# La Fase 6 (API, base de datos y web) tiene su propio docker-compose.yml en fase6_despliegue/.
FROM python:3.13.16-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /tesis
COPY requirements-lock.txt .
RUN pip install -r requirements-lock.txt
COPY . .
CMD ["bash"]
