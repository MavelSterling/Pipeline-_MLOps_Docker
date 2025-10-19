# Instrucciones de Uso del Sistema de Diagnóstico Médico

## 📋 Descripción General

El Sistema de Diagnóstico Médico es una aplicación web desarrollada con Flask y Docker que permite a los médicos realizar diagnósticos asistidos por inteligencia artificial basados en los síntomas del paciente. El sistema simula un modelo de machine learning que puede manejar tanto enfermedades comunes como enfermedades huérfanas.

---

## 🚀 Instalación y Configuración

### Prerrequisitos

- **Docker**: Versión 20.10 o superior
- **Docker Compose**: Versión 2.0 o superior (opcional)
- **Git**: Para clonar el repositorio

### Verificar Instalación de Docker

```bash
# Verificar versión de Docker
docker --version

# Verificar que Docker esté ejecutándose
docker info
```

---

## 🏗️ Construcción de la Imagen Docker

### 1. Clonar el Repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd Pipeline-_MLOps_Docker
```

### 2. Construir la Imagen

```bash
# Construir la imagen con el nombre 'medical-diagnosis-service'
docker build -t medical-diagnosis-service .

# Verificar que la imagen se construyó correctamente
docker images | grep medical-diagnosis-service
```

### 3. Verificar la Construcción

```bash
# Ver detalles de la imagen
docker inspect medical-diagnosis-service

# Ver historial de construcción
docker history medical-diagnosis-service
```

---

## 🏃‍♂️ Ejecución del Servicio

### 1. Ejecutar el Contenedor

```bash
# Ejecutar el contenedor en modo interactivo
docker run -p 5000:5000 medical-diagnosis-service

# Ejecutar en segundo plano (modo detached)
docker run -d -p 5000:5000 --name medical-diagnosis medical-diagnosis-service

# Ejecutar con variables de entorno personalizadas
docker run -p 5000:5000 -e FLASK_ENV=development medical-diagnosis-service
```

### 2. Verificar el Estado del Servicio

```bash
# Ver contenedores en ejecución
docker ps

# Ver logs del contenedor
docker logs medical-diagnosis

# Ver logs en tiempo real
docker logs -f medical-diagnosis
```

### 3. Health Check

```bash
# Verificar salud del servicio
curl http://localhost:5000/health

# Respuesta esperada:
# {
#   "status": "healthy",
#   "service": "medical-diagnosis-service",
#   "version": "1.0.0"
# }
```

---

## 🌐 Uso del Servicio

### 1. Interfaz Web

**URL**: http://localhost:5000

La interfaz web proporciona una interfaz intuitiva para que los médicos ingresen los síntomas del paciente:

1. **Acceder a la interfaz**: Abrir http://localhost:5000 en el navegador
2. **Ingresar síntomas**: Usar la escala de 0-10 para cada síntoma
3. **Realizar diagnóstico**: Hacer clic en "Realizar Diagnóstico"
4. **Revisar resultados**: Analizar el diagnóstico y recomendaciones

### 2. API REST

#### Endpoint Principal: POST /predict

**URL**: http://localhost:5000/predict

**Método**: POST

**Content-Type**: application/json

**Ejemplo de Request**:

```json
{
  "fiebre": 8,
  "dolor_cabeza": 6,
  "nausea": 4,
  "fatiga": 5,
  "dolor_pecho": 2
}
```

**Ejemplo de Response**:

```json
{
  "diagnosis": "ENFERMEDAD_LEVE",
  "confidence": 0.756,
  "most_likely_condition": "infeccion_respiratoria",
  "condition_confidence": 0.823,
  "symptom_score": 0.756,
  "pattern_scores": {
    "infeccion_respiratoria": 0.823,
    "migrana": 0.456,
    "gastroenteritis": 0.234
  },
  "recommendations": [
    "Monitorear síntomas de cerca",
    "Considerar consulta médica si los síntomas persisten",
    "Mantener reposo y hidratación adecuada",
    "Evitar actividades extenuantes"
  ],
  "input_symptoms": {
    "fiebre": 8,
    "dolor_cabeza": 6,
    "nausea": 4,
    "fatiga": 5,
    "dolor_pecho": 2
  }
}
```

#### Otros Endpoints

**GET /health**: Health check del servicio

```bash
curl http://localhost:5000/health
```

**GET /symptoms**: Lista de síntomas disponibles

```bash
curl http://localhost:5000/symptoms
```

**GET /api/docs**: Documentación de la API

```bash
curl http://localhost:5000/api/docs
```

---

## 🧪 Casos de Prueba

### Casos de Ejemplo Incluidos

El archivo `data/sample_symptoms.json` contiene casos de prueba predefinidos:

1. **Caso 001**: Resfriado común (ENFERMEDAD_LEVE)
2. **Caso 002**: Migraña (ENFERMEDAD_LEVE)
3. **Caso 003**: Emergencia cardíaca (ENFERMEDAD_AGUDA)
4. **Caso 004**: Diabetes no controlada (ENFERMEDAD_CRONICA)
5. **Caso 005**: Paciente sano (NO_ENFERMO)

### Probar Casos de Ejemplo

```bash
# Caso 1: Resfriado común
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fiebre": 6,
    "dolor_cabeza": 4,
    "congestion_nasal": 8,
    "dolor_garganta": 7,
    "tos": 5,
    "fatiga": 3
  }'

# Caso 2: Emergencia cardíaca
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "dolor_pecho": 10,
    "dificultad_respirar": 9,
    "mareos": 7,
    "nausea": 6,
    "fatiga": 8
  }'
```

---

## 🔧 Configuración Avanzada

### Variables de Entorno

```bash
# Configuración de desarrollo
docker run -p 5000:5000 \
  -e FLASK_ENV=development \
  -e FLASK_DEBUG=1 \
  medical-diagnosis-service

# Configuración de producción
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -e WORKERS=4 \
  medical-diagnosis-service
```

### Configuración de Red

```bash
# Crear red personalizada
docker network create medical-network

# Ejecutar con red personalizada
docker run -p 5000:5000 \
  --network medical-network \
  --name medical-diagnosis \
  medical-diagnosis-service
```

### Volúmenes para Persistencia

```bash
# Montar volumen para logs
docker run -p 5000:5000 \
  -v /host/logs:/app/logs \
  medical-diagnosis-service

# Montar volumen para datos
docker run -p 5000:5000 \
  -v /host/data:/app/data \
  medical-diagnosis-service
```

---

## 🐳 Docker Compose (Opcional)

### Archivo docker-compose.yml

```yaml
version: '3.8'

services:
  medical-diagnosis:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - WORKERS=4
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Comandos Docker Compose

```bash
# Construir y ejecutar
docker-compose up --build

# Ejecutar en segundo plano
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

---

## 🔍 Monitoreo y Debugging

### Ver Logs del Servicio

```bash
# Logs en tiempo real
docker logs -f medical-diagnosis

# Últimas 100 líneas
docker logs --tail 100 medical-diagnosis

# Logs con timestamp
docker logs -t medical-diagnosis
```

### Inspeccionar Contenedor

```bash
# Información detallada del contenedor
docker inspect medical-diagnosis

# Estadísticas de uso de recursos
docker stats medical-diagnosis

# Procesos dentro del contenedor
docker exec medical-diagnosis ps aux
```

### Debugging

```bash
# Acceder al shell del contenedor
docker exec -it medical-diagnosis /bin/bash

# Ejecutar comando específico
docker exec medical-diagnosis python -c "import model; print('Model loaded successfully')"

# Ver variables de entorno
docker exec medical-diagnosis env
```

---

## 🛠️ Mantenimiento

### Actualización del Servicio

```bash
# Detener contenedor actual
docker stop medical-diagnosis

# Eliminar contenedor
docker rm medical-diagnosis

# Reconstruir imagen
docker build -t medical-diagnosis-service .

# Ejecutar nueva versión
docker run -d -p 5000:5000 --name medical-diagnosis medical-diagnosis-service
```

### Limpieza de Recursos

```bash
# Eliminar contenedores detenidos
docker container prune

# Eliminar imágenes no utilizadas
docker image prune

# Limpieza completa (¡CUIDADO!)
docker system prune -a
```

---

## 🚨 Solución de Problemas

### Problemas Comunes

1. **Puerto 5000 ocupado**:

   ```bash
   # Usar puerto diferente
   docker run -p 5001:5000 medical-diagnosis-service
   ```
2. **Error de permisos**:

   ```bash
   # Verificar permisos de Docker
   sudo usermod -aG docker $USER
   ```
3. **Contenedor no inicia**:

   ```bash
   # Ver logs de error
   docker logs medical-diagnosis

   # Verificar configuración
   docker inspect medical-diagnosis
   ```
4. **API no responde**:

   ```bash
   # Verificar health check
   curl http://localhost:5000/health

   # Verificar conectividad
   docker exec medical-diagnosis curl localhost:5000/health
   ```

### Logs de Error Comunes

- **"Port already in use"**: Cambiar puerto o detener servicio que lo usa
- **"Container not found"**: Verificar nombre del contenedor
- **"Image not found"**: Reconstruir la imagen
- **"Permission denied"**: Verificar permisos de Docker

---

## 📞 Soporte

Para soporte técnico o reportar problemas:

1. **Revisar logs**: `docker logs medical-diagnosis`
2. **Verificar documentación**: Revisar este archivo
3. **Consultar issues**: Buscar en el repositorio de GitHub
4. **Contactar desarrolladores**: Felipe Guerra, Mavelyn Sterling

---

## 📚 Referencias Adicionales

- [Documentación de Docker](https://docs.docker.com/)
- [Documentación de Flask](https://flask.palletsprojects.com/)
- [Diseño del Pipeline MLOps](pipeline_design.md)
- [Diagrama del Pipeline](pipeline_diagram.md)
