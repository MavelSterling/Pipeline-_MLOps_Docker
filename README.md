# Pipeline de MLOps para Diagnóstico de Enfermedades

---

📌 **Maestría en Inteligencia Artificial Aplicada**

📒 **MLOps - Taller de Pipeline y Docker**

---

## 👥 Integrantes del Proyecto

* **Felipe Guerra**
* **Mavelyn Sterling**

---

## 🎯 Objetivo del Proyecto

Desarrollar un sistema de MLOps completo para el diagnóstico médico que sea capaz de predecir, dados los síntomas de un paciente, si es posible que sufra de alguna enfermedad. El sistema debe funcionar tanto para enfermedades comunes (con abundantes datos) como para enfermedades huérfanas (con datos limitados).

---

## 📋 Estructura del Proyecto

```
Pipeline-_MLOps_Docker/
├── README.md                           # Este archivo
├── requirements.txt                    # Dependencias de Python
├── .gitignore                         # Archivos a excluir de Git
├── .venv/                             # Entorno virtual de Python
├── docs/                              # Documentación del pipeline
│   ├── pipeline_design.md            # Diseño del pipeline de MLOps
│   ├── pipeline_diagram.md           # Diagrama del proceso
│   └── usage_instructions.md         # Instrucciones de uso
├── src/                              # Código fuente del servicio
│   ├── app.py                        # Aplicación Flask principal
│   ├── model.py                      # Función de diagnóstico médico
│   ├── requirements.txt              # Dependencias (copia)
│   └── templates/                    # Plantillas HTML
│       └── index.html               # Interfaz web
├── docker/                           # Archivos de Docker
│   └── Dockerfile                   # Configuración de Docker
├── data/                            # Datos de ejemplo
│   └── sample_symptoms.json         # Casos de prueba
├── Dockerfile                       # Dockerfile principal
├── docker-compose.yml               # Configuración Docker Compose
├── deploy.sh                        # Script de despliegue (Linux/Mac)
├── deploy.ps1                       # Script de despliegue (Windows)
├── setup_dev.py                     # Script de configuración de desarrollo
└── test_system.py                   # Script de pruebas
```

---

## 🚀 Inicio Rápido

### Prerrequisitos

- Docker instalado
- Python 3.8+ (para desarrollo local)

### Desarrollo Local

#### Opción 1: Configuración Automática (Recomendada)

```bash
# Ejecutar script de configuración automática
python setup_dev.py
```

#### Opción 2: Configuración Manual

1. **Crear entorno virtual:**

```bash
python -m venv .venv
```

2. **Activar entorno virtual:**

```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

4. **Ejecutar aplicación:**

```bash
python src/app.py
```

### Construcción y Ejecución con Docker

1. **Construir la imagen Docker:**

```bash
docker build -t medical-diagnosis-service .
```

2. **Ejecutar el contenedor:**

```bash
docker run -p 5000:5000 medical-diagnosis-service
```

3. **Acceder al servicio:**
   - Interfaz web: http://localhost:5000
   - API endpoint: http://localhost:5000/predict

---

## 📊 Pipeline de MLOps

El pipeline completo incluye las siguientes etapas:

1. **Diseño y Análisis**
2. **Ingesta y Preparación de Datos**
3. **Desarrollo y Entrenamiento de Modelos**
4. **Validación y Testing**
5. **Despliegue en Producción**
6. **Monitoreo y Mantenimiento**

Para más detalles, consulta [docs/pipeline_design.md](docs/pipeline_design.md)

---

## 🏥 Servicio de Diagnóstico

El servicio permite a los médicos ingresar síntomas del paciente y obtener un diagnóstico en tiempo real con los siguientes estados:

- **NO ENFERMO**: Paciente sin indicios de enfermedad
- **ENFERMEDAD LEVE**: Síntomas leves que requieren observación
- **ENFERMEDAD AGUDA**: Condición que requiere atención inmediata
- **ENFERMEDAD CRÓNICA**: Condición de larga duración que requiere tratamiento continuo

---

## 📖 Documentación

- [Diseño del Pipeline](docs/pipeline_design.md)
- [Instrucciones de Uso](docs/usage_instructions.md)

---

## 🔧 Tecnologías Utilizadas

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Containerización**: Docker
- **ML**: Scikit-learn, Pandas, NumPy

---

*Proyecto desarrollado por Felipe Guerra y Mavelyn Sterling para el taller de MLOps - Maestría en Inteligencia Artificial Aplicada*
