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

## 📋 Estructura del ProyectoPipeline-_MLOps_Docker/

├── README.md                           # Este archivo
├── requirements.txt                    # Dependencias de Python
├── .gitignore                         # Archivos a excluir de Git
├── .venv/                             # Entorno virtual de Python
├── docs/                              # Documentación del pipeline
│   ├── pipeline_design.md            # Diseño del pipeline de MLOps
├── src/                              # Código fuente del servicio
│   ├── app.py                        # Aplicación Flask principal
│   ├── model.py                      # Función de diagnóstico médico
│   ├── requirements.txt              # Dependencias
│   └── templates/                    # Plantillas HTML
│       └── index.html               # Interfaz web
├── data/                            # Datos de ejemplo
│   └── sample_symptoms.json         # Casos de prueba
├── Dockerfile                       # Dockerfile principal
├── docker-compose.yml               # Configuración Docker Compose

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

---

## 🏥 Servicio de Diagnóstico

El servicio permite a los médicos ingresar síntomas del paciente y obtener un diagnóstico en tiempo real con los siguientes estados:

- **NO ENFERMO**: Paciente sin indicios de enfermedad
- **ENFERMEDAD LEVE**: Síntomas leves que requieren observación
- **ENFERMEDAD AGUDA**: Condición que requiere atención inmediata
- **ENFERMEDAD CRÓNICA**: Condición de larga duración que requiere tratamiento continuo

---

## 🧪 Casos de Uso

A continuación, algunos ejemplos tomados de `data/sample_symptoms.json` para ilustrar cuándo el sistema determina que un paciente está enfermo o no.

- Nota: para evaluar correctamente, se deben ingresar mínimo 3 síntomas por paciente.
- **NO ENFERMO (CASE_005)**: `fatiga=2`, `dolor_muscular=1`, `mareos=1` → Diagnóstico esperado: **NO ENFERMO**
- **ENFERMEDAD LEVE (CASE_001)**: `fiebre=6`, `congestion_nasal=8`, `dolor_garganta=7` → Diagnóstico esperado: **ENFERMEDAD LEVE**
- **ENFERMEDAD AGUDA (CASE_003)**: `dolor_pecho=10`, `dificultad_respirar=9`, `fatiga=8` → Diagnóstico esperado: **ENFERMEDAD AGUDA**
- **ENFERMEDAD CRÓNICA (CASE_004)**: `perdida_peso=8`, `cambios_vision=7`, `fatiga=9` → Diagnóstico esperado: **ENFERMEDAD CRÓNICA**

---

## 📖 Documentación

- [Diseño del Pipeline](docs/pipeline_design.md)

---

## 🔧 Tecnologías Utilizadas

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Containerización**: Docker
- **ML**: Scikit-learn, Pandas, NumPy

---

*Proyecto desarrollado por Felipe Guerra y Mavelyn Sterling para el taller de MLOps - Maestría en Inteligencia Artificial Aplicada*
