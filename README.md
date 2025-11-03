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
## 📋 Estructura del Proyecto Pipeline_MLOps_Docker/
```


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
├── Dockerfile                       # Dockerfile 

```
---

## 🚀 Inicio Rápido

### Prerrequisitos

- Docker instalado
- Python 3.8+ (para ejecución sin docker)

### Ejecución sin Docker

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
---

## 📊 Pipeline de MLOps

El pipeline completo incluye las siguientes etapas:

1. **Diseño y Análisis**
2. **Desarrollo del modelo y manejo de datos**
3. **Producción, monitoreo y mejora continua**

---

## 🏥 Servicio de Diagnóstico

El servicio permite a los médicos ingresar síntomas del paciente y obtener un diagnóstico en tiempo real con los siguientes estados:

- **NO ENFERMO**: Paciente sin indicios de enfermedad
- **MOLESTIAS LEVES**: Paciente con síntomas o molestias muy leves
- **ENFERMEDAD LEVE**: Síntomas leves que requieren observación
- **ENFERMEDAD AGUDA**: Condición que requiere atención inmediata
- **ENFERMEDAD CRÓNICA**: Condición de larga duración que requiere tratamiento continuo

---

## 🧪 Casos de Uso

A continuación, se muestran algunos ejemplos de casos de uso:

- Nota: para evaluar correctamente, se deben ingresar mínimo 3 síntomas por paciente.
- `fatiga=2`, `dolor_muscular=1`, `mareos=1` → Diagnóstico esperado: **NO ENFERMO**
- `fiebre=3`, `dolor_cabeza=3`, `dificultad_respirar=5` → Diagnóstico esperado: **MOLESTIAS LEVES**
- `fiebre=10`, `dolor_pecho=8`, `dificultad_respirar=7` → Diagnóstico esperado: **ENFERMEDAD LEVE**
- `dolor_pecho=7`, `dificultad_respirar=9`, `fatiga=8` → Diagnóstico esperado: **ENFERMEDAD AGUDA**
- `dolor_pecho=10`, `Tos=10`, `dificultad_respirar=10` → Diagnóstico esperado: **ENFERMEDAD CRÓNICA**

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
