# Pipeline de MLOps para Diagnóstico de Enfermedades

---

📌 **Maestría en Inteligencia Artificial Aplicada**

📒 **MLOps - Taller #3: Reestructuración de Pipeline End-to-End**

🔄 **Versión 2.0 - Actualizado con Tecnologías MLOps Modernas**

---

## 👥 Integrantes del Proyecto

* **Felipe Guerra**
* **Mavelyn Sterling**

---

## 🎯 Objetivo del Proyecto

Desarrollar un **pipeline de MLOps end-to-end de nivel empresarial** para diagnóstico médico que predice, dados los síntomas de un paciente, si sufre de alguna enfermedad.

### Características principales:

- ✅ Soporte para **enfermedades comunes** (abundantes datos) y **enfermedades huérfanas** (datos limitados)
- ✅ **Despliegue flexible:** Local (offline), Híbrido (fallback), o Nube completa
- ✅ **Plataforma:** Azure + Databricks + Kubernetes
- ✅ **Cumplimiento normativo:** HIPAA, GDPR, FDA (SaMD)
- ✅ **Explicabilidad:** SHAP values en cada predicción
- ✅ **Escalabilidad:** 3-20 pods con autoscaling, >100 req/s
- ✅ **Monitoreo proactivo:** Detección automática de data drift
- ✅ **CI/CD completo:** Despliegues automatizados sin downtime

---

## 📋 Estructura del Proyecto Pipeline_MLOps_Docker/

```
Pipeline_MLOps_Docker/
├── README.md                              # Este archivo (Actualizado Taller #3)
├── CHANGELOG.md                           # Comparación Taller #1 vs #3
├── requirements.txt                       # Dependencias de Python
├── .gitignore                             # Archivos a excluir de Git
├── .venv/                                 # Entorno virtual de Python
├── docs/                                  # Documentación del pipeline
│   ├── pipeline_design.md                 # Diseño original (Taller #1)
│   ├── pipeline_design_v2.md              # Diseño completo MLOps (Taller #3)
│   └── arquitectura_tecnica.md            #  Arquitectura técnica detallada
├── src/                                   # Código fuente del servicio
│   ├── app.py                             # Aplicación Flask principal
│   ├── model.py                           # Función de diagnóstico médico
│   ├── requirements.txt                   # Dependencias
│   └── templates/                         # Plantillas HTML
│       └── index.html                     # Interfaz web
└── Dockerfile                             # Configuración Docker

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

## 📊 Pipeline de MLOps End-to-End

El pipeline reestructurado incluye **9 etapas** con **38+ tecnologías MLOps** específicas:

1. **Ingesta y Almacenamiento de Datos**

   - Azure Data Factory, Data Lake Gen2, Delta Lake, Event Hubs
2. **Procesamiento y Feature Engineering**

   - Databricks + Spark, Feature Store, Great Expectations, DVC
3. **Entrenamiento de Modelos**

   - MLflow, Optuna, XGBoost/LightGBM, Few-shot learning, AutoML
4. **Validación y Testing**

   - Pytest, Deepchecks, SHAP/LIME, validación clínica con médicos
5. **CI/CD Pipeline**

   - Azure DevOps, Docker, ACR, Terraform, Blue-Green/Canary deployment
6. **Despliegue en Producción**

   - Azure Kubernetes Service, FastAPI, API Management, multi-región
7. **Monitoreo y Observabilidad**

   - Prometheus, Grafana, Evidently AI (drift), Application Insights
8. **Reentrenamiento y Mejora Continua**

   - Databricks Jobs, Logic Apps, feedback loop médico, A/B testing
9. **Seguridad y Gobernanza**

   - Azure AD, Key Vault, Policy, cumplimiento HIPAA/GDPR/FDA

### 📈 Comparación con versión inicial:

| Aspecto        | Taller #1  | Taller #3 (actual)              |
| -------------- | ---------- | ------------------------------- |
| Tecnologías   | 3 básicas | 38+ MLOps enterprise            |
| Despliegue     | Solo local | Local + Híbrido + Nube         |
| CI/CD          | ❌ No      | ✅ Automatizado completo        |
| Monitoreo      | ❌ Vago    | ✅ Prometheus + Grafana + Drift |
| Explicabilidad | ❌ No      | ✅ SHAP en cada predicción     |
| Escalabilidad  | ❌ No      | ✅ AKS autoscaling 3-20 pods    |
| Seguridad      | ❌ Básica | ✅ HIPAA/GDPR compliant         |

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

Para información completa sobre los entregables del Taller #3, ver la sección **[📦 Entregables del Taller #3](#-entregables-del-taller-3)** más abajo.

### 📘 Resumen de documentos:

- **[Pipeline MLOps v2.0 (Taller #3)](docs/pipeline_design_v2.md)** ⭐ - Entregable principal
- **[CHANGELOG Taller #1 → #3](CHANGELOG.md)** - Comparación de 100+ cambios
- **[Arquitectura Técnica](docs/arquitectura_tecnica.md)** - Diagramas y flujos detallados
- **[Pipeline Original (Taller #1)](docs/pipeline_design.md)** - Referencia del taller #1

---

## 🔧 Stack Tecnológico Completo (38+ Tecnologías)

### **☁️ Plataforma Cloud**

- **Azure Data Factory** - Orquestación de ingesta
- **Azure Data Lake Gen2** - Data lake escalable
- **Azure Kubernetes Service (AKS)** - Orquestación de contenedores
- **Azure Machine Learning** - Plataforma ML alternativa
- **Azure Functions** - Inferencia batch serverless
- **Azure API Management** - Gateway centralizado
- **Azure Cosmos DB** - Base de datos NoSQL distribuida
- **Azure Key Vault** - Gestión de secretos
- **Azure Monitor / Application Insights** - Observabilidad

### **🔬 Data & ML**

- **Databricks** - Plataforma unificada (Spark + ML + Colaboración)
- **Delta Lake** - Formato datos con ACID transactions
- **Feature Store (Databricks)** - Gestión centralizada de features
- **Apache Spark** - Procesamiento distribuido
- **MLflow** - Experiment tracking & model registry
- **DVC** - Versionado de datasets
- **Great Expectations** - Validación de calidad de datos

### **🤖 Machine Learning**

- **Scikit-learn, XGBoost, LightGBM** - Algoritmos ML
- **PyTorch** - Deep learning & few-shot learning
- **Optuna** - Hyperparameter tuning
- **AutoML (Databricks)** - Exploración automatizada
- **SHAP / LIME** - Explicabilidad de modelos

### **🚀 DevOps & Infrastructure**

- **Azure DevOps / GitHub Actions** - CI/CD
- **Docker** - Containerización
- **Azure Container Registry (ACR)** - Registro privado
- **Terraform / ARM Templates** - Infrastructure as Code
- **Kubernetes** - Orquestación (AKS)
- **NGINX Ingress** - Load balancing

### **📊 Monitoring & Observability**

- **Prometheus** - Métricas en tiempo real
- **Grafana** - Dashboards visuales
- **Evidently AI** - Detección de data/model drift
- **Seldon Alibi** - Explicabilidad & outliers
- **Azure Log Analytics** - Logs centralizados
- **PagerDuty** - Sistema de alertas

### **🔐 Security & Governance**

- **Azure Active Directory** - Identidad & autenticación
- **OAuth2 / JWT** - Tokens de seguridad
- **Azure Policy** - Cumplimiento normativo
- **Azure Purview** - Gobernanza de datos
- **TLS 1.3 / mTLS** - Encriptación

### **🧪 Testing & Quality**

- **Pytest** - Testing automatizado
- **Deepchecks** - Validación específica ML
- **Coverage.py** - Cobertura de tests

### **🔄 Backend & APIs**

- **FastAPI** - Framework web de alto rendimiento
- **Uvicorn** - ASGI server
- **Pydantic** - Validación de datos

---

## 🎓 Contexto Académico

### **Taller #3: Reestructuración de Pipeline MLOps**

Este proyecto representa la **evolución completa** de la propuesta inicial (Taller #1) incorporando todos los conocimientos adquiridos sobre MLOps a lo largo del curso.

---

## 📦 Entregables del Taller #3

### **🎯 Documentos Principales:**

1. **[📘 Pipeline MLOps v2.0 - Completo (Taller #3)](docs/pipeline_design_v2.md)** ⭐ **ENTREGABLE PRINCIPAL**

   - Este es el **documento principal del Taller #3**
   - 2000+ líneas de especificaciones técnicas detalladas
   - 9 etapas del pipeline con 38+ tecnologías MLOps
   - Código de ejemplo funcional para cada componente
   - Justificación completa de decisiones tecnológicas
2. **[📋 CHANGELOG - Taller #1 vs Taller #3](CHANGELOG.md)**

   - Comparación exhaustiva de 100+ cambios entre versiones
   - Evolución de arquitectura conceptual a implementable
   - Tabla comparativa de tecnologías (3 → 38+)
   - Justificación de cada mejora realizada
3. **[🏗️ Arquitectura Técnica Detallada](docs/arquitectura_tecnica.md)**

   - Diagramas ASCII de arquitectura completa (9 capas)
   - Flujos de datos: entrenamiento, inferencia, reentrenamiento
   - Especificaciones técnicas de cada componente
   - Matriz de seguridad RBAC y checklist de implementación

### **📚 Documento de Referencia Histórica:**

- **[Pipeline Original (Taller #1)](docs/pipeline_design.md)**
  - Propuesta inicial conceptual del Taller #1
  - Mantenido como referencia histórica para comparación

---

**Cambios principales del Taller #1 al Taller #3:**

- ✅ De 3 tecnologías básicas → 38+ tecnologías MLOps enterprise
- ✅ De propuesta conceptual → Blueprint implementable completo
- ✅ De despliegue local único → Local + Híbrido + Nube
- ✅ De sin CI/CD → Pipeline automatizado completo
- ✅ De sin monitoreo → Observabilidad completa con alertas
- ✅ De black-box → Explicabilidad con SHAP en cada predicción
- ✅ De seguridad básica → Cumplimiento HIPAA/GDPR/FDA

---

## 💰 Estimación de Costos

### **Infraestructura Azure (producción):**

- Azure Data Lake Storage: ~$100/mes (1 TB)
- Databricks (Standard): ~$2,000/mes
- AKS (3-20 nodes autoscaling): ~$800-3,000/mes
- Azure Cosmos DB: ~$500/mes
- Otros servicios: ~$600/mes

**Total:** ~$4,000-6,500/mes (escalable según tráfico)

### **Alternativa económica:**

- Azure ML Endpoints (sin AKS): ~$1,500/mes
- Databricks Community: Gratis
- **Total:** ~$2,000-3,000/mes

### **Versión local (zonas rurales):**

- **Costo:** $0 (después de setup inicial)

---

## 📈 Métricas de Éxito

### **Técnicas:**

- Disponibilidad: 99.9% SLA
- Latencia p99: <500ms
- Throughput: >100 req/s

### **Machine Learning:**

- F1-score (enfermedades comunes): >0.85
- Recall (casos críticos): >0.95
- NPV: >0.98

### **Negocio:**

- Tiempo de diagnóstico: -30%
- Casos críticos detectados temprano: +25%
- Satisfacción médicos: >4.0/5.0

---

## 🚀 Roadmap de Implementación

### **Fase 1: Fundación (Mes 1-2)**

-  Setup de infraestructura Azure (Terraform)
-  Configuración de Databricks workspace
-  Implementación de ingesta de datos (ADF)
-  Setup de Feature Store

### **Fase 2: Modelado (Mes 2-3)**

-  Entrenamiento de modelos base (comunes + huérfanas)
-  Implementación de few-shot learning
-  Setup de MLflow tracking y registry
-  Validación con médicos especialistas

### **Fase 3: Despliegue (Mes 3-4)**

-  Setup de AKS y configuración de pods
-  Implementación de FastAPI
-  CI/CD pipeline con Azure DevOps
-  Configuración de API Management

### **Fase 4: Monitoreo (Mes 4-5)**

-  Setup de Prometheus + Grafana
-  Implementación de Evidently AI (drift)
-  Configuración de alertas (PagerDuty)
-  Dashboards para médicos y ML team

### **Fase 5: Piloto (Mes 5-6)**

-  Despliegue en 2-3 hospitales piloto
-  Recolección de feedback
-  Iteraciones basadas en uso real
-  Validación de métricas de negocio

### **Fase 6: Producción (Mes 6+)**

-  Escalado nacional
-  Submission FDA (si aplica)
-  Marketing y adopción
-  Mejora continua

---

*Proyecto desarrollado por Felipe Guerra y Mavelyn Sterling para el Taller #3 de MLOps - Maestría en Inteligencia Artificial Aplicada*

*Versión 2.0 - Pipeline MLOps End-to-End de Nivel Empresarial*
