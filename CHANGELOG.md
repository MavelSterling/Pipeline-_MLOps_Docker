# CHANGELOG - Pipeline de MLOps para Diagnóstico Médico

## 📋 Registro de Cambios: Taller #1 → Taller #3

Este documento detalla todos los cambios, mejoras y reestructuraciones realizadas entre la propuesta inicial (Taller #1) y la propuesta final (Taller #3).

---

## 🎯 Resumen Ejecutivo de Cambios

| Aspecto                    | Taller #1 (Versión Inicial) | Taller #3 (Versión Mejorada)         |
| -------------------------- | ---------------------------- | ------------------------------------- |
| **Nivel de detalle** | Conceptual, general          | Técnico, específico, implementable  |
| **Tecnologías**     | No especificadas             | 30+ tecnologías MLOps detalladas     |
| **Plataforma**       | Local con Docker             | Azure + Databricks + local            |
| **Orquestación**    | No definida                  | Databricks Jobs + Azure DevOps        |
| **Monitoreo**        | Mencionado vagamente         | Prometheus + Grafana + Evidently AI   |
| **CI/CD**            | No presente                  | Pipeline completo Azure DevOps        |
| **Seguridad**        | Mencionada superficialmente  | Implementación HIPAA + GDPR completa |
| **Explicabilidad**   | No definida                  | SHAP + LIME integrados                |
| **Escalabilidad**    | No especificada              | AKS con autoscaling 3-20 pods         |
| **Despliegue**       | Solo local                   | Local + Híbrido + Nube               |

---

## 📊 CAMBIOS POR ETAPA DEL PIPELINE

### **ETAPA 1: Ingesta y Almacenamiento de Datos**

#### ❌ **Versión Taller #1:**

```
- Descripción genérica: "Ingesta de Datos"
- No especifica fuentes de datos
- No menciona formatos
- No define almacenamiento
- No considera streaming
```

#### ✅ **Versión Taller #3:**

```
+ Azure Data Factory para orquestación de ingesta
+ Azure Data Lake Storage Gen2 (almacenamiento escalable)
+ Delta Lake con ACID transactions y time travel
+ Azure Event Hubs para streaming en tiempo real
+ Soporte para HL7, FHIR, JSON
+ Arquitectura Bronze-Silver-Gold (medallion)
+ Particionado por fecha para optimización
+ Encriptación AES-256 en reposo
+ Integración segura con EHR via OAuth2
```

**Razón del cambio:**

- Necesidad de especificar tecnologías concretas y escalables
- Soporte para múltiples fuentes de datos heterogéneas
- Cumplimiento con regulaciones de salud

---

### **ETAPA 2: Procesamiento y Feature Engineering**

#### ❌ **Versión Taller #1:**

```
- "Validación y Limpieza" (sin detalles)
- "Análisis Exploratorio" (sin herramientas)
- No menciona Feature Store
- No especifica procesamiento distribuido
- No define validación de calidad de datos
```

#### ✅ **Versión Taller #3:**

```
+ Databricks Notebooks + Apache Spark (procesamiento distribuido)
+ Feature Store (Databricks) para consistencia train/inference
+ Great Expectations para validación automática de calidad
+ DVC (Data Version Control) para versionado de datasets
+ PySpark + Pandas UDFs para transformaciones personalizadas
+ 150+ features para enfermedades comunes
+ 50+ features especializadas para enfermedades huérfanas
+ Embeddings con BioBERT para texto médico
+ Imputación KNN/MICE para valores faltantes
+ SMOTE para balanceo de clases
```

**Razón del cambio:**

- Necesidad de procesar millones de registros (Spark)
- Evitar feature drift entre entrenamiento e inferencia (Feature Store)
- Garantizar calidad de datos con validaciones automáticas
- Reproducibilidad con versionado de datos

---

### **ETAPA 3: Entrenamiento de Modelos**

#### ❌ **Versión Taller #1:**

```
- Mención genérica de "Modelos Especializados"
- No especifica algoritmos
- No define estrategia para enfermedades huérfanas
- No menciona experiment tracking
- No define hyperparameter tuning
- No especifica modelo ensemble
```

#### ✅ **Versión Taller #3:**

```
+ MLflow para experiment tracking y model registry
+ Azure Machine Learning como alternativa/complementaria
+ Optuna/Hyperopt para hyperparameter tuning (500 trials)
+ AutoML (Databricks) para exploración rápida
+ XGBoost, LightGBM, Random Forest para enfermedades comunes
+ Few-shot learning + Transfer learning para enfermedades huérfanas
+ Siamese Networks para comparar casos similares
+ Meta-learning (MAML) para adaptación rápida
+ Ensemble con meta-learner y lógica de precaución clínica
+ Model Registry con estados: Staging → Production → Archived
+ Entrenamiento distribuido en GPU clusters (Standard_NC6s_v3)
+ Logging completo de parámetros, métricas y artefactos
```

**Razón del cambio:**

- Necesidad de tracking formal de experimentos (reproducibilidad)
- Estrategia específica para pocos datos (enfermedades huérfanas)
- Gestión profesional del ciclo de vida de modelos
- Optimización automática de hiperparámetros

---

### **ETAPA 4: Validación y Testing**

#### ❌ **Versión Taller #1:**

```
- "Validación Cruzada" (sin detalles)
- "Evaluación de Modelos" (sin métricas específicas)
- Mención vaga de "validación humana"
- No define tests automatizados
- No menciona explicabilidad
```

#### ✅ **Versión Taller #3:**

```
+ Pytest para unit tests e integration tests
+ Great Expectations para data validation
+ Deepchecks para validación específica de ML
+ SHAP/LIME para explicabilidad
+ Métricas clínicas específicas:
  - Sensibilidad casos urgentes: >95%
  - Especificidad casos leves: >80%
  - NPV (Negative Predictive Value): >98%
+ Análisis de fairness (disparate impact, equal opportunity)
+ Validación por subgrupos (edad, género, tipo enfermedad)
+ Panel de 5+ médicos especialistas (criterio >90% acuerdo)
+ Tests de latencia (<500ms p99)
+ Tests de throughput (>100 req/s)
+ Dashboard de feature importance
+ Ejemplos contrafactuales para interpretación
```

**Razón del cambio:**

- Necesidad de testing automatizado completo
- Validación no solo técnica sino clínica
- Explicabilidad crítica en contexto médico
- Detección de bias y garantía de fairness

---

### **ETAPA 5: CI/CD Pipeline**

#### ❌ **Versión Taller #1:**

```
- NO EXISTÍA en la propuesta original
- Solo había Dockerfile manual
- No se mencionaba automatización
- No había estrategia de despliegue
```

#### ✅ **Versión Taller #3:**

```
+ Azure DevOps Pipelines / GitHub Actions completo
+ Pipeline de 5 etapas:
  1. Build & Test
  2. BuildDocker & Push to ACR
  3. Deploy to Staging
  4. A/B Testing (24h)
  5. Deploy to Production
+ Docker multi-stage builds optimizados
+ Azure Container Registry (ACR) privado
+ Terraform/ARM Templates para Infrastructure as Code
+ pytest + coverage automático (target >85%)
+ Blue-Green Deployment
+ Canary Release (5% → 25% → 50% → 100%)
+ Rollback automático si detecta issues
+ Health checks y readiness probes
```

**Razón del cambio:**

- Automatización completa del despliegue (reduce errores humanos)
- Despliegues seguros con testing previo en staging
- Estrategias de despliegue sin downtime
- Rollback rápido en caso de problemas

---

### **ETAPA 6: Despliegue en Producción**

#### ❌ **Versión Taller #1:**

```
- Solo Flask + Docker local
- Sin estrategia de escalabilidad
- Sin load balancing
- Sin alta disponibilidad
- API no especificada
```

#### ✅ **Versión Taller #3:**

```
+ Azure Kubernetes Service (AKS) con orquestación completa
+ Azure ML Endpoints como alternativa simplificada
+ FastAPI (async, alto rendimiento, validación automática)
+ NGINX Ingress Controller (load balancing, TLS)
+ Azure API Management (gateway centralizado)
+ Horizontal Pod Autoscaler (3-20 pods)
+ Configuración de recursos (CPU, memoria) por pod
+ Liveness y Readiness probes
+ Multi-región (West US 2 + East US 2)
+ Azure Functions para inferencia batch asíncrona
+ Tres modos de despliegue:
  1. Local (offline, para zonas rurales)
  2. Híbrido (cloud + local fallback)
  3. Nube completa (hospitales grandes)
+ API REST con validación Pydantic
+ Endpoints: /predict, /health, /ready, /metrics
+ Geo-replication para disaster recovery
```

**Razón del cambio:**

- Necesidad de escalabilidad horizontal (muchos usuarios)
- Alta disponibilidad (99.9% SLA)
- Múltiples opciones de despliegue según contexto
- API profesional con validación y documentación automática

---

### **ETAPA 7: Monitoreo y Observabilidad**

#### ❌ **Versión Taller #1:**

```
- "Monitoreo en Tiempo Real" (sin herramientas)
- No especifica métricas
- No define alertas
- No menciona data drift
- No especifica dashboards
```

#### ✅ **Versión Taller #3:**

```
+ Azure Monitor (infraestructura)
+ Application Insights (telemetría de aplicaciones)
+ Prometheus (métricas custom en tiempo real)
+ Grafana (3 dashboards: Operacional, ML Performance, Clínico)
+ Evidently AI (detección automática de data drift)
+ Seldon Alibi (explicabilidad y outliers)
+ Azure Log Analytics (logs centralizados)
+ PagerDuty/Azure Alerts (sistema de alertas)
+ Métricas monitoreadas:
  - Infraestructura: CPU, memoria, latencia, throughput
  - Aplicación: requests/s, errores 4xx/5xx, latencia p50/p95/p99
  - ML: data drift (KS test, PSI), prediction drift, concept drift
  - Clínicas: tasa detección casos urgentes, falsos positivos, tiempo respuesta
+ Alertas con severidad (critical, warning, info)
+ Canales: PagerDuty (critical), Teams/Slack (warning), Email (info)
+ SHAP values en cada predicción para auditoría
+ Almacenamiento de explicaciones en Cosmos DB
```

**Razón del cambio:**

- Monitoreo proactivo (detectar problemas antes de que afecten usuarios)
- Observabilidad completa (logs, métricas, traces)
- Detección temprana de degradación del modelo
- Cumplimiento con requerimientos de auditoría

---

### **ETAPA 8: Reentrenamiento y Mejora Continua**

#### ❌ **Versión Taller #1:**

```
- "Re-entrenamiento" (sin detalles)
- Flecha de retroalimentación en diagrama
- No especifica triggers
- No define estrategia
- No menciona feedback loop
```

#### ✅ **Versión Taller #3:**

```
+ Databricks Jobs para orquestación de reentrenamiento
+ Azure Logic Apps para workflows automatizados
+ Azure Cosmos DB para almacenar feedback médico
+ Triggers automáticos:
  1. Programado (cada 2 semanas)
  2. Data drift >0.7 por >24h
  3. Degradación métricas (F1 cae >5%)
  4. Acumulación de 500+ feedback de médicos
+ Pipeline completo de reentrenamiento:
  1. Extraer nuevos datos
  2. Validar calidad (Great Expectations)
  3. Combinar con históricos
  4. Feature engineering
  5. Entrenar modelo nuevo
  6. Evaluar y comparar con actual
  7. Promover a Staging si supera
  8. A/B testing en producción (24h-7 días)
  9. Promover a Production si exitoso
+ Estrategia Champion/Challenger (shadow mode)
+ Canary release gradual (10% → 25% → 50% → 100%)
+ Rollback automático si issues
+ Sistema de feedback médico:
  - Recolección de diagnósticos reales post-predicción
  - Alertas en caso de falsos negativos críticos
  - Análisis de discrepancias
+ Versionado completo en MLflow
+ Auditoría de cambios de modelo
```

**Razón del cambio:**

- Necesidad de adaptación continua (concept drift es inevitable)
- Feedback loop formal con médicos
- Reentrenamiento seguro (sin romper producción)
- Decisiones basadas en datos, no intuitivas

---

### **ETAPA 9: Seguridad y Gobernanza**

#### ❌ **Versión Taller #1:**

```
- Mención superficial de "privacidad"
- "Controles de acceso, trazabilidad"
- No especifica tecnologías
- No define cumplimiento normativo
- No menciona encriptación
- No define auditoría
```

#### ✅ **Versión Taller #3:**

```
+ Azure Active Directory (SSO, MFA, RBAC)
+ Azure Key Vault (gestión de secretos con rotación automática)
+ Azure Policy (enforcement de políticas de compliance)
+ Azure Security Center (detección de amenazas)
+ Azure Purview (gobernanza de datos, linaje)
+ Audit logs completos (retención 7 años)
+ Encriptación:
  - En tránsito: TLS 1.3, mTLS entre microservicios
  - En reposo: AES-256 en Storage, Cosmos DB, ACR
+ Roles definidos:
  - MedicoGeneral (predicciones, explicaciones)
  - MedicoAdmin (+ métricas, dashboard)
  - DataScientist (+ MLflow, experimentos)
  - Auditor (solo lectura logs)
+ Cumplimiento normativo:
  - HIPAA compliant (BAA con Azure)
  - GDPR (derecho al olvido, portabilidad, minimización)
  - FDA (si aplica como SaMD): validación clínica, trazabilidad
+ Auditoría de predicciones:
  - Quién: user_id del médico
  - Cuándo: timestamp UTC
  - Qué: hash de inputs (no datos crudos), predicción, modelo version
  - Dónde: hospital, IP
+ Anonimización:
  - SHA-256 hash de patient_id
  - Eliminación de PII antes de almacenar
  - Differential privacy en métricas agregadas
+ Disaster Recovery:
  - RPO: 1 hora (backups incrementales)
  - RTO: 4 horas (failover automático)
  - Geo-replication (West US, East US, Europe West)
+ Business Continuity:
  - Azure Site Recovery
  - Failover automático en <5 minutos
  - Versionado de modelos (últimas 10 versiones)
```

**Razón del cambio:**

- Cumplimiento obligatorio con HIPAA/GDPR/FDA
- Protección de datos sensibles de salud
- Trazabilidad completa para auditorías
- Gestión profesional de identidades y accesos

---

## 🆕 COMPONENTES COMPLETAMENTE NUEVOS

### **1. Feature Store**

- **NO EXISTÍA** en Taller #1
- **Ahora:** Databricks Feature Store centralizado
- **Beneficio:** Consistencia entre training e inference, reutilización de features

### **2. Experiment Tracking**

- **NO EXISTÍA** en Taller #1
- **Ahora:** MLflow con tracking completo de experimentos
- **Beneficio:** Reproducibilidad, comparación de modelos, auditoría

### **3. Model Registry**

- **NO EXISTÍA** en Taller #1
- **Ahora:** MLflow Model Registry con Staging/Production
- **Beneficio:** Gestión formal del ciclo de vida de modelos

### **4. CI/CD Pipeline**

- **NO EXISTÍA** en Taller #1
- **Ahora:** Azure DevOps con 5 etapas automatizadas
- **Beneficio:** Despliegues seguros, rápidos y sin errores humanos

### **5. Data Drift Detection**

- **NO EXISTÍA** en Taller #1
- **Ahora:** Evidently AI con alertas automáticas
- **Beneficio:** Detección temprana de degradación del modelo

### **6. Explicabilidad**

- **NO EXISTÍA** en Taller #1
- **Ahora:** SHAP/LIME en cada predicción
- **Beneficio:** Interpretabilidad para médicos, cumplimiento regulatorio

### **7. API Management**

- **NO EXISTÍA** en Taller #1
- **Ahora:** Azure API Management con políticas
- **Beneficio:** Seguridad centralizada, rate limiting, analytics

### **8. Orquestación de Contenedores**

- **Taller #1:** Docker manual
- **Taller #3:** AKS con autoscaling, load balancing, multi-región
- **Beneficio:** Escalabilidad automática, alta disponibilidad

### **9. Infrastructure as Code**

- **NO EXISTÍA** en Taller #1
- **Ahora:** Terraform/ARM Templates
- **Beneficio:** Reproducibilidad de infraestructura, versionado

### **10. Estrategia Few-Shot Learning**

- **Taller #1:** Mencionado vagamente
- **Taller #3:** Implementación concreta con Siamese Networks, MAML
- **Beneficio:** Predicción efectiva con pocos datos (enfermedades huérfanas)

---

## 🔄 CAMBIOS EN EL DIAGRAMA DEL PIPELINE

### **Taller #1:**

```
- Diagrama Mermaid simple con 14 nodos
- Flujo lineal con un loop de retroalimentación
- Sin mencionar tecnologías específicas
- Sin separación clara de responsabilidades
- Sin incluir seguridad/gobernanza
```

### **Taller #3:**

```
+ Diagrama Mermaid completo con 50+ nodos
+ 9 subgrafos claramente separados:
  1. Ingesta y Almacenamiento
  2. Procesamiento y Feature Engineering
  3. Entrenamiento de Modelos
  4. Validación y Testing
  5. CI/CD Pipeline
  6. Despliegue en Producción
  7. Monitoreo y Observabilidad
  8. Reentrenamiento y Mejora Continua
  9. Seguridad y Gobernanza
+ Cada nodo especifica tecnología exacta
+ Conexiones explícitas entre componentes
+ Código de colores para diferentes tecnologías
+ Incluye tanto flujos batch como streaming
+ Muestra múltiples opciones de despliegue
```

---

## 🔄CAMBIOS EN LA DOCUMENTACIÓN

### **Taller #1:**

```
- README.md: 150 líneas, básico
- pipeline_design.md: 82 líneas, conceptual
- Sin especificaciones técnicas
- Sin justificaciones de decisiones
- Sin consideraciones de costos
- Sin métricas de éxito
```

### **Taller #3:**

```
+ pipeline_design_v2.md: 2000+ líneas, extremadamente detallado
+ Cada etapa con:
  - Tecnologías específicas con justificación
  - Código de ejemplo funcional
  - Configuraciones YAML completas
  - Suposiciones explícitas
  - Implicaciones de decisiones
+ Secciones nuevas:
  - Tabla comparativa de tecnologías
  - Estrategia para datos limitados
  - Modos de despliegue (local/híbrido/nube)
  - Métricas de éxito (técnicas, ML, negocio)
  - Consideraciones éticas (bias, fairness, privacidad)
  - Referencias y recursos
+ CHANGELOG.md completo (este documento)
+ README.md actualizado con referencias a v2
```

---

## 🔄CAMBIOS EN COSTOS ESTIMADOS

### **Taller #1:**

```
- No especificaba costos
- Solo mencionaba "computador local" o "servidor"
```

### **Taller #3:**

```
+ Estimación mensual de infraestructura Azure:
  - Azure Data Lake Storage: ~$100/mes (1 TB)
  - Databricks (Standard): ~$2,000/mes (cluster intermitente)
  - AKS (3 nodes Standard_D4s_v3): ~$800/mes
  - Azure Cosmos DB: ~$500/mes (autoscale 4000-8000 RU/s)
  - Application Insights: ~$200/mes
  - Azure DevOps: ~$100/mes (5 usuarios)
  - Otros servicios (Key Vault, Monitor, etc.): ~$300/mes
  
  TOTAL: ~$4,000-5,000/mes (startup)
  Escalado: ~$8,000-10,000/mes (producción con alto tráfico)

+ Alternativa económica (equipo pequeño):
  - Azure ML Endpoints en vez de AKS: ~$1,500/mes
  - Databricks Community Edition: gratis (limitado)
  - TOTAL: ~$2,000-3,000/mes

+ Versión local: $0 (solo hardware del médico)
```

---

## 🔄CAMBIOS EN SUPOSICIONES

### **Taller #1:**

```
- Pocas suposiciones explícitas
- Vagamente mencionadas en el texto
- Sin implicaciones claras
```

### **Taller #3:**

```
+ Sección completa "Suposiciones y Decisiones de Diseño"
+ Suposiciones sobre datos:
  - Disponibilidad (>10,000 casos comunes, <100 raros)
  - Calidad (70% con labels verificados)
  - Formato (HL7, FHIR estándar)
  - Temporalidad (snapshot vs series de tiempo)
+ Suposiciones sobre infraestructura:
  - Budget ($5K-10K/mes)
  - Equipo (2-3 ML engineers + 1 DevOps + médicos)
  - Tráfico (~10K predicciones/día, pico 100 req/s)
+ Suposiciones sobre regulación:
  - Clasificación como SaMD (FDA Clase II)
  - Cumplimiento HIPAA obligatorio
  - Responsabilidad del médico, no del sistema
+ Cada suposición con:
  - Implicación si es cierta
  - Plan B si no se cumple
```

---

## 🔄CAMBIOS EN ESTRATEGIA DE MODELADO

### **Taller #1:**

```
- "Modelo para enfermedades comunes" (sin detalles)
- "Modelo para enfermedades huérfanas" (sin detalles)
- "Modelo Ensemble" (sin especificar cómo)
```

### **Taller #3:**

```
+ Modelo 1 - Enfermedades Comunes:
  - Algoritmos: XGBoost, LightGBM, Random Forest
  - Datos: >10,000 casos por condición
  - Entrenamiento: GPU cluster (Standard_NC6s_v3)
  - Validación: 5-fold cross-validation estratificada
  - Métricas objetivo: Accuracy >85%, F1 >0.85

+ Modelo 2 - Enfermedades Huérfanas:
  - Estrategia: Few-shot learning + Transfer learning
  - Base model: Preentrenado en datos médicos generales
  - Fine-tuning: <100 ejemplos por condición
  - Técnicas: SMOTE, Siamese Networks, MAML
  - Prioridad: Alta sensibilidad (recall >95%)

+ Modelo 3 - Ensemble Final:
  - Arquitectura: Stacking con meta-learner
  - Lógica de agregación: Priorizar riesgo clínico
  - Regla: "El peor caso manda" (principio de precaución)
  - Código completo de implementación incluido

+ Hyperparameter Tuning:
  - Optuna con Bayesian optimization
  - 500 trials por experimento
  - Búsqueda distribuida en Databricks
  - Early stopping en validación

+ Transfer Learning:
  - Preentrenar en MIMIC-III (público)
  - Fine-tuning en datos propios
  - Aprovechar features compartidas
```

---

## 🛡️ CAMBIOS EN SEGURIDAD

### **Taller #1:**

```
- Mención de "privacidad" y "controles de acceso"
- Sin implementación concreta
- Sin cumplimiento normativo
```

### **Taller #3:**

```
+ Autenticación:
  - Azure AD con SSO
  - OAuth2/JWT tokens
  - MFA obligatorio para accesos críticos

+ Autorización:
  - RBAC con 4 roles definidos
  - Permisos granulares por endpoint
  - Principio de least privilege

+ Encriptación:
  - TLS 1.3 en tránsito
  - mTLS entre microservicios
  - AES-256 en reposo
  - Certificados en Key Vault

+ Secrets Management:
  - Azure Key Vault
  - Rotación automática
  - NUNCA en código/Git

+ Auditoría:
  - Logs de cada predicción
  - Retención 7 años (HIPAA)
  - Trazabilidad completa (who, what, when, where)

+ Cumplimiento:
  - HIPAA (BAA con Azure)
  - GDPR (derecho al olvido implementado)
  - FDA si aplica (trazabilidad completa)

+ Disaster Recovery:
  - RPO 1h, RTO 4h
  - Geo-replication 3 regiones
  - Failover automático
```

---

## 📈 CAMBIOS EN MÉTRICAS DE ÉXITO

### **Taller #1:**

```
- No definía métricas de éxito
- Solo mencionaba "validación cruzada" y "evaluación"
```

### **Taller #3:**

```
+ Métricas Técnicas:
  - Disponibilidad API: 99.9% SLA
  - Latencia p99: <500ms
  - Throughput: >100 req/s
  - Data drift detectado: <7 días
  - Cobertura de tests: >85%

+ Métricas de ML:
  - F1-score (comunes): >0.85
  - Recall (críticos): >0.95
  - Precision (leves): >0.80
  - NPV: >0.98

+ Métricas de Negocio:
  - Tiempo de diagnóstico: -30%
  - Satisfacción médicos: >4.0/5.0
  - Casos críticos detectados temprano: +25%
  - Falsos negativos críticos: <2%

+ Métricas de Fairness:
  - Disparate impact: <1.2
  - Equal opportunity diff: <0.05
  - Performance por subgrupo (edad, género)
```

---

## 🌍 NUEVAS CAPACIDADES

| Capacidad                             | Taller #1     | Taller #3                          |
| ------------------------------------- | ------------- | ---------------------------------- |
| **Escalabilidad horizontal**    | ❌ No         | ✅ AKS 3-20 pods autoscaling       |
| **Alta disponibilidad**         | ❌ No         | ✅ Multi-región con failover      |
| **Monitoreo en tiempo real**    | ❌ Mencionado | ✅ Prometheus + Grafana completo   |
| **Explicabilidad**              | ❌ No         | ✅ SHAP en cada predicción        |
| **Data drift detection**        | ❌ No         | ✅ Evidently AI automático        |
| **Reentrenamiento automático** | ❌ No         | ✅ Databricks Jobs con triggers    |
| **CI/CD**                       | ❌ No         | ✅ Azure DevOps 5 etapas           |
| **A/B Testing**                 | ❌ No         | ✅ Canary release gradual          |
| **Despliegue sin downtime**     | ❌ No         | ✅ Blue-Green + Canary             |
| **Feature Store**               | ❌ No         | ✅ Databricks Feature Store        |
| **Versioning de datos**         | ❌ No         | ✅ DVC + Delta Lake                |
| **Versioning de modelos**       | ❌ No         | ✅ MLflow Model Registry           |
| **Inferencia batch**            | ❌ No         | ✅ Azure Functions                 |
| **Multi-modo despliegue**       | ❌ Solo local | ✅ Local + Híbrido + Nube         |
| **Cumplimiento HIPAA/GDPR**     | ❌ Mencionado | ✅ Implementación completa        |
| **Disaster Recovery**           | ❌ No         | ✅ RPO 1h, RTO 4h, geo-replication |
| **API Documentation**           | ❌ No         | ✅ OpenAPI automática (FastAPI)   |
| **Rate Limiting**               | ❌ No         | ✅ Azure API Management            |
| **Caching**                     | ❌ No         | ✅ Cache 1h para diagnósticos     |
| **Feedback Loop**               | ❌ No         | ✅ Sistema formal médico feedback |

---

## 🎯 CONCLUSIÓN DE CAMBIOS

### **Impacto de los cambios:**

1. **De concepto a implementable:**

   - Taller #1 era una idea general
   - Taller #3 es un blueprint completo para implementar
2. **De local a enterprise:**

   - Taller #1 era solo Docker local
   - Taller #3 soporta desde médico rural hasta hospital nacional
3. **De monolítico a distribuido:**

   - Taller #1 era un servicio único
   - Taller #3 es arquitectura de microservicios escalable
4. **De reactivo a proactivo:**

   - Taller #1 no monitoreaba
   - Taller #3 detecta y alerta problemas antes de impacto
5. **De black-box a interpretable:**

   - Taller #1 no explicaba predicciones
   - Taller #3 provee SHAP values y justificaciones
6. **De inseguro a enterprise-grade:**

   - Taller #1 sin autenticación/encriptación
   - Taller #3 cumple HIPAA/GDPR/FDA
7. **De estático a adaptativo:**

   - Taller #1 sin reentrenamiento
   - Taller #3 reentrenamiento automático con triggers

---

## 📚 TECNOLOGÍAS AGREGADAS (30+)

**Taller #1:** Docker, Flask, Python básico (3 tecnologías)

**Taller #3:**

1. Azure Data Factory
2. Azure Data Lake Gen2
3. Delta Lake
4. Azure Event Hubs
5. Databricks
6. Apache Spark
7. Feature Store (Databricks)
8. Great Expectations
9. DVC
10. MLflow
11. Optuna
12. Azure Machine Learning
13. Scikit-learn, XGBoost, LightGBM, PyTorch
14. AutoML
15. Pytest
16. Deepchecks
17. SHAP / LIME
18. Azure DevOps / GitHub Actions
19. Docker (optimizado)
20. Azure Container Registry
21. Terraform
22. FastAPI
23. Azure Kubernetes Service
24. NGINX Ingress
25. Azure API Management
26. Azure Functions
27. Azure Monitor
28. Application Insights
29. Prometheus
30. Grafana
31. Evidently AI
32. Seldon Alibi
33. Azure Active Directory
34. Azure Key Vault
35. Azure Policy
36. Azure Purview
37. Azure Cosmos DB
38. Azure Logic Apps

**Total: 38 tecnologías específicas** (vs 3 en Taller #1)
-------------------------------------------------------

---

*Proyecto desarrollado por Felipe Guerra y Mavelyn Sterling para el Taller #3 de MLOps - Maestría en Inteligencia Artificial Aplicada*

*Versión 2.0 - Pipeline MLOps End-to-End de Nivel Empresarial*
