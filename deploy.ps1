# Script de despliegue para el Sistema de Diagnóstico Médico
# Desarrollado para el taller de Pipeline de MLOps + Docker
# Versión PowerShell para Windows

param(
    [string]$Action = "deploy"
)

# Función para imprimir mensajes con color
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Función para verificar si Docker está instalado
function Test-Docker {
    Write-Info "Verificando instalación de Docker..."
    
    try {
        $dockerVersion = docker --version
        if ($LASTEXITCODE -ne 0) {
            throw "Docker no está instalado"
        }
        
        $dockerInfo = docker info 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Docker no está ejecutándose"
        }
        
        Write-Success "Docker está instalado y ejecutándose"
        return $true
    }
    catch {
        Write-Error "Docker no está disponible: $_"
        return $false
    }
}

# Función para limpiar recursos existentes
function Clear-Resources {
    Write-Info "Limpiando recursos existentes..."
    
    # Detener y eliminar contenedor existente
    $existingContainer = docker ps -a --format "table {{.Names}}" | Select-String "medical-diagnosis"
    if ($existingContainer) {
        Write-Info "Deteniendo contenedor existente..."
        docker stop medical-diagnosis 2>$null
        docker rm medical-diagnosis 2>$null
    }
    
    # Eliminar imagen existente
    $existingImage = docker images --format "table {{.Repository}}" | Select-String "medical-diagnosis-service"
    if ($existingImage) {
        Write-Info "Eliminando imagen existente..."
        docker rmi medical-diagnosis-service 2>$null
    }
    
    Write-Success "Limpieza completada"
}

# Función para construir la imagen
function Build-Image {
    Write-Info "Construyendo imagen Docker..."
    
    # Verificar que el Dockerfile existe
    if (-not (Test-Path "Dockerfile")) {
        Write-Error "Dockerfile no encontrado en el directorio actual"
        exit 1
    }
    
    # Construir la imagen
    docker build -t medical-diagnosis-service . --no-cache
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Imagen construida exitosamente"
        return $true
    } else {
        Write-Error "Error al construir la imagen"
        return $false
    }
}

# Función para ejecutar el contenedor
function Start-Container {
    Write-Info "Ejecutando contenedor..."
    
    # Crear directorio de logs si no existe
    if (-not (Test-Path "logs")) {
        New-Item -ItemType Directory -Path "logs" -Force | Out-Null
    }
    
    # Ejecutar el contenedor
    docker run -d `
        --name medical-diagnosis `
        -p 5000:5000 `
        -v "${PWD}/logs:/app/logs" `
        -v "${PWD}/data:/app/data:ro" `
        --restart unless-stopped `
        medical-diagnosis-service
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Contenedor ejecutándose"
        return $true
    } else {
        Write-Error "Error al ejecutar el contenedor"
        return $false
    }
}

# Función para verificar el estado del servicio
function Test-Service {
    Write-Info "Verificando estado del servicio..."
    
    # Esperar a que el servicio esté listo
    Write-Info "Esperando a que el servicio esté listo..."
    Start-Sleep -Seconds 10
    
    # Verificar health check
    for ($i = 1; $i -le 30; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -TimeoutSec 5 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Success "Servicio está funcionando correctamente"
                return $true
            }
        }
        catch {
            Write-Info "Intento $i/30: Esperando respuesta del servicio..."
            Start-Sleep -Seconds 2
        }
    }
    
    Write-Error "El servicio no respondió después de 60 segundos"
    Write-Info "Verificando logs del contenedor..."
    docker logs medical-diagnosis
    return $false
}

# Función para mostrar información del servicio
function Show-ServiceInfo {
    Write-Info "Información del servicio:"
    Write-Host ""
    Write-Host "🌐 Interfaz Web: http://localhost:5000" -ForegroundColor Cyan
    Write-Host "🔗 API Endpoint: http://localhost:5000/predict" -ForegroundColor Cyan
    Write-Host "❤️  Health Check: http://localhost:5000/health" -ForegroundColor Cyan
    Write-Host "📚 Documentación: http://localhost:5000/api/docs" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📊 Comandos útiles:" -ForegroundColor Yellow
    Write-Host "  Ver logs: docker logs medical-diagnosis"
    Write-Host "  Ver estado: docker ps"
    Write-Host "  Detener: docker stop medical-diagnosis"
    Write-Host "  Eliminar: docker rm medical-diagnosis"
    Write-Host ""
}

# Función para ejecutar pruebas
function Invoke-Tests {
    $response = Read-Host "¿Deseas ejecutar las pruebas del sistema? (y/n)"
    
    if ($response -match "^[Yy]$") {
        Write-Info "Ejecutando pruebas del sistema..."
        
        # Verificar si el script de prueba existe
        if (Test-Path "test_system.py") {
            python test_system.py
        } else {
            Write-Warning "Script de prueba no encontrado. Saltando pruebas."
        }
    } else {
        Write-Info "Saltando pruebas del sistema"
    }
}

# Función principal
function Start-Deployment {
    Write-Host "🏥 Sistema de Diagnóstico Médico - Script de Despliegue" -ForegroundColor Magenta
    Write-Host "==================================================" -ForegroundColor Magenta
    Write-Host ""
    
    # Verificar prerrequisitos
    if (-not (Test-Docker)) {
        exit 1
    }
    
    # Limpiar recursos existentes
    Clear-Resources
    
    # Construir imagen
    if (-not (Build-Image)) {
        exit 1
    }
    
    # Ejecutar contenedor
    if (-not (Start-Container)) {
        exit 1
    }
    
    # Verificar servicio
    if (-not (Test-Service)) {
        exit 1
    }
    
    # Mostrar información
    Show-ServiceInfo
    
    # Ejecutar pruebas (opcional)
    Invoke-Tests
    
    Write-Success "¡Despliegue completado exitosamente!"
    Write-Info "El servicio está disponible en http://localhost:5000"
}

# Manejo de acciones
switch ($Action.ToLower()) {
    "clean" {
        Clear-Resources
    }
    "build" {
        if (Test-Docker) {
            Build-Image
        }
    }
    "run" {
        if (Test-Docker) {
            Start-Container
            Test-Service
            Show-ServiceInfo
        }
    }
    "test" {
        Invoke-Tests
    }
    "logs" {
        docker logs -f medical-diagnosis
    }
    "stop" {
        docker stop medical-diagnosis
        Write-Success "Servicio detenido"
    }
    "restart" {
        docker restart medical-diagnosis
        Write-Success "Servicio reiniciado"
    }
    "status" {
        docker ps --filter name=medical-diagnosis
    }
    "deploy" {
        Start-Deployment
    }
    default {
        Write-Host "Uso: .\deploy.ps1 [clean|build|run|test|logs|stop|restart|status|deploy]"
        Write-Host "Acción por defecto: deploy"
    }
}
