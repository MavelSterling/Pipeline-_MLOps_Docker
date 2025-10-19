#!/bin/bash

# Script de despliegue para el Sistema de Diagnóstico Médico
# Desarrollado para el taller de Pipeline de MLOps + Docker

set -e  # Salir si cualquier comando falla

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes con color
print_message() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Función para verificar si Docker está instalado
check_docker() {
    print_message "Verificando instalación de Docker..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker no está instalado. Por favor instala Docker primero."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker no está ejecutándose. Por favor inicia Docker primero."
        exit 1
    fi
    
    print_success "Docker está instalado y ejecutándose"
}

# Función para limpiar recursos existentes
cleanup() {
    print_message "Limpiando recursos existentes..."
    
    # Detener y eliminar contenedor existente
    if docker ps -a --format 'table {{.Names}}' | grep -q "medical-diagnosis"; then
        print_message "Deteniendo contenedor existente..."
        docker stop medical-diagnosis 2>/dev/null || true
        docker rm medical-diagnosis 2>/dev/null || true
    fi
    
    # Eliminar imagen existente
    if docker images --format 'table {{.Repository}}' | grep -q "medical-diagnosis-service"; then
        print_message "Eliminando imagen existente..."
        docker rmi medical-diagnosis-service 2>/dev/null || true
    fi
    
    print_success "Limpieza completada"
}

# Función para construir la imagen
build_image() {
    print_message "Construyendo imagen Docker..."
    
    # Verificar que el Dockerfile existe
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile no encontrado en el directorio actual"
        exit 1
    fi
    
    # Construir la imagen
    docker build -t medical-diagnosis-service . --no-cache
    
    if [ $? -eq 0 ]; then
        print_success "Imagen construida exitosamente"
    else
        print_error "Error al construir la imagen"
        exit 1
    fi
}

# Función para ejecutar el contenedor
run_container() {
    print_message "Ejecutando contenedor..."
    
    # Crear directorio de logs si no existe
    mkdir -p logs
    
    # Ejecutar el contenedor
    docker run -d \
        --name medical-diagnosis \
        -p 5000:5000 \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/data:/app/data:ro" \
        --restart unless-stopped \
        medical-diagnosis-service
    
    if [ $? -eq 0 ]; then
        print_success "Contenedor ejecutándose"
    else
        print_error "Error al ejecutar el contenedor"
        exit 1
    fi
}

# Función para verificar el estado del servicio
verify_service() {
    print_message "Verificando estado del servicio..."
    
    # Esperar a que el servicio esté listo
    print_message "Esperando a que el servicio esté listo..."
    sleep 10
    
    # Verificar health check
    for i in {1..30}; do
        if curl -f http://localhost:5000/health &>/dev/null; then
            print_success "Servicio está funcionando correctamente"
            return 0
        fi
        print_message "Intento $i/30: Esperando respuesta del servicio..."
        sleep 2
    done
    
    print_error "El servicio no respondió después de 60 segundos"
    print_message "Verificando logs del contenedor..."
    docker logs medical-diagnosis
    exit 1
}

# Función para mostrar información del servicio
show_service_info() {
    print_message "Información del servicio:"
    echo ""
    echo "🌐 Interfaz Web: http://localhost:5000"
    echo "🔗 API Endpoint: http://localhost:5000/predict"
    echo "❤️  Health Check: http://localhost:5000/health"
    echo "📚 Documentación: http://localhost:5000/api/docs"
    echo ""
    echo "📊 Comandos útiles:"
    echo "  Ver logs: docker logs medical-diagnosis"
    echo "  Ver estado: docker ps"
    echo "  Detener: docker stop medical-diagnosis"
    echo "  Eliminar: docker rm medical-diagnosis"
    echo ""
}

# Función para ejecutar pruebas
run_tests() {
    print_message "¿Deseas ejecutar las pruebas del sistema? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_message "Ejecutando pruebas del sistema..."
        
        # Verificar si el script de prueba existe
        if [ -f "test_system.py" ]; then
            python3 test_system.py
        else
            print_warning "Script de prueba no encontrado. Saltando pruebas."
        fi
    else
        print_message "Saltando pruebas del sistema"
    fi
}

# Función principal
main() {
    echo "🏥 Sistema de Diagnóstico Médico - Script de Despliegue"
    echo "=================================================="
    echo ""
    
    # Verificar prerrequisitos
    check_docker
    
    # Limpiar recursos existentes
    cleanup
    
    # Construir imagen
    build_image
    
    # Ejecutar contenedor
    run_container
    
    # Verificar servicio
    verify_service
    
    # Mostrar información
    show_service_info
    
    # Ejecutar pruebas (opcional)
    run_tests
    
    print_success "¡Despliegue completado exitosamente!"
    print_message "El servicio está disponible en http://localhost:5000"
}

# Manejo de argumentos
case "${1:-}" in
    "clean")
        cleanup
        ;;
    "build")
        check_docker
        build_image
        ;;
    "run")
        check_docker
        run_container
        verify_service
        show_service_info
        ;;
    "test")
        run_tests
        ;;
    "logs")
        docker logs -f medical-diagnosis
        ;;
    "stop")
        docker stop medical-diagnosis
        print_success "Servicio detenido"
        ;;
    "restart")
        docker restart medical-diagnosis
        print_success "Servicio reiniciado"
        ;;
    "status")
        docker ps --filter name=medical-diagnosis
        ;;
    *)
        main
        ;;
esac
