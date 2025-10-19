#!/usr/bin/env python3
"""
Script de configuración del entorno de desarrollo
Sistema de Diagnóstico Médico - MLOps Pipeline
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_message(message, color="blue"):
    """Imprime mensajes con color"""
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m", 
        "yellow": "\033[93m",
        "red": "\033[91m",
        "end": "\033[0m"
    }
    print(f"{colors.get(color, colors['blue'])}[INFO]{colors['end']} {message}")

def check_python_version():
    """Verifica la versión de Python"""
    print_message("Verificando versión de Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_message("❌ Se requiere Python 3.8 o superior", "red")
        print_message(f"Versión actual: {version.major}.{version.minor}.{version.micro}", "red")
        return False
    
    print_message(f"✅ Python {version.major}.{version.minor}.{version.micro} detectado", "green")
    return True

def create_virtual_environment():
    """Crea el entorno virtual"""
    print_message("Creando entorno virtual...")
    
    venv_path = Path(".venv")
    if venv_path.exists():
        print_message("⚠️  El entorno virtual ya existe", "yellow")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print_message("✅ Entorno virtual creado exitosamente", "green")
        return True
    except subprocess.CalledProcessError as e:
        print_message(f"❌ Error creando entorno virtual: {e}", "red")
        return False

def get_activation_command():
    """Obtiene el comando de activación según el SO"""
    if platform.system() == "Windows":
        return ".venv\\Scripts\\activate"
    else:
        return "source .venv/bin/activate"

def install_dependencies():
    """Instala las dependencias"""
    print_message("Instalando dependencias...")
    
    # Determinar el ejecutable de pip según el SO
    if platform.system() == "Windows":
        pip_executable = ".venv\\Scripts\\pip"
    else:
        pip_executable = ".venv/bin/pip"
    
    try:
        # Actualizar pip
        subprocess.run([pip_executable, "install", "--upgrade", "pip"], check=True)
        
        # Instalar dependencias
        subprocess.run([pip_executable, "install", "-r", "requirements.txt"], check=True)
        
        print_message("✅ Dependencias instaladas exitosamente", "green")
        return True
    except subprocess.CalledProcessError as e:
        print_message(f"❌ Error instalando dependencias: {e}", "red")
        return False

def create_activation_scripts():
    """Crea scripts de activación para diferentes sistemas"""
    print_message("Creando scripts de activación...")
    
    # Script para Windows
    windows_script = """@echo off
echo Activando entorno virtual...
call .venv\\Scripts\\activate.bat
echo Entorno virtual activado.
echo Para ejecutar la aplicación: python src/app.py
echo Para desactivar: deactivate
"""
    
    with open("activate_env.bat", "w") as f:
        f.write(windows_script)
    
    # Script para Linux/Mac
    unix_script = """#!/bin/bash
echo "Activando entorno virtual..."
source .venv/bin/activate
echo "Entorno virtual activado."
echo "Para ejecutar la aplicación: python src/app.py"
echo "Para desactivar: deactivate"
"""
    
    with open("activate_env.sh", "w") as f:
        f.write(unix_script)
    
    # Hacer ejecutable en sistemas Unix
    if platform.system() != "Windows":
        os.chmod("activate_env.sh", 0o755)
    
    print_message("✅ Scripts de activación creados", "green")

def verify_installation():
    """Verifica que la instalación sea correcta"""
    print_message("Verificando instalación...")
    
    try:
        # Verificar que Flask se puede importar
        if platform.system() == "Windows":
            python_executable = ".venv\\Scripts\\python"
        else:
            python_executable = ".venv/bin/python"
        
        result = subprocess.run([
            python_executable, "-c", 
            "import flask, pandas, numpy, sklearn; print('Todas las dependencias importadas correctamente')"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print_message("✅ Verificación exitosa", "green")
            print(result.stdout.strip())
            return True
        else:
            print_message("❌ Error en la verificación", "red")
            print(result.stderr)
            return False
            
    except Exception as e:
        print_message(f"❌ Error en verificación: {e}", "red")
        return False

def show_next_steps():
    """Muestra los próximos pasos"""
    activation_cmd = get_activation_command()
    
    print_message("🎉 ¡Configuración completada exitosamente!", "green")
    print("\n" + "="*60)
    print("PRÓXIMOS PASOS:")
    print("="*60)
    print(f"1. Activar entorno virtual:")
    print(f"   {activation_cmd}")
    print("\n2. Ejecutar la aplicación:")
    print("   python src/app.py")
    print("\n3. Acceder al servicio:")
    print("   http://localhost:5000")
    print("\n4. Ejecutar pruebas:")
    print("   python test_system.py")
    print("\n5. Construir con Docker:")
    print("   docker build -t medical-diagnosis-service .")
    print("   docker run -p 5000:5000 medical-diagnosis-service")
    print("\n" + "="*60)

def main():
    """Función principal"""
    print("🏥 Sistema de Diagnóstico Médico - Configuración de Desarrollo")
    print("="*70)
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Crear entorno virtual
    if not create_virtual_environment():
        sys.exit(1)
    
    # Instalar dependencias
    if not install_dependencies():
        sys.exit(1)
    
    # Crear scripts de activación
    create_activation_scripts()
    
    # Verificar instalación
    if not verify_installation():
        print_message("⚠️  La verificación falló, pero puedes intentar ejecutar la aplicación", "yellow")
    
    # Mostrar próximos pasos
    show_next_steps()

if __name__ == "__main__":
    main()
