#!/usr/bin/env python3
"""
Script de prueba para el Sistema de Diagnóstico Médico
Desarrollado para el taller de Pipeline de MLOps + Docker
"""

import requests
import json
import time
import sys
from typing import Dict, Any

# Configuración
BASE_URL = "http://localhost:5000"
TIMEOUT = 10

def test_health_check():
    """Prueba el endpoint de health check"""
    print("🔍 Probando health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check exitoso: {data['status']}")
            return True
        else:
            print(f"❌ Health check falló: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en health check: {e}")
        return False

def test_symptoms_endpoint():
    """Prueba el endpoint de síntomas disponibles"""
    print("🔍 Probando endpoint de síntomas...")
    try:
        response = requests.get(f"{BASE_URL}/symptoms", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Síntomas disponibles: {len(data['available_symptoms'])} síntomas")
            return True
        else:
            print(f"❌ Endpoint de síntomas falló: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en endpoint de síntomas: {e}")
        return False

def test_diagnosis(symptoms: Dict[str, int], expected_diagnosis: str = None):
    """Prueba el endpoint de diagnóstico"""
    print(f"🔍 Probando diagnóstico con síntomas: {symptoms}")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=symptoms,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            diagnosis = data.get('diagnosis', 'ERROR')
            confidence = data.get('confidence', 0.0)
            
            print(f"✅ Diagnóstico: {diagnosis} (Confianza: {confidence:.3f})")
            
            if expected_diagnosis and diagnosis == expected_diagnosis:
                print(f"✅ Diagnóstico coincide con el esperado: {expected_diagnosis}")
            elif expected_diagnosis:
                print(f"⚠️  Diagnóstico no coincide. Esperado: {expected_diagnosis}, Obtenido: {diagnosis}")
            
            return True
        else:
            print(f"❌ Diagnóstico falló: {response.status_code}")
            print(f"Respuesta: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error en diagnóstico: {e}")
        return False

def load_test_cases():
    """Carga casos de prueba desde el archivo JSON"""
    try:
        with open('data/sample_symptoms.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['sample_cases']
    except Exception as e:
        print(f"❌ Error cargando casos de prueba: {e}")
        return []

def run_test_cases():
    """Ejecuta todos los casos de prueba"""
    print("🧪 Ejecutando casos de prueba...")
    test_cases = load_test_cases()
    
    if not test_cases:
        print("❌ No se pudieron cargar casos de prueba")
        return False
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Caso {i}: {case['case_id']} ---")
        print(f"Descripción: {case['description']}")
        
        success = test_diagnosis(
            case['symptoms'], 
            case['expected_diagnosis']
        )
        
        if success:
            passed += 1
        
        time.sleep(1)  # Pausa entre pruebas
    
    print(f"\n📊 Resultados: {passed}/{total} casos pasaron")
    return passed == total

def test_api_documentation():
    """Prueba el endpoint de documentación de la API"""
    print("🔍 Probando documentación de la API...")
    try:
        response = requests.get(f"{BASE_URL}/api/docs", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Documentación de API disponible: {data['title']}")
            return True
        else:
            print(f"❌ Documentación de API falló: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en documentación de API: {e}")
        return False

def test_error_handling():
    """Prueba el manejo de errores"""
    print("🔍 Probando manejo de errores...")
    
    # Prueba con datos inválidos
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={},  # Datos vacíos
            timeout=TIMEOUT
        )
        
        if response.status_code == 400:
            print("✅ Manejo de datos vacíos correcto")
        else:
            print(f"⚠️  Respuesta inesperada para datos vacíos: {response.status_code}")
        
        # Prueba con endpoint inexistente
        response = requests.get(f"{BASE_URL}/nonexistent", timeout=TIMEOUT)
        if response.status_code == 404:
            print("✅ Manejo de endpoint inexistente correcto")
        else:
            print(f"⚠️  Respuesta inesperada para endpoint inexistente: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Error en pruebas de manejo de errores: {e}")
        return False

def main():
    """Función principal del script de prueba"""
    print("🏥 Iniciando pruebas del Sistema de Diagnóstico Médico")
    print("=" * 60)
    
    # Lista de pruebas
    tests = [
        ("Health Check", test_health_check),
        ("Síntomas Disponibles", test_symptoms_endpoint),
        ("Documentación API", test_api_documentation),
        ("Manejo de Errores", test_error_handling),
        ("Casos de Prueba", run_test_cases)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Ejecutando: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name}: PASÓ")
            else:
                print(f"❌ {test_name}: FALLÓ")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
        
        time.sleep(1)  # Pausa entre pruebas
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)
    print(f"Pruebas pasadas: {passed_tests}/{total_tests}")
    print(f"Porcentaje de éxito: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 ¡Todas las pruebas pasaron exitosamente!")
        sys.exit(0)
    else:
        print("⚠️  Algunas pruebas fallaron. Revisar logs para más detalles.")
        sys.exit(1)

if __name__ == "__main__":
    main()
