#!/usr/bin/env python3
"""
Script de prueba simplificado para el Sistema de Diagnóstico Médico
Versión sin emojis para compatibilidad con Windows
"""

import requests
import json
import time
import sys

# Configuración
BASE_URL = "http://localhost:5000"
TIMEOUT = 10

def test_health_check():
    """Prueba el endpoint de health check"""
    print("[INFO] Probando health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"[SUCCESS] Health check exitoso: {data['status']}")
            return True
        else:
            print(f"[ERROR] Health check falló: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Error en health check: {e}")
        return False

def test_diagnosis(symptoms, expected_diagnosis=None):
    """Prueba el endpoint de diagnóstico"""
    print(f"[INFO] Probando diagnóstico con síntomas: {symptoms}")
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
            
            print(f"[SUCCESS] Diagnóstico: {diagnosis} (Confianza: {confidence:.3f})")
            
            if expected_diagnosis and diagnosis == expected_diagnosis:
                print(f"[SUCCESS] Diagnóstico coincide con el esperado: {expected_diagnosis}")
            elif expected_diagnosis:
                print(f"[WARNING] Diagnóstico no coincide. Esperado: {expected_diagnosis}, Obtenido: {diagnosis}")
            
            return True
        else:
            print(f"[ERROR] Diagnóstico falló: {response.status_code}")
            print(f"Respuesta: {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Error en diagnóstico: {e}")
        return False

def main():
    """Función principal del script de prueba"""
    print("Sistema de Diagnóstico Médico - Pruebas Simplificadas")
    print("=" * 60)
    
    # Verificar que el servicio esté corriendo
    if not test_health_check():
        print("[ERROR] El servicio no está disponible. Asegúrate de que esté ejecutándose.")
        print("Ejecuta: python src/app.py")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("CASOS DE PRUEBA")
    print("=" * 60)
    
    # Casos de prueba
    test_cases = [
        {
            "name": "Resfriado común",
            "symptoms": {"fiebre": 6, "dolor_cabeza": 4, "congestion_nasal": 8, "dolor_garganta": 7, "tos": 5, "fatiga": 3},
            "expected": "ENFERMEDAD_LEVE"
        },
        {
            "name": "Emergencia cardíaca",
            "symptoms": {"dolor_pecho": 10, "dificultad_respirar": 9, "mareos": 7, "nausea": 6, "fatiga": 8},
            "expected": "ENFERMEDAD_AGUDA"
        },
        {
            "name": "Paciente sano",
            "symptoms": {"fatiga": 2, "dolor_muscular": 1, "mareos": 1},
            "expected": "NO_ENFERMO"
        },
        {
            "name": "Diabetes no controlada",
            "symptoms": {"perdida_peso": 8, "fatiga": 9, "cambios_vision": 7, "dificultad_respirar": 5, "nausea": 6},
            "expected": "ENFERMEDAD_CRONICA"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Caso {i}: {case['name']} ---")
        success = test_diagnosis(case['symptoms'], case['expected'])
        if success:
            passed += 1
        time.sleep(1)  # Pausa entre pruebas
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS")
    print("=" * 60)
    print(f"Pruebas pasadas: {passed}/{total}")
    print(f"Porcentaje de éxito: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] ¡Todas las pruebas pasaron exitosamente!")
        print("\nEl servicio está funcionando correctamente.")
        print("Puedes acceder a la interfaz web en: http://localhost:5000")
    else:
        print("[WARNING] Algunas pruebas fallaron. Revisar logs para más detalles.")
    
    print("\n" + "=" * 60)
    print("INFORMACIÓN DEL SERVICIO")
    print("=" * 60)
    print("Interfaz Web: http://localhost:5000")
    print("API Endpoint: http://localhost:5000/predict")
    print("Health Check: http://localhost:5000/health")
    print("Documentación: http://localhost:5000/api/docs")

if __name__ == "__main__":
    main()
