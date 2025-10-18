"""
Módulo de diagnóstico médico para el sistema de MLOps
Desarrollado para el taller de Pipeline de MLOps + Docker
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDiagnosisModel:
    """
    Modelo de diagnóstico médico que simula la predicción de enfermedades
    basado en síntomas del paciente.
    
    Este modelo simula un sistema de ML real que podría incluir:
    - Modelos de deep learning para enfermedades comunes
    - Few-shot learning para enfermedades huérfanas
    - Ensemble methods para combinación de predicciones
    """
    
    def __init__(self):
        """Inicializa el modelo de diagnóstico médico"""
        self.symptom_weights = self._initialize_symptom_weights()
        self.disease_patterns = self._initialize_disease_patterns()
        self.severity_thresholds = self._initialize_severity_thresholds()
        
    def _initialize_symptom_weights(self) -> Dict[str, float]:
        """Inicializa los pesos de importancia de los síntomas"""
        return {
            'fiebre': 0.8,
            'dolor_cabeza': 0.6,
            'nausea': 0.5,
            'fatiga': 0.4,
            'dolor_pecho': 0.9,
            'dificultad_respirar': 0.95,
            'dolor_abdominal': 0.7,
            'mareos': 0.5,
            'perdida_peso': 0.6,
            'tos': 0.6,
            'congestion_nasal': 0.3,
            'dolor_garganta': 0.4,
            'dolor_muscular': 0.4,
            'dolor_articular': 0.5,
            'erupcion_cutanea': 0.6,
            'sangrado': 0.8,
            'cambios_vision': 0.7,
            'confusion': 0.9,
            'convulsiones': 0.95,
            'dolor_espalda': 0.5
        }
    
    def _initialize_disease_patterns(self) -> Dict[str, List[str]]:
        """Inicializa los patrones de síntomas para diferentes tipos de enfermedades"""
        return {
            'infeccion_respiratoria': ['fiebre', 'tos', 'congestion_nasal', 'dolor_garganta'],
            'gastroenteritis': ['nausea', 'dolor_abdominal', 'fatiga'],
            'migrana': ['dolor_cabeza', 'nausea', 'mareos'],
            'ansiedad': ['dolor_pecho', 'dificultad_respirar', 'mareos', 'fatiga'],
            'diabetes': ['perdida_peso', 'fatiga', 'cambios_vision'],
            'hipertension': ['dolor_cabeza', 'mareos', 'dolor_pecho'],
            'artritis': ['dolor_articular', 'dolor_muscular', 'fatiga'],
            'enfermedad_cardiaca': ['dolor_pecho', 'dificultad_respirar', 'fatiga'],
            'enfermedad_renal': ['fatiga', 'nausea', 'dolor_espalda'],
            'enfermedad_hepatica': ['fatiga', 'nausea', 'dolor_abdominal', 'erupcion_cutanea'],
            'enfermedad_autoimmune': ['fatiga', 'dolor_articular', 'erupcion_cutanea', 'fiebre'],
            'cancer': ['perdida_peso', 'fatiga', 'dolor_abdominal', 'sangrado'],
            'enfermedad_neurologica': ['confusion', 'convulsiones', 'cambios_vision', 'mareos']
        }
    
    def _initialize_severity_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """Inicializa los umbrales para determinar la severidad de la enfermedad"""
        return {
            'NO_ENFERMO': (0.0, 0.3),
            'ENFERMEDAD_LEVE': (0.3, 0.6),
            'ENFERMEDAD_AGUDA': (0.6, 0.8),
            'ENFERMEDAD_CRONICA': (0.8, 1.0)
        }
    
    def calculate_symptom_score(self, symptoms: Dict[str, Union[float, int]]) -> float:
        """
        Calcula un score basado en los síntomas del paciente
        
        Args:
            symptoms: Diccionario con síntomas y su intensidad (0-10)
            
        Returns:
            Score normalizado entre 0 y 1
        """
        total_score = 0.0
        total_weight = 0.0
        
        for symptom, intensity in symptoms.items():
            if symptom in self.symptom_weights:
                # Normalizar intensidad a 0-1
                normalized_intensity = min(max(intensity / 10.0, 0.0), 1.0)
                weight = self.symptom_weights[symptom]
                total_score += normalized_intensity * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
            
        return total_score / total_weight
    
    def detect_disease_patterns(self, symptoms: Dict[str, Union[float, int]]) -> Dict[str, float]:
        """
        Detecta patrones de enfermedades basado en los síntomas
        
        Args:
            symptoms: Diccionario con síntomas y su intensidad
            
        Returns:
            Diccionario con scores para cada patrón de enfermedad
        """
        pattern_scores = {}
        
        for disease, pattern_symptoms in self.disease_patterns.items():
            score = 0.0
            matched_symptoms = 0
            
            for symptom in pattern_symptoms:
                if symptom in symptoms and symptoms[symptom] > 0:
                    matched_symptoms += 1
                    # Normalizar intensidad y aplicar peso
                    intensity = min(max(symptoms[symptom] / 10.0, 0.0), 1.0)
                    weight = self.symptom_weights.get(symptom, 0.5)
                    score += intensity * weight
            
            # Normalizar por número de síntomas en el patrón
            if len(pattern_symptoms) > 0:
                pattern_scores[disease] = score / len(pattern_symptoms)
            else:
                pattern_scores[disease] = 0.0
                
        return pattern_scores
    
    def determine_severity(self, overall_score: float, pattern_scores: Dict[str, float]) -> str:
        """
        Determina la severidad de la enfermedad basado en el score general y patrones
        
        Args:
            overall_score: Score general de síntomas (0-1)
            pattern_scores: Scores de patrones de enfermedades
            
        Returns:
            Estado de la enfermedad: NO_ENFERMO, ENFERMEDAD_LEVE, ENFERMEDAD_AGUDA, ENFERMEDAD_CRONICA
        """
        # Ajustar score basado en patrones de enfermedades específicas
        max_pattern_score = max(pattern_scores.values()) if pattern_scores else 0.0
        adjusted_score = (overall_score + max_pattern_score) / 2.0
        
        # Determinar severidad basado en umbrales
        for severity, (min_threshold, max_threshold) in self.severity_thresholds.items():
            if min_threshold <= adjusted_score < max_threshold:
                return severity
        
        # Si el score es muy alto, clasificar como crónica
        if adjusted_score >= 1.0:
            return 'ENFERMEDAD_CRONICA'
        
        return 'NO_ENFERMO'
    
    def predict_diagnosis(self, symptoms: Dict[str, Union[float, int]]) -> Dict[str, Union[str, float, Dict]]:
        """
        Función principal para predecir el diagnóstico médico
        
        Args:
            symptoms: Diccionario con síntomas y su intensidad (0-10)
            
        Returns:
            Diccionario con el diagnóstico completo
        """
        try:
            # Validar entrada
            if not symptoms or len(symptoms) < 3:
                raise ValueError("Se requieren al menos 3 síntomas para el diagnóstico")
            
            # Calcular score general de síntomas
            overall_score = self.calculate_symptom_score(symptoms)
            
            # Detectar patrones de enfermedades
            pattern_scores = self.detect_disease_patterns(symptoms)
            
            # Determinar severidad
            severity = self.determine_severity(overall_score, pattern_scores)
            
            # Encontrar la enfermedad más probable
            most_likely_disease = max(pattern_scores.items(), key=lambda x: x[1]) if pattern_scores else ("ninguna", 0.0)
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(severity, most_likely_disease[0])
            
            result = {
                'diagnosis': severity,
                'confidence': round(overall_score, 3),
                'most_likely_condition': most_likely_disease[0],
                'condition_confidence': round(most_likely_disease[1], 3),
                'symptom_score': round(overall_score, 3),
                'pattern_scores': {k: round(v, 3) for k, v in pattern_scores.items()},
                'recommendations': recommendations,
                'input_symptoms': symptoms
            }
            
            logger.info(f"Diagnóstico generado: {severity} con confianza {overall_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en el diagnóstico: {str(e)}")
            return {
                'error': str(e),
                'diagnosis': 'ERROR',
                'confidence': 0.0
            }
    
    def _generate_recommendations(self, severity: str, condition: str) -> List[str]:
        """Genera recomendaciones basadas en la severidad y condición detectada"""
        recommendations = []
        
        if severity == 'NO_ENFERMO':
            recommendations = [
                "Continuar con el monitoreo regular de la salud",
                "Mantener hábitos de vida saludables",
                "Consultar si aparecen nuevos síntomas"
            ]
        elif severity == 'ENFERMEDAD_LEVE':
            recommendations = [
                "Monitorear síntomas de cerca",
                "Considerar consulta médica si los síntomas persisten",
                "Mantener reposo y hidratación adecuada",
                "Evitar actividades extenuantes"
            ]
        elif severity == 'ENFERMEDAD_AGUDA':
            recommendations = [
                "CONSULTA MÉDICA INMEDIATA RECOMENDADA",
                "Buscar atención médica en las próximas 24 horas",
                "Monitorear signos vitales regularmente",
                "Evitar automedicación",
                "Considerar visita a urgencias si empeora"
            ]
        elif severity == 'ENFERMEDAD_CRONICA':
            recommendations = [
                "CONSULTA MÉDICA URGENTE REQUERIDA",
                "Buscar atención especializada inmediatamente",
                "Posible hospitalización requerida",
                "Monitoreo médico continuo necesario",
                "Seguimiento con especialista recomendado"
            ]
        
        return recommendations

# Instancia global del modelo
diagnosis_model = MedicalDiagnosisModel()

def predict_medical_diagnosis(symptoms: Dict[str, Union[float, int]]) -> Dict[str, Union[str, float, Dict]]:
    """
    Función de conveniencia para realizar predicciones de diagnóstico médico
    
    Args:
        symptoms: Diccionario con síntomas y su intensidad (0-10)
                 Ejemplo: {'fiebre': 8, 'dolor_cabeza': 6, 'nausea': 4}
    
    Returns:
        Diccionario con el diagnóstico completo
    """
    return diagnosis_model.predict_diagnosis(symptoms)

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de síntomas de un paciente
    example_symptoms = {
        'fiebre': 8,
        'dolor_cabeza': 7,
        'nausea': 5,
        'fatiga': 6,
        'dolor_pecho': 3
    }
    
    # Realizar diagnóstico
    result = predict_medical_diagnosis(example_symptoms)
    
    print("=== DIAGNÓSTICO MÉDICO ===")
    print(f"Diagnóstico: {result['diagnosis']}")
    print(f"Confianza: {result['confidence']}")
    print(f"Condición más probable: {result['most_likely_condition']}")
    print(f"Confianza de condición: {result['condition_confidence']}")
    print("\nRecomendaciones:")
    for rec in result['recommendations']:
        print(f"- {rec}")
