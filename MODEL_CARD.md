# Model card: XRay AI Lab

## Estado

Prototipo educativo. **No es un dispositivo médico, no está validado clínicamente y no debe
utilizarse para tomar decisiones sobre pacientes.**

## Uso previsto

- Aprender clasificación supervisada de imágenes.
- Practicar transferencia de aprendizaje, evaluación e inferencia.
- Probar el pipeline primero con imágenes sintéticas y después con un conjunto autorizado.

## Usos fuera de alcance

- Diagnóstico, triaje o recomendación de tratamiento.
- Uso con datos personales sin autorización, desidentificación y controles adecuados.
- Afirmar que una clase predicha prueba la presencia o ausencia de una enfermedad.

## Datos

El repositorio no distribuye radiografías. Quien entrene debe documentar origen, licencia,
criterios de inclusión, equipo/hospital, demografía, prevalencia y división por paciente.

## Métricas mínimas antes de cualquier investigación seria

Reportar matriz de confusión, sensibilidad/recall por clase, especificidad, precisión, F1,
ROC-AUC/PR-AUC, calibración e intervalos de confianza. Evaluar además un hospital externo.

## Riesgos conocidos

Fuga de pacientes entre particiones, etiquetas ruidosas, atajos visuales, desbalance,
cambio de distribución, falta de calibración, sesgo demográfico y confianza excesiva.

