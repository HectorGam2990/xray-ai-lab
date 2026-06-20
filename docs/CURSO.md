# Curso guiado: cómo pensar y construir un clasificador de imágenes

## 1. El mapa mental

Un clasificador supervisado aprende una función:

```text
imagen (píxeles) → modelo con parámetros → puntuación por clase
```

Durante el entrenamiento conocemos la respuesta correcta. La **función de pérdida** mide el
error; la retropropagación calcula qué parámetros contribuyeron a él; el optimizador modifica
esos parámetros. Repetir este ciclo no garantiza que el modelo comprenda medicina: solo que
encuentra patrones útiles para las etiquetas que recibió.

El flujo se repite en casi cualquier detector de IA:

1. Definir exactamente la entrada, la salida y el uso permitido.
2. Obtener y dividir datos representativos.
3. Aplicar el mismo preprocesamiento de forma reproducible.
4. Entrenar con una pérdida y un optimizador.
5. Evaluar en datos nunca usados para ajustar el modelo.
6. Guardar modelo, etiquetas y configuración juntos.
7. Servir predicciones y vigilar errores/cambios de distribución.

## 2. Recorrido archivo por archivo

### `data.py`: convertir evidencia en tensores

`ImageFolder` usa el nombre de cada subcarpeta como etiqueta. Así evitamos escribir lógica
especial para `NORMAL`, `PNEUMONIA` u otra clase. `Grayscale(3)` conserva la apariencia gris
pero produce tres canales, porque EfficientNet fue preentrenada con RGB. `Resize` estandariza
el tamaño; `ToTensor` convierte píxeles a números; `Normalize` usa la escala esperada por los
pesos preentrenados.

`DataLoader` agrupa ejemplos en **batches**. Un lote aprovecha operaciones vectorizadas y evita
cargar todo el conjunto en memoria. `shuffle=True` cambia el orden al entrenar. Nunca se mezcla
la validación: medir debe ser determinista.

La rotación aleatoria es **data augmentation**: crea pequeñas variaciones plausibles para
reducir memorización. En medicina no toda transformación visual es válida; un volteo puede
cambiar lateralidad y una rotación extrema puede fabricar anatomía imposible.

`balanced_class_weights` hace más costoso equivocarse en una clase escasa. Es mejor que mirar
solo exactitud: si 95 % de imágenes son normales, responder siempre “normal” logra 95 % sin
aprender la enfermedad.

### `model.py`: transferencia de aprendizaje

EfficientNet-B0 ya aprendió bordes, texturas y formas sobre muchas imágenes. Sustituimos su
última capa por una nueva de `N` salidas. Al principio congelamos el **backbone** y entrenamos
solo esa cabeza: requiere menos cómputo y menos datos.

Después de conseguir una línea base puedes usar `--unfreeze` con una tasa de aprendizaje menor
(por ejemplo `1e-4`) para *fine-tuning*. No empieces allí: más parámetros entrenables también
permiten sobreajustar más rápido.

### `train.py`: aprender sin mezclar responsabilidades

`seed_everything` reduce variación entre experimentos. No promete reproducibilidad absoluta en
todo hardware, pero vuelve las comparaciones mucho más justas.

`run_epoch` tiene dos modos:

- Con optimizador: activa gradientes, ejecuta `backward` y actualiza pesos.
- Sin optimizador: valida sin modificar el modelo.

Los **logits** son puntuaciones sin normalizar. `CrossEntropyLoss` aplica internamente la
operación adecuada; por eso no debemos llamar `softmax` antes de la pérdida. `AdamW` es el
optimizador: decide el tamaño y dirección acumulada de cada actualización. `zero_grad` borra
gradientes del lote anterior porque PyTorch los acumula por diseño.

Una **epoch** es una pasada por todo entrenamiento. Al final medimos validación y guardamos el
checkpoint solo si baja su pérdida. El checkpoint incluye pesos y orden de etiquetas: guardar
solo el modelo sin el significado de sus salidas es un error de producción muy común.

### `metrics.py`: medir el error que importa

La matriz de confusión cuenta cada combinación `real → predicho`. De ella salen:

- **Recall/sensibilidad:** de los casos reales de una clase, cuántos detectamos.
- **Precisión/valor predictivo positivo:** de lo que marcamos como esa clase, cuánto acertamos.
- **F1:** equilibrio entre precisión y recall.
- **Macro:** promedio donde cada clase pesa lo mismo, aunque una sea minoritaria.

La exactitud sola casi nunca basta. En un estudio serio también necesitas especificidad,
ROC-AUC, PR-AUC, calibración, intervalos de confianza y resultados por subgrupo. Además, el
umbral de decisión debe elegirse según el costo de falsos positivos y falsos negativos; `argmax`
solo es una línea base.

### `predict.py`: inferencia reproducible

Reconstruimos la misma arquitectura, cargamos pesos y aplicamos la misma transformación. La
decoración `inference_mode` desactiva el cálculo de gradientes, lo que reduce memoria y tiempo.
`softmax` convierte logits en números que suman uno. Son probabilidades del modelo, **no certeza
médica**; una red puede estar muy segura y equivocarse.

### `app.py`: interfaz, no inteligencia nueva

Streamlit solo recibe la imagen y presenta el resultado. `cache_resource` evita recargar el
modelo en cada interacción. Mantener la interfaz separada del modelo permite reutilizar el
mismo predictor en una API, una app o un proceso por lotes.

### `demo_data.py`: probar el sistema antes de confiar en los datos

Genera círculos y cuadrados. Estos datos no demuestran utilidad médica; sirven para comprobar
que carga, entrenamiento, guardado e inferencia funcionan de extremo a extremo. Probar primero
con un problema controlado es una costumbre senior: reduce la cantidad de causas posibles de
un fallo.

## 3. Conceptos básicos

- **Muestra:** una imagen.
- **Feature:** señal numérica usada por el modelo; al inicio son píxeles, después patrones.
- **Etiqueta:** respuesta que queremos predecir.
- **Parámetro/peso:** número que el entrenamiento ajusta.
- **Hiperparámetro:** decisión nuestra, como tasa de aprendizaje o batch size.
- **Entrenamiento:** datos que modifican pesos.
- **Validación:** datos que ayudan a elegir configuración/checkpoint.
- **Test:** evaluación final, usada una vez después de decidir el sistema.
- **Underfitting:** ni entrenamiento funciona; falta capacidad, tiempo o señal.
- **Overfitting:** entrenamiento funciona y validación no; el modelo memorizó atajos.

## 4. Conceptos intermedios

### Fuga de datos

Ocurre cuando información del futuro o de validación entra en entrenamiento. En radiografías,
varias imágenes del mismo paciente, marcas del hospital o duplicados pueden revelar la respuesta.
La solución comienza con identificadores de paciente y una partición previa a copiar imágenes.

### Desbalance y prevalencia

La proporción de positivos afecta precisión y valor predictivo. Un modelo evaluado en un conjunto
50/50 puede comportarse muy distinto en una clínica donde la enfermedad es rara. Reporta la
prevalencia y simula el contexto de uso previsto.

### Regularización

Congelar capas, aumentar datos, `weight_decay`, *label smoothing* y detener temprano limitan la
memorización. Ninguna técnica reemplaza mejores datos.

### Reproducibilidad

Registra versión de código, dependencias, semilla, datos, hiperparámetros y checkpoint. Un número
sin el experimento que lo produjo no es conocimiento reproducible.

## 5. Conceptos avanzados para un proyecto médico real

1. **Ground truth:** “cáncer” suele requerir confirmación clínica/patológica; un reporte textual
   ruidoso no necesariamente es verdad absoluta.
2. **Validación externa:** prueba hospitales, equipos, fechas y poblaciones no presentes al
   desarrollar. Un split aleatorio interno es insuficiente.
3. **Calibración:** si el modelo dice 80 % en cien casos similares, aproximadamente ochenta
   deberían ser positivos. Softmax no lo garantiza.
4. **Cambio de distribución:** nuevos equipos, protocolos o prevalencias degradan modelos.
5. **Atajos espurios:** la red puede mirar letras, tubos, bordes o compresión en vez de anatomía.
6. **Equidad:** compara sensibilidad y falsos positivos por sexo, edad, origen y otros grupos
   pertinentes, con tamaños e incertidumbre adecuados.
7. **Explicabilidad:** mapas de calor ayudan a investigar, pero no prueban razonamiento causal.
8. **Incertidumbre y abstención:** un sistema responsable sabe enviar casos dudosos a revisión.
9. **Monitoreo:** rendimiento, entrada y calibración deben vigilarse después del despliegue.
10. **Regulación y seguridad:** investigación, prototipo y dispositivo clínico son categorías muy
    diferentes. Intervienen especialistas médicos, privacidad, gestión de riesgo y autoridades.

## 6. Cómo piensa un programador senior

La meta no es el menor número de líneas; es el menor sistema que siga siendo correcto, legible,
comprobable y modificable. Buenas decisiones presentes aquí:

- Una sola función para entrenar y validar, parametrizada por la presencia del optimizador.
- Etiquetas descubiertas por carpetas, no condiciones repetidas por enfermedad.
- Transformaciones centralizadas para impedir divergencia entre entrenamiento e inferencia.
- Checkpoint autocontenido con pesos y etiquetas.
- Dependencias declaradas y CI automático.
- Funciones cortas con una responsabilidad y comentarios sobre el **porqué**, no cada sintaxis.

Evita comprimir tres ideas en una línea si dificulta depuración. Las líneas eliminadas solo son
una mejora cuando también eliminan estados, duplicación o posibilidades de error.

## 7. Ruta de aprendizaje práctica

1. Ejecuta datos sintéticos y dos epochs. Lee cada JSON de métricas.
2. Cambia `batch-size`, `lr` y epochs de uno en uno; registra resultados.
3. Provoca desbalance generando menos ejemplos de una clase y observa los pesos.
4. Añade un conjunto `test` y una función de evaluación final.
5. Implementa especificidad y ROC-AUC con pruebas unitarias.
6. Usa un conjunto público autorizado, dividido por paciente, y crea una *data card*.
7. Compara EfficientNet contra un baseline sencillo y explica por qué gana o pierde.
8. Añade calibración, umbral configurable y una opción “abstenerse”.

La señal de progreso no es que el modelo entregue un porcentaje bonito. Es que puedes explicar
de dónde salió, qué error comete, qué evidencia falta y bajo qué condiciones dejaría de confiarse.
