# XRay AI Lab

Un laboratorio pequeño y legible para aprender **clasificación de imágenes con IA**. Usa
PyTorch, transferencia de aprendizaje con EfficientNet-B0, métricas por clase y una interfaz
Streamlit. El ejemplo sugerido es `NORMAL/PNEUMONIA`, pero el código descubre automáticamente
cualquier número de clases a partir de carpetas.

> [!WARNING]
> Es software educativo, no un diagnóstico ni un dispositivo médico. No uses sus resultados
> para decisiones clínicas. Una radiografía por sí sola tampoco demuestra que exista cáncer.

## Qué vas a aprender

1. Convertir imágenes y etiquetas en lotes numéricos.
2. Reutilizar una red preentrenada (*transfer learning*).
3. Entrenar, validar, medir y guardar solo el mejor modelo.
4. Cargar exactamente el mismo modelo para una predicción nueva.
5. Reconocer problemas reales: fuga de datos, desbalance, sobreajuste y cambio de distribución.

La explicación completa, archivo por archivo y desde nivel básico hasta avanzado, está en
[`docs/CURSO.md`](docs/CURSO.md). Los límites del modelo están en [`MODEL_CARD.md`](MODEL_CARD.md).

## Inicio rápido en PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"

# Datos falsos seguros: círculos contra cuadrados.
.\.venv\Scripts\xray-demo-data

# La primera ejecución descarga los pesos preentrenados.
.\.venv\Scripts\xray-train --data data/demo --epochs 2

# Interfaz visual.
.\.venv\Scripts\streamlit run src/xray_ai/app.py
```

También puedes predecir desde la terminal:

```powershell
.\.venv\Scripts\xray-predict data/demo/val/CIRCLE/000.png
```

## Formato de un conjunto real

```text
data/real/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

Luego ejecuta:

```powershell
.\.venv\Scripts\xray-train --data data/real --epochs 10
```

Separa `train`, `val` y `test` **por paciente**, no mezclando imágenes al azar. Si una persona
aparece en entrenamiento y validación, el modelo puede memorizarla y producir métricas falsas.
No subas el directorio `data/` a GitHub; ya está ignorado.

## Arquitectura

```text
carpetas de imágenes → transformaciones → EfficientNet → logits → pérdida/métricas
                                                        ↓
                                              models/best.pt
                                                        ↓
                                          CLI o aplicación Streamlit
```

## Comandos de calidad

```powershell
.\.venv\Scripts\ruff check .
.\.venv\Scripts\pytest
```

GitHub Actions ejecuta ambos en cada `push` y *pull request*.

## Fuentes para profundizar

- [Documentación oficial de EfficientNet-B0 en torchvision](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html)
- [Tutorial oficial de transferencia de aprendizaje de PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Colección oficial NIH ChestX-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [Dispositivos médicos con IA: FDA](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices)

Antes de descargar un conjunto, revisa su licencia, documentación, procedencia y calidad de
etiquetas. Que los datos sean públicos no convierte automáticamente cualquier uso en válido.

## Licencia

Código MIT. Los conjuntos de datos conservan sus propias licencias.

