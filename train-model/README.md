# 🚀 Entrenamiento del Modelo de Anti-Spoofing

Este directorio contiene todo lo necesario para entrenar un modelo de Machine Learning que distingue entre selfies reales y falsos (spoofed/fake).

## 📁 Estructura

```
train-model/
├── README.md                          # Este archivo
├── GUIA_ENTRENAMIENTO_ANTISPOOFING.md # Guía detallada del proceso
├── train_antispoofing.py              # Script principal de entrenamiento
└── requirements.txt                   # Dependencias Python
```

## 📋 Requisitos Previos

### 1. Estructura del Dataset

El script espera que el dataset esté organizado de la siguiente manera:

```
Dataset/
├── Train/
│   ├── Real/          # Selfies reales (70% del total)
│   └── Fake/          # Selfies falsos/spoofed (70% del total)
├── Validation/
│   ├── Real/          # 15% del total
│   └── Fake/          # 15% del total
└── Test/
    ├── Real/          # 15% del total
    └── Fake/          # 15% del total
```

**Nota**: El script busca el dataset en `../Dataset` por defecto (relativo a esta carpeta).

### 2. Requisitos del Dataset

- **Mínimo recomendado**: 5,000 imágenes por clase (Real/Fake)
- **Ideal**: 10,000+ imágenes por clase
- **Balance**: 50/50 entre Real y Fake
- **Resolución mínima**: 224x224 píxeles
- **Formatos**: JPG, PNG

### 3. Hardware Recomendado

- **GPU**: NVIDIA con 8GB+ RAM (RTX 3060 o superior) - **Recomendado**
- **CPU**: Funciona pero será 4-6x más lento
- **RAM**: 16GB+ recomendado
- **Espacio en disco**: ~2GB para el modelo entrenado

## 🔧 Instalación

### 1. Crear entorno virtual (recomendado)

```bash
# Desde la raíz del proyecto
cd train-model

# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
# En macOS/Linux:
source venv/bin/activate
# En Windows:
# venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
# Instalar PyTorch según tu sistema
# Para CPU:
pip install torch torchvision

# Para GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Para GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Instalar otras dependencias
pip install -r requirements.txt
```

### 3. Verificar instalación

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

## 🎯 Uso

### 1. Verificar Dataset (Recomendado)

Antes de entrenar, verifica que tu dataset esté correctamente estructurado:

```bash
# Desde la carpeta train-model
python verify_dataset.py
```

Este script verificará:
- ✅ Estructura de carpetas (Train/Validation/Test con Real/Fake)
- ✅ Número de imágenes por clase
- ✅ Balance entre clases
- ✅ Requisitos mínimos (5,000 imágenes/clase)
- ✅ Validez de las imágenes

### 2. Ejecución Básica del Entrenamiento

```bash
# Desde la carpeta train-model
python train_antispoofing.py
```

El script automáticamente:
- Busca el dataset en `../Dataset`
- Carga Train, Validation y Test
- Entrena el modelo EfficientNet-B2
- Guarda el mejor modelo en `./models/`
- Evalúa en el conjunto de test
- Exporta el modelo a TorchScript (`.pt`) para producción

### Opciones Avanzadas

```bash
# Especificar ruta del dataset
python train_antispoofing.py --dataset-root /ruta/al/Dataset

# Ajustar batch size (reducir si hay error de memoria)
python train_antispoofing.py --batch-size 16

# Cambiar número de épocas
python train_antispoofing.py --epochs 100

# Cambiar learning rate
python train_antispoofing.py --learning-rate 0.0005

# Ver todas las opciones
python train_antispoofing.py --help
```

### Parámetros Disponibles

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--dataset-root` | `../Dataset` | Ruta raíz del dataset |
| `--batch-size` | `32` | Tamaño del batch (reducir si hay error de memoria) |
| `--epochs` | `50` | Número máximo de épocas |
| `--learning-rate` | `0.001` | Learning rate inicial |
| `--patience` | `15` | Épocas sin mejora para early stopping |
| `--output-dir` | `./models` | Directorio para guardar modelos |
| `--num-workers` | `4` | Workers para carga de datos |
| `--model-name` | `antispoofing_efficientnet_b2` | Nombre del modelo |

## 📊 Salida

El script genera:

1. **Modelos guardados** en `./models/`:
   - `antispoofing_efficientnet_b2_best.pth` - Mejor modelo (state dict)
   - `antispoofing_efficientnet_b2.pt` - Modelo TorchScript para producción
   - `antispoofing_efficientnet_b2_full.pth` - Modelo completo con historial

2. **Logs**:
   - `training.log` - Log completo del entrenamiento
   - Salida en consola con progreso en tiempo real

3. **Métricas finales**:
   - Accuracy, Precision, Recall, F1-Score
   - Matriz de confusión

## ⏱️ Tiempo Estimado

- **Con GPU (RTX 3060)**: 2-4 horas para 50 épocas
- **Con CPU**: 8-12 horas para 50 épocas

El entrenamiento puede detenerse antes si se activa early stopping.

## 🔍 Monitoreo

Durante el entrenamiento verás:

```
Epoch 1/50 (45.2s): Train Loss=0.5234, Train Acc=75.23%, Val Loss=0.4123, Val Acc=82.15%, LR=0.001000
   ✅ Nuevo mejor modelo guardado (Val Acc: 82.15%)
Epoch 2/50 (43.8s): Train Loss=0.4012, Train Acc=83.45%, Val Loss=0.3456, Val Acc=85.67%, LR=0.000987
   ✅ Nuevo mejor modelo guardado (Val Acc: 85.67%)
...
```

## 🎯 Objetivos de Rendimiento

El modelo debería alcanzar:
- **Accuracy**: > 85%
- **Precision (Fake)**: > 80%
- **Recall (Fake)**: > 85%
- **F1-Score**: > 82%

## 🐛 Solución de Problemas

### Error: "CUDA out of memory"
- Reduce el `--batch-size` (ej: `--batch-size 16` o `--batch-size 8`)
- Cierra otras aplicaciones que usen GPU

### Error: "No se encontró la carpeta Train/Validation/Test"
- Verifica que el dataset esté en `../Dataset` o usa `--dataset-root` para especificar la ruta

### Error: "ModuleNotFoundError: No module named 'torch'"
- Asegúrate de haber activado el entorno virtual
- Reinstala las dependencias: `pip install -r requirements.txt`

### El entrenamiento es muy lento
- Verifica que CUDA esté disponible: `python -c "import torch; print(torch.cuda.is_available())"`
- Si no hay GPU, considera usar Google Colab o AWS

## 📚 Documentación Adicional

Para más detalles sobre el proceso de entrenamiento, arquitectura del modelo y estrategias de optimización, consulta:

- `GUIA_ENTRENAMIENTO_ANTISPOOFING.md` - Guía completa y detallada

## 🔄 Integración con el Servicio

Una vez entrenado el modelo:

1. Copia `models/antispoofing_efficientnet_b2.pt` a `../models/` en la raíz del proyecto
2. El modelo se cargará automáticamente en `src/antispoofing_enhanced.py` (si está configurado)
3. El servicio usará el modelo ML junto con las heurísticas en un ensemble

## 📝 Notas

- El modelo usa **transfer learning** con EfficientNet-B2 pre-entrenado en ImageNet
- Se aplica **data augmentation** automáticamente durante el entrenamiento
- El modelo se exporta a **TorchScript** para máxima compatibilidad
- Se implementa **early stopping** para evitar overfitting
- Los pesos de las clases se balancean automáticamente si hay desbalance

---

**Última actualización**: 2025-11-27  
**Versión**: 1.0

