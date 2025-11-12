# Arquitectura del Sistema Anti-Spoofing

## 📐 Visión General

El sistema de detección de documentos vs selfies implementa una arquitectura de **detección en cascada** que combina análisis heurísticos rápidos con clasificación ML como fallback.

```
┌─────────────────┐
│  Imagen Input   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  ImageProcessor             │
│  • Validar formato          │
│  • Validar tamaño           │
│  • Normalizar color (RGB)   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  HeuristicDetector          │
│  • Análisis de forma        │
│  • Detección de texto       │
│  • Detección de rostro      │
│  • Aspecto (aspect ratio)   │
└────────┬────────────────────┘
         │
    ┌────┴─────┐
    │           │
    ▼           ▼
 CONC?     Reglas OK?
 │           │
 │           ├─→ Documento/Selfie
 │           │
 ▼           ▼
┌─────────────────────────────┐
│  MLClassifier               │
│  (MobileNetV2 TorchScript)  │
│  • Binary classification    │
│  • Score ≥ 0.85 → Documento│
│  • Score < 0.85 → Selfie   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  JSON Response              │
│  { "response": "...",       │
│    "confidence": 0.92,      │
│    "method": "..." }        │
└─────────────────────────────┘
```

## 🏗️ Estructura de Módulos

### 1. **ImageProcessor** (`src/image_processor.py`)

Responsable de carga y validación inicial de imágenes.

**Responsabilidades:**
- Cargar bytes de imagen a PIL Image
- Validar formato (JPEG, PNG)
- Validar tamaño máximo (10MB)
- Convertir a RGB normalizado
- Conversión PIL ↔ OpenCV

**Métodos principales:**
```python
validate_and_load(image_bytes) → PIL.Image
pil_to_cv2(image_pil) → np.ndarray
get_image_dimensions(image_pil) → (width, height)
get_aspect_ratio(image_pil) → float
```

### 2. **HeuristicDetector** (`src/heuristic_detector.py`)

Implementa análisis rápidos basados en características visuales.

**Algoritmos:**
- **Edge Detection**: Canny edge detection para encontrar contornos rectangulares
- **Text Detection**: OCR con Tesseract para detectar presencia de texto
- **Face Detection**: Haar Cascade Classifier para detectar rostros humanos
- **Aspect Ratio**: Validar aspecto rectangular típico de documentos

**Flujo de decisión:**
```
SI (rectangular + texto + NO rostro) → DOCUMENTO
SI (rostro + NO texto) → SELFIE
SI (aspecto doc + texto) → DOCUMENTO
SINO → Pasar a ML
```

**Características configurables:**
```python
EDGE_THRESHOLD_LOW = 60
EDGE_THRESHOLD_HIGH = 180
MIN_CONTOUR_AREA = 8000
MIN_FACE_AREA_RATIO = 0.3
MIN_TEXT_LENGTH = 10
```

### 3. **MLClassifier** (`src/ml_classifier.py`)

Clasificador CNN basado en MobileNetV2 preentrenado.

**Arquitectura:**
```
Input (224×224 RGB)
    ↓
MobileNetV2 features (pretrained, frozen)
    ↓
Custom Head:
  - Linear(1280 → 256) + ReLU + Dropout
  - Linear(256 → 1)
    ↓
Sigmoid → [0, 1] probability
    ↓
≥ 0.85 → DOCUMENTO
< 0.85 → SELFIE
```

**Características:**
- Weights pretrained en ImageNet (congelados parcialmente)
- Fine-tuning de capas finales
- Umbral ajustable (default: 0.85)
- Soporte GPU/CPU automático
- Serialización TorchScript para producción

### 4. **DocumentDetector** (`src/detector.py`)

Orquestador principal que coordina el flujo de decisión.

**Responsabilidades:**
- Aplicar reglas heurísticas
- Fallback a ML si es necesario
- Retornar resultado estructurado con metadatos

**Métodos:**
```python
detect(image_pil) → {"response": "...", "confidence": float, "method": str}
_ml_classification(image_pil) → dict
```

### 5. **FastAPI Service** (`main.py`)

Servidor REST que expone la funcionalidad.

**Endpoints:**
- `GET /health` - Health check
- `POST /detect` - Detección individual
- `POST /detect/batch` - Procesamiento batch
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc documentation

**Manejo de errores:**
- 400: Archivo inválido, formato no soportado
- 400: Archivo vacío o mayor a 10MB
- 500: Error en procesamiento

## 🔄 Flujo de Procesamiento

### Caso 1: Documento Claro

```
Entrada: Imagen de cédula
    ↓
Preprocesamiento → RGB válida
    ↓
Heurística → rectangulo=SÍ, texto=SÍ, rostro=NO
    ↓
↳ RESULTADO: "id document detect" (method: heuristic_rule_1)
```

### Caso 2: Selfie Claro

```
Entrada: Foto facial
    ↓
Preprocesamiento → RGB válida
    ↓
Heurística → rectangulo=NO, texto=NO, rostro=SÍ
    ↓
↳ RESULTADO: "is selfie" (method: heuristic_rule_2)
```

### Caso 3: Ambiguo → ML

```
Entrada: Imagen con documento de fondo + rostro
    ↓
Preprocesamiento → RGB válida
    ↓
Heurística → rectangulo=SÍ, texto=?, rostro=SÍ (inconcluso)
    ↓
ML Classifier → Score = 0.72
    ↓
0.72 < 0.85
    ↓
↳ RESULTADO: "is selfie" (confidence: 0.28, method: ml_model)
```

## 📊 Rendimiento Esperado

| Métrica | Valor |
|---------|-------|
| Heurística latencia | 100-150ms |
| ML latencia | 400-600ms |
| Throughput (CPU) | ~3-5 img/seg |
| Throughput (GPU) | ~20-30 img/seg |
| Batch (10 img) CPU | ~1-2s |
| Batch (10 img) GPU | ~500-800ms |

## 🔧 Configuración del Entrenamiento

### Dataset Structure

```
data/
├── train/
│   ├── documents/
│   │   ├── cedula_001.jpg
│   │   ├── cedula_002.jpg
│   │   └── ... (500+ imágenes)
│   └── selfies/
│       ├── selfie_001.jpg
│       ├── selfie_002.jpg
│       └── ... (500+ imágenes)
└── val/
    ├── documents/ (100+ imágenes)
    └── selfies/ (100+ imágenes)
```

### Data Augmentation

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10°),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(ImageNet stats)
])
```

### Hiperparámetros

```python
epochs = 30
batch_size = 32
learning_rate = 1e-3
optimizer = Adam
loss = BCEWithLogitsLoss
```

## 🚀 Estrategia de Despliegue

### Local Development

```bash
python main.py
# http://localhost:8000
```

### Docker Container

```bash
docker build -t anti-spoofing .
docker run -p 8000:8000 anti-spoofing
```

### Kubernetes Production

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anti-spoofing-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: detector
  template:
    metadata:
      labels:
        app: detector
    spec:
      containers:
      - name: detector
        image: anti-spoofing:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## 📈 Monitoreo y Métricas

### Métricas por método de detección

```json
{
  "detection_method": "heuristic_rule_1",
  "confidence": null,
  "latency_ms": 120,
  "model_used": false
}
```

### Logs relevantes

```
[2024-01-15 10:30:45] POST /detect - 200 OK
[2024-01-15 10:30:46] Image: 1920x1080, Format: JPEG
[2024-01-15 10:30:46] Detection: document (heuristic_rule_1) in 125ms
```

## 🔐 Consideraciones de Seguridad

1. **Validación de entrada**: 
   - Formato de imagen permitido
   - Tamaño máximo de archivo
   - Verificación MIME type

2. **Manejo de error**:
   - No revelar stack traces en producción
   - Logging seguro de errores

3. **Rate limiting**:
   ```python
   # Usar middleware (no incluido, agregar si es necesario)
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/detect")
   @limiter.limit("100/minute")
   async def detect_image(file: UploadFile):
       ...
   ```

4. **GPU/Memory safety**:
   - Límite de tamaño de archivo
   - Timeout en procesamiento
   - Limpieza de memoria GPU

## 🛠️ Mantenimiento y Evolución

### Mejoras futuras

1. **Modelos alternativos**: EfficientNet, ResNet50
2. **Multimodal**: Integrar análisis de metadatos EXIF
3. **Caché**: Redis para imágenes frecuentes
4. **Analítica**: Tracking de confianza y errores
5. **A/B Testing**: Comparar diferentes modelos

### Reentrenamiento

```bash
# Cada 6 meses o cuando drift significativo
python notebooks/train_mobilenet.py \
  --train-dir data/train_v2 \
  --val-dir data/val_v2 \
  --epochs 50 \
  --output models/model_mobilenet_v2_v2.pt
```

## 📝 Referencias

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [OpenCV Haar Cascades](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [PyTorch TorchScript](https://pytorch.org/docs/stable/jit.html)

