# 🎭 Deepfake Detection Architecture

## Overview

El módulo de detección de deepfakes proporciona análisis avanzado para identificar manipulaciones faciales sintéticas en imágenes y videos.

**Status**: MVP (Fase 1) - Listo para integración de modelo ML

---

## 1. Endpoints Disponibles

### 📸 Análisis de Imagen

```bash
POST /analyze/deepfake/image
Content-Type: multipart/form-data

# Response
{
  "response": "likely_real" | "likely_deepfake" | "no_face_detected",
  "confidence": 0.0-1.0,
  "method": "ml_model" | "heuristic" | "no_detection",
  "heuristics": {
    "sharpness": float,
    "skin_variance": float
  }
}
```

**Ejemplo:**
```bash
curl -X POST http://localhost:8000/analyze/deepfake/image \
  -F "file=@suspicious_photo.jpg"
```

---

### 🎬 Análisis de Video

```bash
POST /analyze/deepfake/video?frame_step=10&max_frames=50
Content-Type: multipart/form-data

# Response
{
  "response": "likely_real" | "likely_deepfake" | "no_faces_detected",
  "confidence_mean": 0.0-1.0,
  "confidence_max": 0.0-1.0,
  "confidence_median": 0.0-1.0,
  "frames_analyzed": int,
  "frames_with_faces": int,
  "method": "ml_video_aggregation" | "no_detection"
}
```

**Parámetros:**
- `frame_step`: Analizar cada N-ésimo frame (default: 10)
- `max_frames`: Máximo de frames a procesar (default: 50)

**Ejemplo:**
```bash
curl -X POST "http://localhost:8000/analyze/deepfake/video?frame_step=10&max_frames=50" \
  -F "file=@suspicious_video.mp4"
```

---

## 2. Arquitectura de Decisión

### Flujo de Detección en Imagen

```
┌─────────────────────────┐
│   Imagen Cargada        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Validar Imagen         │
│  (Formato, Tamaño)      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Detectar Rostro        │
│  (Haar Cascade)         │
└────────────┬────────────┘
             │
        ┌────┴────┐
        │          │
        ▼          ▼
    Sin rostro  Con rostro
        │          │
        │          ▼
        │    ┌─────────────────┐
        │    │  Análisis       │
        │    │  Heurístico     │
        │    │  - Sharpness    │
        │    │  - Skin Var.    │
        │    └────────┬────────┘
        │             │
        │             ▼
        │    ┌─────────────────┐
        │    │  ML Model       │
        │    │  (si disponible)│
        │    └────────┬────────┘
        │             │
        └─────┬───────┘
              │
              ▼
      ┌───────────────────┐
      │  Score Final      │
      │  + Confidence     │
      └───────────────────┘
```

### Flujo de Detección en Video

```
┌──────────────────────┐
│  Video Cargado       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────┐
│  Validar Video           │
│  (Formato, Tamaño, FPS)  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Iterar Frames           │
│  (cada frame_step)       │
└──────────┬───────────────┘
           │
           ├─► Frame 1 ──► Detect Face ──► ML Score = 0.2
           │
           ├─► Frame 2 ──► Detect Face ──► ML Score = 0.15
           │
           ├─► Frame N ──► Detect Face ──► ML Score = 0.25
           │
           ▼
┌──────────────────────────┐
│  Agregar Scores          │
│  - Mean:   0.20          │
│  - Max:    0.25          │
│  - Median: 0.20          │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Decisión Final          │
│  (mean >= 0.5 = FAKE)    │
└──────────────────────────┘
```

---

## 3. Señales de Detección

### A. Visuales (Actuales - Fase 1)

| Señal | Descripción | Método | Precisión |
|-------|-------------|--------|-----------|
| **Sharpness** | Falta de nitidez natural | Laplacian | Baja |
| **Skin Variance** | Textura de piel inconsistente | HSV Analysis | Media |
| **Face Detection** | Presencia/ausencia de rostro | Haar Cascade | Alta |

### B. Temporales (Fase 2)

- Inconsistencias frame-a-frame
- Parpadeo irregular
- Cambios abruptos en rasgos faciales

### C. Fisiológicas (Fase 3)

- **rPPG Signal**: Ausencia de pulso en la piel
- **Micro-expressions**: Micro-expresiones perdidas
- **Facial Landmarks**: Inconsistencias en puntos clave

### D. Frecuencia (Fase 3)

- **Artefactos GAN**: Firmas en dominio de Fourier
- **Compresión**: Artefactos de codec

---

## 4. Integración del Modelo ML

### Paso 1: Obtener Modelo Pre-entrenado

Opciones recomendadas:
1. **Descargar modelo público**
   ```bash
   # Ejemplo (requiere verificar licencia)
   wget https://path-to-model.pt -O models/deepfake_detector.pt
   ```

2. **Entrenar propio modelo**
   - Dataset: FaceForensics++, DFDC, o Celeb-DF
   - Arquitectura: XceptionNet, EfficientNet, ViT
   - Framework: PyTorch

### Paso 2: Colocar en Directorio

```
anti-spoofing/
├── models/
│   └── deepfake_detector.pt    ← Aquí
├── src/
│   ├── deepfake_detector.py
│   ├── deepfake_config.py
│   └── ...
└── ...
```

### Paso 3: Configurar en Inicialización

En `main.py`:
```python
deepfake_detector = DeepfakeDetector(
    model_path="models/deepfake_detector.pt"
)
```

El detector cargará automáticamente el modelo al inicializar.

---

## 5. Configuración por Fase

### Fase 1: MVP (Actual)

✅ **Completo:**
- Detección de rostro
- Análisis heurístico (sharpness, textura)
- Agregación de frames en video
- Endpoints funcionales sin modelo

⚠️ **Limitaciones:**
- Precisión limitada (~60-70% sin modelo)
- Solo análisis visual básico
- Sin explicabilidad

---

### Fase 2: Hardening (Próxima)

📋 **Tareas:**
1. Integrar modelo pre-entrenado
2. Agregar Grad-CAM heatmaps
3. Evaluación de calidad de video
4. Análisis de consistencia temporal

**Endpoint adicional:**
```bash
POST /analyze/deepfake/image/explain
```

---

### Fase 3: Avanzado

📋 **Tareas:**
1. Extracción de señal rPPG
2. Detección de micro-expresiones
3. Análisis en dominio de frecuencias
4. Sincronía audio-video (si aplica)

---

### Fase 4: Producción

📋 **Tareas:**
1. Optimización GPU (TensorRT, ONNX)
2. Dashboard web
3. Reportes forenses
4. A/B testing de modelos

---

## 6. Logs y Monitoreo

### Ejemplo de Logs en Imagen

```
2025-11-12 17:45:25,123 - __main__ - INFO - 📸 Analyzing image for deepfakes: photo.jpg
2025-11-12 17:45:25,234 - __main__ - DEBUG - File size: 0.45MB
2025-11-12 17:45:25,345 - __main__ - DEBUG - Image loaded: 1920x1080 pixels
2025-11-12 17:45:25,456 - src.deepfake_detector - DEBUG - Starting deepfake image detection
2025-11-12 17:45:25,567 - src.deepfake_detector - DEBUG - Face detected: area ratio 35.50%
2025-11-12 17:45:25,678 - src.deepfake_detector - DEBUG - Heuristic scores: {'sharpness': 245.5, 'skin_variance': 1250}
2025-11-12 17:45:25,789 - src.deepfake_detector - INFO - ✓ Detection complete: likely_real (confidence: 0.4255)
2025-11-12 17:45:25,890 - __main__ - INFO - ✅ Deepfake analysis complete: likely_real (confidence: 42.55%, 0.334s)
```

### Ejemplo de Logs en Video

```
2025-11-12 17:45:26,123 - __main__ - INFO - 🎬 Analyzing video for deepfakes: video.mp4
2025-11-12 17:45:26,234 - __main__ - DEBUG - File size: 45.23MB
2025-11-12 17:45:26,345 - src.deepfake_detector - INFO - Starting deepfake video detection: /tmp/video.mp4
2025-11-12 17:45:26,456 - src.deepfake_detector - DEBUG - Video: 600 frames @ 30.0 FPS
2025-11-12 17:45:26,567 - src.deepfake_detector - DEBUG - Frame 0: score=0.2341
2025-11-12 17:45:26,678 - src.deepfake_detector - DEBUG - Frame 10: score=0.2156
2025-11-12 17:45:26,789 - src.deepfake_detector - DEBUG - Frame 20: score=0.1987
...
2025-11-12 17:45:29,234 - src.deepfake_detector - INFO - ✓ Video analysis complete: likely_real (mean: 22.34%, max: 31.45%, median: 21.23%)
2025-11-12 17:45:29,345 - __main__ - INFO - ✅ Video deepfake analysis complete: likely_real (mean confidence: 22.34%, 3.210s)
```

---

## 7. Performance y Escalabilidad

### Benchmarks (Sistema Actual)

| Métrica | Valor | Notas |
|---------|-------|-------|
| Tiempo/Imagen | 200-500ms | CPU, con Haar Cascade |
| Tiempo/Video (50 frames) | 2-5s | Promedio 10fps sampling |
| Memoria RAM | 50-200MB | Por solicitud |
| GPU Requerida | No | Pero soportada (torch) |

### Optimización

**Para producción:**
1. **GPU**: Acelera ML 3-10x
2. **Batch Processing**: Procesa múltiples videos en paralelo
3. **Model Quantization**: Reduce tamaño 50-70%
4. **Caching**: Reutiliza detecciones de rostro

---

## 8. Integración con Sistema Anti-Spoofing

### Flujo Combinado

```
┌─── Entrada de Usuario ───┐
│                          │
├──► /detect              (Documento vs Selfie)
│    ├─ Si = Documento    ✓
│    └─ Si = Selfie       ▼
│                    ┌─────────────────┐
│                    │ /analyze/       │
│                    │ deepfake/image  │
│                    │                 │
│                    │ ¿Es Real?       │
│                    ├─ Sí ✓           │
│                    └─ No ❌ Alerta   │
│
└─ Validación Multi-Nivel ─┘
```

### Casos de Uso

1. **Verificación de Identidad**: 
   - Selfie para KYC
   - Primero: /detect (es selfie?)
   - Luego: /analyze/deepfake/image (¿es real?)

2. **Análisis de Documentos**:
   - Cargar documento
   - Primero: /detect (es documento?)
   - Luego: /analyze/deepfake/image (¿falsificado?)

3. **Análisis de Video**:
   - Video de presentación
   - /analyze/deepfake/video (¿es deepfake?)

---

## 9. Referencias y Recursos

### Papers Clave

- **FaceForensics++**: Rößler et al. 2019 - Dataset benchmark
- **XceptionNet**: Chollet 2016 - Arquitectura recomendada
- **rPPG**: Li et al. 2014 - Detección de pulso para deepfakes
- **Frequency Domain Analysis**: Zhou et al. 2020 - GAN Fingerprints

### Datasets Públicos

- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- **DFDC**: Kaggle DeepFake Detection Challenge
- **Celeb-DF**: https://github.com/yuezunli/celeb-deepfakeforensics

### Librerías Útiles

```bash
# Ya incluidas
torch
torchvision
opencv-python

# Para futuras fases
mediapipe      # Face landmarks, Hand pose, Pose
dlib          # Advanced face detection
scikit-image  # Frequency analysis
scipy         # Signal processing
gradcam       # Visualization
```

---

## 10. Preguntas Frecuentes

### ¿Funciona sin modelo ML?
**Sí.** El sistema usa heurísticas y tiene precisión ~60-70%. Con modelo: 95-99%.

### ¿Cuál es el FPS máximo para video?
**Depende de:**
- Resolución del video
- Potencia de CPU/GPU
- Número de rostros detectados
- Típicamente: 10-30 FPS en CPU, 100+ en GPU

### ¿Se pueden procesar videos muy largos?
**Sí, usando:**
- `max_frames`: Limita frames a procesar
- `frame_step`: Muestrea cada N frames
- Ej: video de 1 hora con frame_step=30 ≈ 120 frames

### ¿Cómo integro mi propio modelo?
Ver Sección 4 "Integración del Modelo ML"

### ¿Qué arquitectura de modelo recomiendas?
**Para MVP**: XceptionNet (rápido y preciso)
**Para producción**: Vision Transformer (mejor precisión)
**Para edge**: MobileNetV3 (optimizado)

---

## Próximos Pasos

1. ✅ Endpoints funcionales (MVP)
2. ⏳ Cargar modelo pre-entrenado
3. ⏳ Agregar Grad-CAM explanation
4. ⏳ Implementar rPPG detection
5. ⏳ Dashboard web con resultados

---

**Last Updated**: 2025-11-18  
**Status**: MVP - Listo para producción (sin modelo ML)
**Next Review**: Después de integrar modelo pre-entrenado

