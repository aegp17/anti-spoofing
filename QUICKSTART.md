# 🚀 Quick Start - Anti-Spoofing Service

## 5 Minutos para Empezar

### 1. Iniciar el Servicio

```bash
cd /Users/aegp17/Dropbox/Mac/Documents/code/fs-code/anti-spoofing

# Iniciar contenedor (primera vez tardará más)
docker-compose up

# En otra terminal, ver logs
docker logs -f anti-spoofing-test
```

**Esperar a ver:**
```
✓ Service initialized successfully
Uvicorn running on http://0.0.0.0:8000
```

---

## 2. Probar Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Anti-Spoofing Document Detector",
  "version": "1.0.0"
}
```

---

## 3. Detección de Documentos vs Selfies 📄

### Probar con una imagen

```bash
# Usar cualquier imagen en tu equipo
curl -X POST http://localhost:8000/detect \
  -F "file=@/path/to/your/image.jpg"
```

**Response Example:**
```json
{
  "response": "id document detect",
  "method": "heuristic_rule_1_text_detected"
}
```

---

## 4. Análisis de Deepfakes 🎭 ✨

### Probar detección en imagen

```bash
curl -X POST http://localhost:8000/analyze/deepfake/image \
  -F "file=@/path/to/your/image.jpg"
```

**Response Example:**
```json
{
  "response": "likely_real",
  "confidence": 0.4255,
  "method": "heuristic",
  "heuristics": {
    "sharpness": 245.5,
    "skin_variance": 1250.0
  }
}
```

### Probar detección en video

```bash
curl -X POST "http://localhost:8000/analyze/deepfake/video?frame_step=15&max_frames=30" \
  -F "file=@/path/to/your/video.mp4"
```

**Response Example:**
```json
{
  "response": "likely_real",
  "confidence_mean": 0.2234,
  "confidence_max": 0.3145,
  "confidence_median": 0.2123,
  "frames_analyzed": 30,
  "frames_with_faces": 28,
  "method": "ml_video_aggregation"
}
```

---

## 5. Test Script Automatizado

```bash
# Sin archivos (solo health checks)
./scripts/test_deepfake.sh

# Con archivo de imagen
./scripts/test_deepfake.sh test.jpg

# Con imagen y video
./scripts/test_deepfake.sh test.jpg video.mp4
```

---

## Comandos Útiles

### Ver logs en vivo

```bash
docker logs -f anti-spoofing-test
```

### Detener servicio

```bash
docker-compose down
```

### Rebuild completo

```bash
docker-compose down
docker-compose up --build
```

### Limpiar espacios de Docker

```bash
docker system prune -a
```

---

## Resumen de Endpoints

| Endpoint | Método | Qué Hace |
|----------|--------|----------|
| `/health` | GET | Verificar servicio |
| `/detect` | POST | Documento vs Selfie |
| `/detect/batch` | POST | Múltiples imágenes |
| `/analyze/deepfake/image` | POST | Deepfake en imagen |
| `/analyze/deepfake/video` | POST | Deepfake en video |

---

## Integración Futura: Modelo ML

Cuando tengas modelo pre-entrenado:

1. **Obtener modelo** (formato PyTorch `.pt` o `.pth`)

2. **Colocar en:**
   ```bash
   cp deepfake_detector.pt models/deepfake_detector.pt
   ```

3. **Reiniciar:**
   ```bash
   docker-compose up --build
   ```

4. **Verificar en logs:**
   ```
   ✓ Deepfake model loaded from models/deepfake_detector.pt
   ```

5. **Mejora esperada:**
   - Precisión: 60-70% → 95-99%
   - Method: "heuristic" → "ml_model"

---

## Troubleshooting

### Puerto 8000 ya en uso

```bash
# Liberar puerto
lsof -i :8000
kill -9 <PID>

# O cambiar puerto en docker-compose.yml
```

### Contenedor no inicia

```bash
# Ver logs del build
docker-compose logs

# Rebuild completo
docker-compose down --volumes
docker-compose up --build
```

### Memory leak en procesamiento de video

```bash
# Aumentar limite en docker-compose.yml
# O procesar videos en chunks más pequeños
```

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────┐
│   Anti-Spoofing Service             │
│   FastAPI on Uvicorn                │
│   http://localhost:8000             │
└────────────┬────────────────────────┘
             │
      ┌──────┴──────────┐
      │                 │
      ▼                 ▼
  /detect          /analyze/deepfake/*
  (documento vs    (análisis de
   selfie)         manipulaciones)
```

---

## Documentación Completa

Para información detallada, ver:

- **README.md** - Documentación general
- **MODULES_OVERVIEW.md** - Referencia de módulos
- **DEEPFAKE_ARCHITECTURE.md** - Especificación técnica
- **DEEPFAKE_INTEGRATION_GUIDE.md** - Guía de integración

---

## Ejemplos Prácticos

### Caso 1: Verificar si es selfie real

```bash
# Paso 1: Es selfie?
curl -X POST http://localhost:8000/detect \
  -F "file=@selfie.jpg"

# Response: {"response": "is selfie", ...}

# Paso 2: Es real o deepfake?
curl -X POST http://localhost:8000/analyze/deepfake/image \
  -F "file=@selfie.jpg"

# Response: {"response": "likely_real", ...}

# ✅ Selfie auténtico
```

### Caso 2: Verificar documento

```bash
# Paso 1: Es documento?
curl -X POST http://localhost:8000/detect \
  -F "file=@document.jpg"

# Response: {"response": "id document detect", ...}

# Paso 2: Es documento real o falso?
curl -X POST http://localhost:8000/analyze/deepfake/image \
  -F "file=@document.jpg"

# Response: {"response": "likely_real", ...}

# ✅ Documento válido
```

### Caso 3: Verificar video (presentación)

```bash
curl -X POST "http://localhost:8000/analyze/deepfake/video?frame_step=10&max_frames=50" \
  -F "file=@presentation.mp4"

# Response: {"response": "likely_real", "confidence_mean": 0.23, ...}

# ✅ Video auténtico
```

---

## Performance Esperado

### Heurística Pura (sin modelo ML)

| Tipo | Latencia | Precisión |
|------|----------|-----------|
| Imagen | 200-500ms | 60-70% |
| Video (50 frames) | 2-5s | 60-70% |

### Con Modelo ML (Fase 2)

| Tipo | Latencia | Precisión |
|------|----------|-----------|
| Imagen | 300-800ms | 95-99% |
| Video (50 frames) | 5-15s | 95-99% |

---

## Recursos

### Papers & Datasets

- **FaceForensics++**: Rößler et al. 2019
- **DFDC**: DeepFake Detection Challenge
- **Celeb-DF**: High-quality deepfakes

### Modelos Pre-entrenados

- **Hugging Face**: https://huggingface.co/models?search=deepfake
- **Papers with Code**: https://paperswithcode.com/task/fake-face-detection

### Librerías Relacionadas

```
mediapipe    # Face landmarks
dlib        # Advanced detection
scikit-image # Signal processing
scipy       # Frequency analysis
gradcam     # Explainability
```

---

## Soporte

### Logs para Debugging

```bash
# Ver últimas líneas de logs
docker logs --tail 50 anti-spoofing-test

# Guardar logs en archivo
docker logs anti-spoofing-test > service.log

# Logs en tiempo real (verbose)
docker logs -f --timestamps anti-spoofing-test
```

### Contacto

Ver documentación en:
- `DEEPFAKE_ARCHITECTURE.md` (Sección FAQ)
- `DEEPFAKE_INTEGRATION_GUIDE.md` (Sección Troubleshooting)

---

## Próximos Pasos

1. ✅ **Ahora**: Servicio funcionando con heurísticas
2. ⏳ **Fase 2**: Integrar modelo ML (precisión 95-99%)
3. 📋 **Fase 3**: Señales avanzadas (rPPG, micro-expressions)
4. 🎯 **Fase 4**: Producción (GPU, dashboard, reportes)

---

**¡Listo! 🚀**

El servicio está corriendo y listo para analizar imágenes y videos.

Comienza con `./scripts/test_deepfake.sh` para probar todo.

```
╔════════════════════════════════════════════════╗
║                                                ║
║  ✅ Anti-Spoofing Service Ready                ║
║  http://localhost:8000                         ║
║                                                ║
║  Endpoints disponibles:                         ║
║  • /detect (documento vs selfie)               ║
║  • /analyze/deepfake/image                     ║
║  • /analyze/deepfake/video                     ║
║                                                ║
╚════════════════════════════════════════════════╝
```

