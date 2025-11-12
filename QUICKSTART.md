# ⚡ Quick Start - Anti-Spoofing Detector

Guía rápida para empezar a usar el servicio de detección de documentos vs selfies en 5 minutos.

## 🚀 Inicio Rápido (Opción A: Local)

### 1. Instalar dependencias

```bash
# Requisito previo: Python 3.10+
python --version

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias Python
pip install -r requirements.txt

# Instalar Tesseract (requerido para OCR)
# macOS:
brew install tesseract

# Linux (Ubuntu/Debian):
sudo apt-get install tesseract-ocr

# Windows: Descargar desde https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Iniciar servidor

```bash
python main.py
```

✅ Servidor ejecutándose en `http://localhost:8000`

### 3. Probar API

#### En otra terminal:

```bash
# Probar con health check
curl http://localhost:8000/health

# Detectar documento
curl -X POST http://localhost:8000/detect \
  -F "file=@/path/to/your/document.jpg"

# Ver documentación interactiva
open http://localhost:8000/docs
```

## 🐳 Inicio Rápido (Opción B: Docker)

### 1. Construir imagen

```bash
docker build -t anti-spoofing .
```

### 2. Ejecutar contenedor

```bash
docker run -p 8000:8000 anti-spoofing
```

✅ Servidor ejecutándose en `http://localhost:8000`

### 3. Con Docker Compose

```bash
docker-compose up -d
```

Verificar estado:
```bash
docker-compose ps
docker logs anti-spoofing-detector
```

Detener:
```bash
docker-compose down
```

## 📝 Casos de Uso

### Caso 1: Detectar documento

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@cedula.jpg"
```

**Respuesta:**
```json
{
  "response": "id document detect",
  "method": "heuristic_rule_1"
}
```

### Caso 2: Detectar selfie

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@selfie.jpg"
```

**Respuesta:**
```json
{
  "response": "is selfie",
  "confidence": 0.92,
  "method": "ml_model"
}
```

### Caso 3: Procesamiento batch

```bash
curl -X POST http://localhost:8000/detect/batch \
  -F "files=@document1.jpg" \
  -F "files=@selfie1.jpg" \
  -F "files=@document2.png"
```

**Respuesta:**
```json
{
  "results": [
    {
      "filename": "document1.jpg",
      "response": "id document detect",
      "method": "heuristic_rule_1"
    },
    {
      "filename": "selfie1.jpg",
      "response": "is selfie",
      "confidence": 0.88,
      "method": "ml_model"
    },
    {
      "filename": "document2.png",
      "response": "id document detect",
      "method": "heuristic_rule_3"
    }
  ]
}
```

## 🧪 Testing Local

Sin necesidad de servidor FastAPI, prueba directamente el detector:

```bash
# Imagen individual
python test_detector.py /path/to/image.jpg

# Batch processing
python test_detector.py --batch /path/to/images/
```

Ejemplo de salida:
```
📸 document.jpg         → {'response': 'id document detect', 'method': 'heuristic_rule_1'}
📸 selfie.jpg           → {'response': 'is selfie', 'confidence': 0.92, 'method': 'ml_model'}
```

## 🤖 Usar con modelo ML (Opcional)

Si tienes un modelo entrenado:

1. **Copiar modelo:**
```bash
cp /path/to/model_mobilenet_v2.pt models/
```

2. **Reiniciar servidor** - El modelo se cargará automáticamente

3. **Verificar en respuesta:**
```json
{
  "response": "is selfie",
  "confidence": 0.92,
  "method": "ml_model"  ← Confirma que está usando ML
}
```

## 🎓 Entrenar Modelo Personalizado

Si quieres entrenar con tus propias imágenes:

### Preparar dataset

```bash
# Estructura necesaria:
data/
├── train/
│   ├── documents/  (500+ imágenes)
│   └── selfies/    (500+ imágenes)
└── val/
    ├── documents/  (100+ imágenes)
    └── selfies/    (100+ imágenes)
```

### Entrenar

```bash
python notebooks/train_mobilenet.py \
  --train-dir data/train \
  --val-dir data/val \
  --epochs 30 \
  --output models/model_mobilenet_v2.pt
```

Verás progreso:
```
📊 MobileNetV2 Document vs Selfie Classifier
Epoch    Train Loss   Val Loss     Val Acc
1        0.6832       0.5421       0.7340
2        0.4521       0.3812       0.8156
3        0.3245       0.2891       0.8623
...
✅ Training complete!
```

## 🔍 Monitoreo en Tiempo Real

### Swagger UI (Recomendado)

```
http://localhost:8000/docs
```

- Interfaz gráfica para probar endpoints
- Documentación interactiva
- Esquemas de request/response

### ReDoc

```
http://localhost:8000/redoc
```

Documentación alternativa más limpia

## 🐛 Troubleshooting

### Error: "Tesseract is not installed"

**Solución:**
```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr

# Luego reinicia el servidor
```

### Error: "CUDA not available"

**Solución:** El servicio funcionará con CPU automáticamente. Para GPU:

```bash
# Instalar CUDA (versión compatible con tu GPU)
# Luego instalar torch con CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Error: "Model not found"

**Solución:** Esto es normal. El servicio funcionará con heurísticas únicamente. Para usar ML:

```bash
# Entrenar modelo o descargar preentrenado
python notebooks/train_mobilenet.py --train-dir data/train --val-dir data/val
```

### Error 400: "Image exceeds maximum size"

**Solución:** Comprimir imagen (máximo 10MB):

```bash
# Reducir tamaño
convert input.jpg -resize 2000x2000 output.jpg

# O ajustar MAX_FILE_SIZE en config.py
```

## 📊 Interpretar Respuestas

### Métodos de detección

| Método | Significado | Confianza |
|--------|------------|-----------|
| `heuristic_rule_1` | Forma rectangular + texto detectados | Alto |
| `heuristic_rule_2` | Rostro prominente detectado | Alto |
| `heuristic_rule_3` | Aspecto rectangular + texto | Alto |
| `ml_model` | Clasificación por red neuronal | Medio-Alto |
| `default_face` | Fallback por presencia de rostro | Bajo |
| `default_document` | Fallback por defecto | Bajo |

### Campos de respuesta

```json
{
  "response": "id document detect",     // ← Clasificación principal
  "method": "heuristic_rule_1",         // ← Método usado (opcional)
  "confidence": 0.92                    // ← Score ML (solo si ML) 0-1
}
```

## 🚨 Casos Edge (Manejados)

| Caso | Resultado | Razón |
|------|-----------|-------|
| Documento borroso | Puede fallar heurística, ML requiere entrenamiento | Mejorar con ML o imagen más clara |
| Documento rotado | OCR puede fallar | Tesseract intenta autodetectar |
| Selfie con documento atrás | Basado en qué sea prominente | ML diferencia mejor estos casos |
| Imagen en blanco | Falsa positiva como selfie | Agregar validación adicional si es crítico |

## 📞 Próximos Pasos

1. **Integrar en tu aplicación:**
   ```python
   import requests
   
   response = requests.post(
       'http://localhost:8000/detect',
       files={'file': open('image.jpg', 'rb')}
   )
   result = response.json()
   print(result['response'])
   ```

2. **Desplegar en producción:**
   - Ver `ARCHITECTURE.md` para Kubernetes
   - Configurar límites de rate
   - Agregar autenticación si es necesario

3. **Mejorar modelo:**
   - Recolectar feedback de usuarios
   - Reentrenar periódicamente
   - Experimentar con otros modelos

4. **Monitorear:**
   - Agregar logging detallado
   - Trackear métricas de precisión
   - Alertas para fallos

## 📚 Recursos

- 📖 [Documentación completa](README.md)
- 🏗️ [Arquitectura del sistema](ARCHITECTURE.md)
- 🧪 [Entrenar modelo](notebooks/train_mobilenet.py)
- 🐳 [Docker Compose](docker-compose.yml)
- 💻 [Ejemplos de API](examples.sh)

---

**¿Necesitas ayuda?** Revisa los logs:

```bash
# Local
tail -f /tmp/detector.log

# Docker
docker logs anti-spoofing-detector -f
```

