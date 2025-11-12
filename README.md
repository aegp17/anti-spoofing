# Anti-Spoofing Document Detector

Servicio de detección de documentos vs selfies utilizando FastAPI, heurísticas avanzadas y clasificación con redes neuronales (MobileNetV2).

## 🎯 Características

- **Detección dual**: Identifica si una imagen es un documento de identidad o un selfie
- **Análisis heurístico**: Detección rápida mediante análisis de formas, texto y rostros
- **Clasificación ML**: Fallback con MobileNetV2 para casos ambiguos
- **Validación de imagen**: Verificación de formato, tamaño y calidad
- **API REST**: Endpoints FastAPI para integración fácil
- **Batch processing**: Soporte para procesamiento de múltiples imágenes
- **Docker ready**: Dockerfile incluido para despliegue containerizado

## 📋 Requisitos

- Python 3.10+
- Tesseract OCR (para detección de texto)
- CUDA compatible (opcional, para aceleración GPU)

## 🚀 Instalación

### Local

```bash
# Clonar repositorio
cd anti-spoofing

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# En macOS/Linux, instalar Tesseract
# macOS:
brew install tesseract

# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
```

### Docker

```bash
# Construir imagen
docker build -t anti-spoofing-detector .

# Ejecutar contenedor
docker run -p 8000:8000 anti-spoofing-detector

# Alternativamente, usar docker-compose
docker-compose up -d
```

## 📖 Uso

### Iniciar servidor

```bash
python main.py
```

El servidor estará disponible en `http://localhost:8000`

### Documentación API

Acceder a `http://localhost:8000/docs` para ver la documentación interactiva (Swagger UI)

### Endpoints

#### 1. **Health Check**

```bash
GET /health
```

Respuesta:
```json
{
  "status": "healthy",
  "service": "Anti-Spoofing Document Detector",
  "version": "1.0.0"
}
```

#### 2. **Detectar imagen individual**

```bash
POST /detect
```

Parámetros:
- `file`: Archivo de imagen (JPEG, PNG)

Ejemplo con curl:
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@/path/to/image.jpg"
```

Respuesta exitosa:
```json
{
  "response": "id document detect",
  "method": "heuristic_rule_1"
}
```

o

```json
{
  "response": "is selfie",
  "confidence": 0.92,
  "method": "ml_model"
}
```

#### 3. **Procesamiento batch**

```bash
POST /detect/batch
```

Ejemplo:
```bash
curl -X POST http://localhost:8000/detect/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.png"
```

Respuesta:
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "response": "id document detect",
      "method": "heuristic_rule_1"
    },
    {
      "filename": "image2.jpg",
      "response": "is selfie",
      "confidence": 0.88,
      "method": "ml_model"
    }
  ]
}
```

## 🧪 Testing Local

### Test individual

```bash
python test_detector.py /path/to/image.jpg
```

### Test batch

```bash
python test_detector.py --batch /path/to/images/directory
```

## 🏗️ Arquitectura

```
anti-spoofing/
├── main.py                 # Aplicación FastAPI
├── src/
│   ├── __init__.py
│   ├── image_processor.py  # Validación y preprocesamiento
│   ├── heuristic_detector.py  # Análisis heurístico
│   ├── ml_classifier.py    # Clasificador CNN
│   └── detector.py         # Orquestador principal
├── models/                 # (Crear: guardar modelos ML aquí)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── test_detector.py
└── README.md
```

## 🧠 Lógica de Decisión

La detección utiliza decisiones jerárquicas:

### 1. Reglas Heurísticas (rápido)

- **Documento**: Detecta forma rectangular + texto + sin rostro prominente
- **Selfie**: Detecta rostro prominente + sin texto
- **Documento**: Detecta aspecto rectangular + texto

### 2. Fallback ML

Si las heurísticas no son concluyentes y el modelo está disponible:
- Score ≥ 0.85 → "id document detect"
- Score < 0.85 → "is selfie"

### 3. Default

Si no hay modelo ML disponible, usa presencia de rostro como criterio final.

## 🤖 Entrenamiento del Modelo ML

Para entrenar MobileNetV2 con tus propios datos:

```bash
python notebooks/train_mobilenet.py \
  --train-dir data/train \
  --val-dir data/val \
  --epochs 30 \
  --output models/model_mobilenet_v2.pt
```

Dataset esperado:
```
data/
├── train/
│   ├── documents/  # Imágenes de documentos
│   └── selfies/    # Imágenes de rostros
└── val/
    ├── documents/
    └── selfies/
```

## 📊 Métricas y Monitoreo

El servicio incluye información de método en cada respuesta:

- `heuristic_rule_1`: Forma rectangular + texto
- `heuristic_rule_2`: Rostro prominente
- `heuristic_rule_3`: Aspecto rectangular + texto
- `ml_model`: Clasificación por red neuronal
- `default_face`: Fallback por presencia de rostro
- `default_document`: Fallback por defecto

## 🔧 Configuración

Variables de entorno (opcional):

```bash
# En archivo .env o al ejecutar
export TESSERACT_PATH=/usr/bin/tesseract  # Si está en ubicación no estándar
```

## ⚠️ Limitaciones

- Tesseract OCR puede tener limitaciones con texto muy pequeño o rotado
- Haar Cascade tiene mejor rendimiento con rostros frontales
- El modelo ML requiere entrenamiento con dataset representativo
- Imágenes de baja calidad pueden afectar la precisión

## 📝 Respuestas de Error

| Código | Mensaje | Causa |
|--------|---------|-------|
| 400 | Empty file uploaded | Archivo vacío |
| 400 | Image exceeds maximum size | Imagen > 10MB |
| 400 | Unsupported image format | Formato no es JPEG/PNG |
| 500 | Internal server error | Error en procesamiento |

## 🚢 Despliegue en Producción

### Opción 1: Docker + Nginx

```bash
# Construir imagen
docker build -t anti-spoofing:latest .

# Ejecutar con límites de recursos
docker run -d \
  --name anti-spoofing \
  -p 8000:8000 \
  -m 4g \
  --cpus="2" \
  -v $(pwd)/models:/app/models \
  anti-spoofing:latest
```

### Opción 2: Kubernetes

```bash
# Ver deployment.yaml (crear en raíz del proyecto)
kubectl apply -f deployment.yaml
```

### Opción 3: Systemd (Linux)

```bash
# Crear servicio systemd
sudo cp anti-spoofing.service /etc/systemd/system/
sudo systemctl enable anti-spoofing
sudo systemctl start anti-spoofing
```

## 📈 Performance

**Heurísticas**: ~100-200ms por imagen  
**Con ML**: ~500-800ms por imagen  
**Batch (10 imágenes)**: ~2-5s  

*Tiempos aproximados en CPU; GPU reduce significativamente*

## 📄 Licencia

Ver archivo LICENSE

## 👨‍💻 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Soporte

Para problemas o preguntas, abre un issue en el repositorio.

