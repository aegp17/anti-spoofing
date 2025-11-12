# 🔌 Guía de Integración - Anti-Spoofing Detector

## Integración en tu aplicación

### Python (Requests)

```python
import requests
import json

def verify_document(image_path: str) -> dict:
    """
    Enviar imagen a servicio de detección.
    
    Args:
        image_path: Ruta a la imagen
        
    Returns:
        Dict con resultado de detección
    """
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            'http://localhost:8000/detect',
            files=files,
            timeout=10
        )
    
    return response.json()

# Uso
result = verify_document('cedula.jpg')
print(result)
# {'response': 'id document detect', 'method': 'heuristic_rule_1_text_detected'}

if result['response'] == 'id document detect':
    print("✅ Documento válido")
else:
    print("❌ No es un documento válido")
```

### JavaScript/Node.js (Fetch API)

```javascript
async function verifyDocument(imagePath) {
    const formData = new FormData();
    
    // Obtener archivo
    const fileInput = document.getElementById('documentInput');
    formData.append('file', fileInput.files[0]);
    
    try {
        const response = await fetch('http://localhost:8000/detect', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}

// Uso en HTML
document.getElementById('uploadBtn').addEventListener('click', async () => {
    const result = await verifyDocument();
    
    if (result.response === 'id document detect') {
        console.log('✅ Documento detectado');
    } else {
        console.log('❌ Selfie detectada');
    }
});
```

### cURL (desde línea de comandos)

```bash
# Detectar documento único
curl -X POST http://localhost:8000/detect \
  -F "file=@documento.jpg"

# Respuesta
# {"response":"id document detect","method":"heuristic_rule_1_text_detected"}

# Procesar múltiples archivos
curl -X POST http://localhost:8000/detect/batch \
  -F "files=@doc1.jpg" \
  -F "files=@doc2.jpg" \
  -F "files=@selfie1.png"

# Respuesta
# {
#   "results": [
#     {"filename": "doc1.jpg", "response": "id document detect", ...},
#     {"filename": "doc2.jpg", "response": "id document detect", ...},
#     {"filename": "selfie1.png", "response": "is selfie", ...}
#   ]
# }
```

### Python (Async con aiohttp)

```python
import aiohttp
import asyncio

async def verify_document_async(image_path: str):
    """Detección asíncrona de documentos."""
    async with aiohttp.ClientSession() as session:
        with open(image_path, 'rb') as f:
            form = aiohttp.FormData()
            form.add_field('file', f, filename=image_path)
            
            async with session.post(
                'http://localhost:8000/detect',
                data=form,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                return await resp.json()

# Uso
result = asyncio.run(verify_document_async('cedula.jpg'))
print(result)
```

## Manejo de errores

### Casos comunes

```python
import requests
from requests.exceptions import RequestException, Timeout

def safe_verify_document(image_path: str, retries: int = 3):
    """Verificación con reintentos y manejo de errores."""
    
    for attempt in range(retries):
        try:
            with open(image_path, 'rb') as f:
                response = requests.post(
                    'http://localhost:8000/detect',
                    files={'file': f},
                    timeout=10
                )
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 400:
                error = response.json()
                print(f"❌ Imagen inválida: {error['detail']}")
                return None
            
            elif response.status_code == 500:
                print(f"⚠️ Error interno (intento {attempt+1}/{retries})")
                continue
            
            else:
                print(f"⚠️ Error inesperado: {response.status_code}")
                return None
        
        except Timeout:
            print(f"⏱️ Timeout (intento {attempt+1}/{retries})")
            continue
        
        except RequestException as e:
            print(f"❌ Error de conexión: {e}")
            return None
    
    print("❌ No se pudo procesar después de los reintentos")
    return None

# Uso
result = safe_verify_document('cedula.jpg', retries=3)
if result:
    print(f"Resultado: {result['response']}")
```

## Validación de respuestas

```python
def is_valid_document(result: dict) -> tuple[bool, str]:
    """
    Validar si es un documento válido.
    
    Returns:
        Tupla (es_valido, razon)
    """
    if not result:
        return False, "No se obtuvo respuesta del servicio"
    
    if 'response' not in result:
        return False, "Respuesta mal formada"
    
    response_type = result['response']
    method = result.get('method', 'unknown')
    
    # Documento detectado
    if response_type == 'id document detect':
        confidence = result.get('confidence', 'N/A')
        return True, f"Documento detectado ({method}, conf: {confidence})"
    
    # Selfie detectado
    elif response_type == 'is selfie':
        confidence = result.get('confidence', 'N/A')
        return False, f"Selfie detectada ({method}, conf: {confidence})"
    
    # Respuesta desconocida
    else:
        return False, f"Tipo de respuesta desconocido: {response_type}"

# Uso
result = {'response': 'id document detect', 'method': 'heuristic_rule_1_text_detected'}
is_valid, reason = is_valid_document(result)

if is_valid:
    print(f"✅ {reason}")
else:
    print(f"❌ {reason}")
```

## Integración con base de datos

```python
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import requests

Base = declarative_base()

class DocumentVerification(Base):
    """Registro de verificaciones de documentos."""
    __tablename__ = "document_verifications"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    image_path = Column(String)
    detection_result = Column(String)  # 'id document detect' o 'is selfie'
    detection_method = Column(String)
    confidence = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

def verify_and_store(user_id: str, image_path: str, session):
    """Verificar documento y almacenar resultado."""
    
    # Enviar a detector
    try:
        with open(image_path, 'rb') as f:
            response = requests.post(
                'http://localhost:8000/detect',
                files={'file': f}
            )
        
        result = response.json()
        
        # Guardar en BD
        verification = DocumentVerification(
            user_id=user_id,
            image_path=image_path,
            detection_result=result.get('response'),
            detection_method=result.get('method'),
            confidence=result.get('confidence', 'N/A')
        )
        
        session.add(verification)
        session.commit()
        
        return result
    
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
        return None

# Uso
engine = create_engine('sqlite:///verifications.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

result = verify_and_store('user123', 'cedula.jpg', session)
print(result)
```

## Monitoreo y métricas

```python
from datetime import datetime
from collections import defaultdict
import time

class DetectionMetrics:
    """Rastrear métricas del servicio."""
    
    def __init__(self):
        self.total_requests = 0
        self.successful = 0
        self.failed = 0
        self.detection_counts = defaultdict(int)
        self.method_counts = defaultdict(int)
        self.latencies = []
    
    def record_request(self, result: dict, latency: float):
        """Registrar solicitud."""
        self.total_requests += 1
        self.latencies.append(latency)
        
        if result:
            self.successful += 1
            self.detection_counts[result['response']] += 1
            self.method_counts[result['method']] += 1
        else:
            self.failed += 1
    
    def get_stats(self) -> dict:
        """Obtener estadísticas."""
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        
        return {
            'total_requests': self.total_requests,
            'successful': self.successful,
            'failed': self.failed,
            'success_rate': self.successful / self.total_requests if self.total_requests > 0 else 0,
            'avg_latency_ms': avg_latency * 1000,
            'detection_distribution': dict(self.detection_counts),
            'method_distribution': dict(self.method_counts)
        }

# Uso
metrics = DetectionMetrics()

for _ in range(100):
    start = time.time()
    result = verify_document('cedula.jpg')
    latency = time.time() - start
    metrics.record_request(result, latency)

print(metrics.get_stats())
# {
#   'total_requests': 100,
#   'successful': 98,
#   'failed': 2,
#   'success_rate': 0.98,
#   'avg_latency_ms': 145.3,
#   'detection_distribution': {'id document detect': 52, 'is selfie': 46},
#   'method_distribution': {'heuristic_rule_1_text_detected': 52, ...}
# }
```

## Deployment en producción

### Docker Compose con persistencia

```yaml
version: '3.8'

services:
  anti-spoofing:
    build: .
    container_name: anti-spoofing-detector
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
  
  nginx:
    image: nginx:alpine
    container_name: anti-spoofing-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - anti-spoofing
    restart: unless-stopped
```

### Rate Limiting (nginx)

```nginx
# nginx.conf
upstream detector {
    server anti-spoofing:8000;
}

limit_req_zone $binary_remote_addr zone=detect_limit:10m rate=10r/s;

server {
    listen 80;
    
    location /detect {
        limit_req zone=detect_limit burst=20 nodelay;
        proxy_pass http://detector;
    }
}
```

## Pruebas de carga

```bash
# Usar Apache Bench
ab -n 1000 -c 10 -p cedula.jpg -T multipart/form-data http://localhost:8000/detect

# Usar wrk
wrk -t4 -c100 -d30s http://localhost:8000/health
```

## Troubleshooting

| Problema | Solución |
|----------|----------|
| Conexión rechazada | Verificar que Docker esté corriendo |
| Timeout | Aumentar timeout en cliente |
| Memoria agotada | Reducir tamaño de imagen o limitar concurrencia |
| OCR no detecta texto | Mejorar imagen, ajustar PSM, entrenar modelo |
| Falsos positivos | Recolectar más datos, reentrenar modelo |

---

**Documentación relacionada:**
- [README.md](README.md) - Documentación principal
- [QUICKSTART.md](QUICKSTART.md) - Inicio rápido
- [ARCHITECTURE.md](ARCHITECTURE.md) - Arquitectura del sistema

