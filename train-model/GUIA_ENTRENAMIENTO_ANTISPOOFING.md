# Guía de Entrenamiento: Modelo de IA para Anti-Spoofing

## 📋 Resumen Ejecutivo

Esta guía describe el proceso para entrenar un modelo de Machine Learning que distingue entre selfies reales y falsos (spoofed/fake), complementando las heurísticas actuales del sistema.

---

## 1. Preparación de Datos

### 1.1 Estructura del Dataset

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

### 1.2 Requisitos del Dataset

- **Mínimo recomendado**: 5,000 imágenes por clase (Real/Fake)
- **Ideal**: 10,000+ imágenes por clase
- **Balance**: 50/50 entre Real y Fake
- **Diversidad**: Diferentes condiciones de iluminación, ángulos, dispositivos, edades, géneros
- **Calidad**: Resolución mínima 224x224 píxeles
- **Formatos**: JPG, PNG (normalizar a RGB)

### 1.3 Técnicas de Data Augmentation

```python
# Transformaciones recomendadas
transforms = [
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.2, contrast=0.2),
    RandomAffine(degrees=0, translate=(0.1, 0.1)),
    GaussianBlur(kernel_size=3, p=0.1)
]
```

**No usar**: Flip vertical (cambia la orientación facial), rotaciones extremas (>30°)

---

## 2. Arquitectura del Modelo

### 2.1 Opciones Recomendadas

#### Opción A: EfficientNet-B2 (Recomendada)
- **Ventajas**: Balance entre precisión y velocidad
- **Parámetros**: ~9M
- **Tiempo inferencia**: ~50ms (CPU), ~10ms (GPU)
- **Precisión esperada**: 85-92%

#### Opción B: MobileNetV3-Large
- **Ventajas**: Muy rápido, ideal para producción
- **Parámetros**: ~5M
- **Tiempo inferencia**: ~30ms (CPU)
- **Precisión esperada**: 80-88%

#### Opción C: ResNet-50
- **Ventajas**: Alta precisión, bien documentado
- **Parámetros**: ~25M
- **Tiempo inferencia**: ~100ms (CPU)
- **Precisión esperada**: 88-94%

### 2.2 Arquitectura Final

```python
import torch
import torch.nn as nn
from torchvision import models

class AntiSpoofingModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # Backbone: EfficientNet-B2
        self.backbone = models.efficientnet_b2(pretrained=pretrained)
        
        # Reemplazar clasificador final
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
```

---

## 3. Proceso de Entrenamiento

### 3.1 Hiperparámetros

```python
config = {
    "batch_size": 32,           # Ajustar según GPU RAM
    "learning_rate": 0.001,      # Usar learning rate scheduler
    "epochs": 50,                # Early stopping en epoch 15 sin mejora
    "weight_decay": 0.0001,      # Regularización L2
    "optimizer": "AdamW",        # Alternativa: Adam
    "scheduler": "CosineAnnealingLR",  # Reducir LR gradualmente
    "loss_function": "CrossEntropyLoss",
    "class_weights": [1.0, 1.0]  # Ajustar si hay desbalance
}
```

### 3.2 Script de Entrenamiento (Esquema)

```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 1. Cargar datos
train_dataset = ImageFolder('Dataset/Train', transform=train_transform)
val_dataset = ImageFolder('Dataset/Validation', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 2. Inicializar modelo
model = AntiSpoofingModel(num_classes=2, pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 3. Optimizador y scheduler
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()

# 4. Loop de entrenamiento
best_val_acc = 0.0
patience = 15
no_improve = 0

for epoch in range(50):
    # Entrenamiento
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validación
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = val_correct / val_total
    scheduler.step()
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
          f"Val Acc={val_acc:.4%}, LR={scheduler.get_last_lr()[0]:.6f}")
```

---

## 4. Evaluación y Métricas

### 4.1 Métricas Clave

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# En conjunto de test
y_true = [...]  # Etiquetas reales
y_pred = [...]  # Predicciones del modelo

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label=1)  # Fake = 1
recall = recall_score(y_true, y_pred, pos_label=1)
f1 = f1_score(y_true, y_pred, pos_label=1)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
# [[True Negatives, False Positives],
#  [False Negatives, True Positives]]
```

### 4.2 Objetivos de Rendimiento

- **Accuracy**: > 85%
- **Precision (Fake)**: > 80% (evitar falsos positivos)
- **Recall (Fake)**: > 85% (detectar la mayoría de fakes)
- **F1-Score**: > 82%

### 4.3 Análisis de Errores

- **Falsos Positivos (Real clasificado como Fake)**: Revisar imágenes con baja calidad, iluminación pobre
- **Falsos Negativos (Fake clasificado como Real)**: Agregar más ejemplos similares al dataset

---

## 5. Optimización y Exportación

### 5.1 Quantization (Opcional)

```python
# Reducir tamaño del modelo para producción
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

### 5.2 Exportación a TorchScript

```python
# Para integración en el servicio actual
model.eval()
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('models/antispoofing_model.pt')
```

### 5.3 Integración en el Servicio

```python
# En src/ml_classifier.py o nuevo archivo
class AntiSpoofingMLModel:
    def __init__(self, model_path='models/antispoofing_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_pil):
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        return {
            "class": "real" if predicted.item() == 0 else "fake",
            "confidence": confidence.item()
        }
```

---

## 6. Estrategia de Integración con Heurísticas

### 6.1 Ensemble (Recomendado)

```python
def detect_with_ensemble(image_pil):
    # Heurísticas (rápido, siempre disponible)
    heuristic_result = enhanced_detector.detect(image_pil)
    heuristic_score = heuristic_result['confidence']
    
    # ML Model (más preciso, requiere modelo entrenado)
    ml_result = ml_model.predict(image_pil)
    ml_score = ml_result['confidence']
    
    # Combinación ponderada
    final_score = 0.3 * heuristic_score + 0.7 * ml_score
    
    if final_score < 0.5:
        return "real"
    else:
        return "fake"
```

### 6.2 Fallback Strategy

- Si el modelo ML no está disponible → usar solo heurísticas
- Si heurísticas son muy inciertas (0.4-0.6) → usar modelo ML como tie-breaker
- Si ambos coinciden → alta confianza

---

## 7. Checklist de Implementación

- [ ] Dataset preparado y balanceado (mínimo 5K imágenes/clase)
- [ ] Data augmentation configurado
- [ ] Modelo seleccionado (EfficientNet-B2 recomendado)
- [ ] Entrenamiento completado (50 epochs o early stopping)
- [ ] Métricas de evaluación > 85% accuracy
- [ ] Modelo exportado a TorchScript (.pt)
- [ ] Integración en `src/antispoofing_enhanced.py`
- [ ] Pruebas en conjunto de test independiente
- [ ] Comparación con baseline de heurísticas
- [ ] Documentación de rendimiento actualizada

---

## 8. Recursos Adicionales

- **PyTorch Tutorial**: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- **EfficientNet Paper**: https://arxiv.org/abs/1905.11946
- **Anti-Spoofing Datasets**: 
  - CASIA-FASD
  - Replay-Attack
  - OULU-NPU
- **Transfer Learning**: Usar modelos pre-entrenados en ImageNet como punto de partida

---

## 9. Notas Finales

- **Tiempo estimado de entrenamiento**: 2-4 horas (GPU) o 8-12 horas (CPU)
- **Hardware recomendado**: GPU con 8GB+ RAM (NVIDIA RTX 3060 o superior)
- **Versión de PyTorch**: 2.0+ recomendada
- **Monitoreo**: Usar TensorBoard o Weights & Biases para visualizar métricas

---

**Última actualización**: 2025-11-27  
**Versión**: 1.0

