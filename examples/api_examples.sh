#!/bin/bash
# Example API calls for anti-spoofing detector

API_URL="http://localhost:8000"

echo "🧪 Anti-Spoofing Detector - Example API Calls"
echo "=============================================="
echo ""

# Check health
echo "1️⃣  Health Check:"
echo "   curl -X GET $API_URL/health"
echo ""

# Single image detection
echo "2️⃣  Single Image Detection:"
echo "   curl -X POST $API_URL/detect \\"
echo "     -F 'file=@/path/to/document.jpg'"
echo ""
echo "   Respuesta esperada (documento):"
echo '   { "response": "id document detect", "method": "heuristic_rule_1" }'
echo ""
echo "   Respuesta esperada (selfie):"
echo '   { "response": "is selfie", "confidence": 0.92, "method": "ml_model" }'
echo ""

# Batch detection
echo "3️⃣  Batch Detection (múltiples imágenes):"
echo "   curl -X POST $API_URL/detect/batch \\"
echo "     -F 'files=@image1.jpg' \\"
echo "     -F 'files=@image2.jpg' \\"
echo "     -F 'files=@image3.png'"
echo ""

# Interactive Swagger documentation
echo "4️⃣  API Documentation (Swagger UI):"
echo "   Abre en el navegador: $API_URL/docs"
echo ""

# ReDoc documentation
echo "5️⃣  Alternative Documentation (ReDoc):"
echo "   Abre en el navegador: $API_URL/redoc"
echo ""

echo "=============================================="
echo "⚡ Para ejecutar ejemplos reales:"
echo "   1. Inicia el servidor: python main.py"
echo "   2. Descarga imágenes de prueba"
echo "   3. Ejecuta los comandos curl anteriores"
echo ""

