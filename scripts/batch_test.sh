#!/bin/bash

# Batch testing script for Anti-Spoofing detector
# Tests all images in test_pics/ folder

TEST_DIR="/Users/aegp17/Dropbox/Mac/Documents/code/fs-code/anti-spoofing/test_pics"
API_URL="http://localhost:8000"

cd "$TEST_DIR" || exit 1

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🧪 PRUEBAS COMPLETAS - Anti-Spoofing Detector                ║"
echo "║  Dataset: Imágenes en test_pics/                              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

TOTAL=0
DOCUMENTS=0
SELFIES=0
ERRORS=0

# Arrays to store results
declare -a DOC_FILES
declare -a SELFIE_FILES
declare -a ERROR_FILES

# Test all images
for file in *.jpg *.jpeg *.png 2>/dev/null; do
    if [ -f "$file" ]; then
        TOTAL=$((TOTAL + 1))
        
        # Make request
        RESULT=$(curl -s -X POST "$API_URL/detect" -F "file=@$file" 2>/dev/null)
        RESPONSE=$(echo "$RESULT" | grep -o '"response":"[^"]*"' | cut -d'"' -f4)
        METHOD=$(echo "$RESULT" | grep -o '"method":"[^"]*"' | cut -d'"' -f4)
        
        if [ -z "$RESPONSE" ]; then
            ERRORS=$((ERRORS + 1))
            ERROR_FILES+=("$file")
            echo "❌ $file"
        elif [ "$RESPONSE" = "id document detect" ]; then
            DOCUMENTS=$((DOCUMENTS + 1))
            DOC_FILES+=("$file")
            echo "📄 $file"
            echo "   └─ Method: $METHOD"
        else
            SELFIES=$((SELFIES + 1))
            SELFIE_FILES+=("$file")
            echo "🤳 $file"
            echo "   └─ Method: $METHOD"
        fi
        
        # Progress indicator
        if [ $((TOTAL % 50)) -eq 0 ]; then
            echo ""
            echo "   ⏳ Procesadas $TOTAL imágenes..."
            echo ""
        fi
    fi
done

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  📊 RESUMEN FINAL DE PRUEBAS                                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Total de imágenes analizadas:     $TOTAL"
echo "✅ Documentos detectados:          $DOCUMENTS"
if [ $DOCUMENTS -gt 0 ]; then
    PCT=$((DOCUMENTS * 100 / TOTAL))
    echo "   Porcentaje: $PCT%"
fi
echo ""
echo "✅ Selfies detectados:             $SELFIES"
if [ $SELFIES -gt 0 ]; then
    PCT=$((SELFIES * 100 / TOTAL))
    echo "   Porcentaje: $PCT%"
fi
echo ""

if [ $ERRORS -gt 0 ]; then
    echo "❌ Errores en procesamiento:       $ERRORS"
    echo ""
fi

ACCURACY=$((100 * (TOTAL - ERRORS) / TOTAL))
echo "📈 Precisión general: $ACCURACY%"
echo ""

# Show distribution
if [ $TOTAL -gt 0 ]; then
    echo "📋 Distribución:"
    echo "   • Documentos: $((DOCUMENTS * 100 / TOTAL))%"
    echo "   • Selfies:    $((SELFIES * 100 / TOTAL))%"
    if [ $ERRORS -gt 0 ]; then
        echo "   • Errores:    $((ERRORS * 100 / TOTAL))%"
    fi
    echo ""
fi

# Show sample files
echo "📂 Muestras detectadas como DOCUMENTOS:"
if [ ${#DOC_FILES[@]} -gt 0 ]; then
    for i in "${!DOC_FILES[@]}"; do
        if [ $i -lt 5 ]; then
            echo "   • ${DOC_FILES[$i]}"
        fi
    done
    if [ ${#DOC_FILES[@]} -gt 5 ]; then
        echo "   ... y $((${#DOC_FILES[@]} - 5)) más"
    fi
else
    echo "   (ninguno)"
fi
echo ""

echo "📂 Muestras detectadas como SELFIES:"
if [ ${#SELFIE_FILES[@]} -gt 0 ]; then
    for i in "${!SELFIE_FILES[@]}"; do
        if [ $i -lt 5 ]; then
            echo "   • ${SELFIE_FILES[$i]}"
        fi
    done
    if [ ${#SELFIE_FILES[@]} -gt 5 ]; then
        echo "   ... y $((${#SELFIE_FILES[@]} - 5)) más"
    fi
else
    echo "   (ninguno)"
fi
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "✅ Pruebas completadas"
echo "╚════════════════════════════════════════════════════════════════╝"

