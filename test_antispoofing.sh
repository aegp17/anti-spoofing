#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

DATASET_DIR="/Users/aegp17/Dropbox/Mac/Documents/code/fs-code/anti-spoofing/Dataset/Test"
SAMPLES_PER_CATEGORY=100
API_URL="http://localhost:8000/detect/antispoofing"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  🎭 Anti-Spoofing Detection Test       ║${NC}"
echo -e "${BLUE}║  100 Real + 100 Fake Samples           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Get random real images
REAL_IMAGES=($(find "$DATASET_DIR/Real" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | sort -R | head -$SAMPLES_PER_CATEGORY))

# Get random fake images
FAKE_IMAGES=($(find "$DATASET_DIR/Fake" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | sort -R | head -$SAMPLES_PER_CATEGORY))

# Counters
REAL_SELFIE_CORRECT=0
REAL_DOCUMENT_WRONG=0
REAL_CONF_TOTAL=0

FAKE_DETECTED_CORRECT=0
FAKE_MISSED=0
FAKE_CONF_TOTAL=0

echo -e "${CYAN}→ Probando 100 imágenes REALES...${NC}"

COUNTER=1
for img in "${REAL_IMAGES[@]}"; do
    RESPONSE=$(curl -s -X POST "$API_URL" -F "file=@$img" 2>/dev/null)
    RESULT=$(echo "$RESPONSE" | jq -r '.response' 2>/dev/null)
    CONFIDENCE=$(echo "$RESPONSE" | jq -r '.confidence' 2>/dev/null)
    
    # For anti-spoofing, we expect "selfie"
    if [ "$RESULT" == "selfie" ]; then
        ((REAL_SELFIE_CORRECT++))
        # Extract integer part of confidence
        CONF_INT=${CONFIDENCE%.*}
        REAL_CONF_TOTAL=$((REAL_CONF_TOTAL + CONF_INT))
    else
        ((REAL_DOCUMENT_WRONG++))
    fi
    
    # Progress indicator
    if (( COUNTER % 10 == 0 )); then
        echo -ne "\r  [$COUNTER/$SAMPLES_PER_CATEGORY] "
    fi
    ((COUNTER++))
done

if [ "$REAL_SELFIE_CORRECT" -gt 0 ]; then
    REAL_CONFIDENCE_AVG=$((REAL_CONF_TOTAL / REAL_SELFIE_CORRECT))
else
    REAL_CONFIDENCE_AVG=0
fi

echo -e "\r  [$SAMPLES_PER_CATEGORY/$SAMPLES_PER_CATEGORY] ✓ Completado"
echo ""

echo -e "${CYAN}→ Probando 100 imágenes FAKE (spoofing)...${NC}"

COUNTER=1
for img in "${FAKE_IMAGES[@]}"; do
    RESPONSE=$(curl -s -X POST "$API_URL" -F "file=@$img" 2>/dev/null)
    RESULT=$(echo "$RESPONSE" | jq -r '.response' 2>/dev/null)
    CONFIDENCE=$(echo "$RESPONSE" | jq -r '.confidence' 2>/dev/null)
    
    # For anti-spoofing, we want to detect these as NOT selfie (or document)
    if [ "$RESULT" != "selfie" ]; then
        ((FAKE_DETECTED_CORRECT++))
        CONF_INT=${CONFIDENCE%.*}
        FAKE_CONF_TOTAL=$((FAKE_CONF_TOTAL + CONF_INT))
    else
        ((FAKE_MISSED++))
    fi
    
    # Progress indicator
    if (( COUNTER % 10 == 0 )); then
        echo -ne "\r  [$COUNTER/$SAMPLES_PER_CATEGORY] "
    fi
    ((COUNTER++))
done

if [ "$FAKE_DETECTED_CORRECT" -gt 0 ]; then
    FAKE_CONFIDENCE_AVG=$((FAKE_CONF_TOTAL / FAKE_DETECTED_CORRECT))
else
    FAKE_CONFIDENCE_AVG=0
fi

echo -e "\r  [$SAMPLES_PER_CATEGORY/$SAMPLES_PER_CATEGORY] ✓ Completado"
echo ""
echo ""

# Calculate percentages
REAL_ACCURACY=$((REAL_SELFIE_CORRECT * 100 / SAMPLES_PER_CATEGORY))
FAKE_ACCURACY=$((FAKE_DETECTED_CORRECT * 100 / SAMPLES_PER_CATEGORY))
TOTAL_CORRECT=$((REAL_SELFIE_CORRECT + FAKE_DETECTED_CORRECT))
TOTAL_SAMPLES=$((SAMPLES_PER_CATEGORY * 2))
OVERALL_ACCURACY=$((TOTAL_CORRECT * 100 / TOTAL_SAMPLES))

# Print table
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   📊 RESULTADOS (TABLA RESUMEN)                ║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║${NC} Categoría      │   Correctas   │  Porcentaje  │   Confianza  ${BLUE}║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
printf "${BLUE}║${NC} %-14s │ %3d/%-3d      │    %3d%%      │    %3d%%      ${BLUE}║${NC}\n" "REAL (Selfie)" "$REAL_SELFIE_CORRECT" "$SAMPLES_PER_CATEGORY" "$REAL_ACCURACY" "$REAL_CONFIDENCE_AVG"
printf "${BLUE}║${NC} %-14s │ %3d/%-3d      │    %3d%%      │    %3d%%      ${BLUE}║${NC}\n" "FAKE (Spoofing)" "$FAKE_DETECTED_CORRECT" "$SAMPLES_PER_CATEGORY" "$FAKE_ACCURACY" "$FAKE_CONFIDENCE_AVG"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
printf "${BLUE}║${NC} %-14s │ %3d/%-3d      │    %3d%%      │    -         ${BLUE}║${NC}\n" "GENERAL" "$TOTAL_CORRECT" "$TOTAL_SAMPLES" "$OVERALL_ACCURACY"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
