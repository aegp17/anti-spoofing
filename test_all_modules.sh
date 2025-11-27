#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

ANTISPOOFING_DATASET_DIR="/Users/aegp17/Dropbox/Mac/Documents/code/fs-code/anti-spoofing/Dataset/Test"
DOCUMENT_TEST_DIR="/Users/aegp17/Dropbox/Mac/Documents/code/fs-code/anti-spoofing/test_pics"
SAMPLES_PER_CATEGORY=100

ANTISPOOFING_URL="http://localhost:8000/detect/antispoofing"
DOCUMENT_URL="http://localhost:8000/detect/document"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           🎭 MULTI-MODULE DETECTION TEST                      ║${NC}"
echo -e "${BLUE}║  Anti-Spoofing: 100 Real + 100 Fake | Document Detection     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# TEST 1: ANTI-SPOOFING DETECTION
# ============================================================================

echo -e "${YELLOW}═════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}1️⃣  ANTI-SPOOFING DETECTION (Real Selfies vs Fake)${NC}"
echo -e "${YELLOW}═════════════════════════════════════════════════════════════════${NC}"
echo ""

# Get random real images
REAL_IMAGES=($(find "$ANTISPOOFING_DATASET_DIR/Real" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | sort -R | head -$SAMPLES_PER_CATEGORY))

# Get random fake images
FAKE_IMAGES=($(find "$ANTISPOOFING_DATASET_DIR/Fake" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | sort -R | head -$SAMPLES_PER_CATEGORY))

# Counters for anti-spoofing
REAL_SELFIE_CORRECT=0
REAL_CONF_TOTAL=0
FAKE_DETECTED_CORRECT=0
FAKE_CONF_TOTAL=0

echo -e "${CYAN}→ Probando 100 imágenes REALES (selfies auténticas)...${NC}"

COUNTER=1
for img in "${REAL_IMAGES[@]}"; do
    RESPONSE=$(curl -s -X POST "$ANTISPOOFING_URL" -F "file=@$img" 2>/dev/null)
    RESULT=$(echo "$RESPONSE" | jq -r '.response' 2>/dev/null)
    CONFIDENCE=$(echo "$RESPONSE" | jq -r '.confidence' 2>/dev/null)
    
    if [ "$RESULT" == "selfie" ]; then
        ((REAL_SELFIE_CORRECT++))
        CONF_INT=${CONFIDENCE%.*}
        REAL_CONF_TOTAL=$((REAL_CONF_TOTAL + CONF_INT))
    fi
    
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

echo -e "${CYAN}→ Probando 100 imágenes FAKE (spoofed/fake)...${NC}"

COUNTER=1
for img in "${FAKE_IMAGES[@]}"; do
    RESPONSE=$(curl -s -X POST "$ANTISPOOFING_URL" -F "file=@$img" 2>/dev/null)
    RESULT=$(echo "$RESPONSE" | jq -r '.response' 2>/dev/null)
    CONFIDENCE=$(echo "$RESPONSE" | jq -r '.confidence' 2>/dev/null)
    
    if [ "$RESULT" != "selfie" ]; then
        ((FAKE_DETECTED_CORRECT++))
        CONF_INT=${CONFIDENCE%.*}
        FAKE_CONF_TOTAL=$((FAKE_CONF_TOTAL + CONF_INT))
    fi
    
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

# Calculate anti-spoofing percentages
REAL_ACCURACY=$((REAL_SELFIE_CORRECT * 100 / SAMPLES_PER_CATEGORY))
FAKE_ACCURACY=$((FAKE_DETECTED_CORRECT * 100 / SAMPLES_PER_CATEGORY))
ANTISPOOFING_TOTAL_CORRECT=$((REAL_SELFIE_CORRECT + FAKE_DETECTED_CORRECT))
ANTISPOOFING_TOTAL_SAMPLES=$((SAMPLES_PER_CATEGORY * 2))
ANTISPOOFING_OVERALL=$((ANTISPOOFING_TOTAL_CORRECT * 100 / ANTISPOOFING_TOTAL_SAMPLES))

# Print anti-spoofing results
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       🎭 ANTI-SPOOFING RESULTS (Real Selfies vs Fake)         ║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║${NC} Categoría      │   Correctas   │  Porcentaje  │   Confianza  ${BLUE}║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
printf "${BLUE}║${NC} %-14s │ %3d/%-3d      │    %3d%%      │    %3d%%      ${BLUE}║${NC}\n" "REAL (Selfie)" "$REAL_SELFIE_CORRECT" "$SAMPLES_PER_CATEGORY" "$REAL_ACCURACY" "$REAL_CONFIDENCE_AVG"
printf "${BLUE}║${NC} %-14s │ %3d/%-3d      │    %3d%%      │    %3d%%      ${BLUE}║${NC}\n" "FAKE (Spoofing)" "$FAKE_DETECTED_CORRECT" "$SAMPLES_PER_CATEGORY" "$FAKE_ACCURACY" "$FAKE_CONFIDENCE_AVG"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
printf "${BLUE}║${NC} %-14s │ %3d/%-3d      │    %3d%%      │    -         ${BLUE}║${NC}\n" "GENERAL" "$ANTISPOOFING_TOTAL_CORRECT" "$ANTISPOOFING_TOTAL_SAMPLES" "$ANTISPOOFING_OVERALL"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# TEST 2: DOCUMENT DETECTION
# ============================================================================

echo -e "${YELLOW}═════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}2️⃣  DOCUMENT DETECTION (ID Document vs Selfie)${NC}"
echo -e "${YELLOW}═════════════════════════════════════════════════════════════════${NC}"
echo ""

# Get all images from test_pics
TEST_PICS=($(find "$DOCUMENT_TEST_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | sort))

if [ ${#TEST_PICS[@]} -eq 0 ]; then
    echo -e "${RED}❌ No test images found in $DOCUMENT_TEST_DIR${NC}"
    echo ""
else
    echo -e "${CYAN}→ Probando ${#TEST_PICS[@]} imágenes de test_pics...${NC}"
    echo ""
    
    DOCUMENT_COUNT=0
    SELFIE_COUNT=0
    UNKNOWN_COUNT=0
    
    COUNTER=1
    for img in "${TEST_PICS[@]}"; do
        FILENAME=$(basename "$img")
        RESPONSE=$(curl -s -X POST "$DOCUMENT_URL" -F "file=@$img" 2>/dev/null)
        RESULT=$(echo "$RESPONSE" | jq -r '.response' 2>/dev/null)
        METHOD=$(echo "$RESPONSE" | jq -r '.method' 2>/dev/null)
        
        # Determine type
        if [ "$RESULT" == "id document detect" ]; then
            echo -e "  ${GREEN}✓${NC} [$COUNTER/${#TEST_PICS[@]}] DOCUMENTO: $FILENAME (método: $METHOD)"
            ((DOCUMENT_COUNT++))
        elif [ "$RESULT" == "is selfie" ]; then
            echo -e "  ${GREEN}✓${NC} [$COUNTER/${#TEST_PICS[@]}] SELFIE: $FILENAME (método: $METHOD)"
            ((SELFIE_COUNT++))
        else
            echo -e "  ${RED}?${NC} [$COUNTER/${#TEST_PICS[@]}] DESCONOCIDO: $FILENAME (respuesta: $RESULT)"
            ((UNKNOWN_COUNT++))
        fi
        
        ((COUNTER++))
    done
    
    echo ""
    
    # Print document detection results
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║        📄 DOCUMENT DETECTION RESULTS (ID vs Selfie)           ║${NC}"
    echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC}  Tipo          │   Cantidad   │  Porcentaje                  ${BLUE}║${NC}"
    echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
    DOCUMENT_PERCENT=$((DOCUMENT_COUNT * 100 / (${#TEST_PICS[@]} + 1)))
    SELFIE_PERCENT=$((SELFIE_COUNT * 100 / (${#TEST_PICS[@]} + 1)))
    UNKNOWN_PERCENT=$((UNKNOWN_COUNT * 100 / (${#TEST_PICS[@]} + 1)))
    printf "${BLUE}║${NC}  Documentos    │     %3d      │    %3d%%                     ${BLUE}║${NC}\n" "$DOCUMENT_COUNT" "$DOCUMENT_PERCENT"
    printf "${BLUE}║${NC}  Selfies       │     %3d      │    %3d%%                     ${BLUE}║${NC}\n" "$SELFIE_COUNT" "$SELFIE_PERCENT"
    printf "${BLUE}║${NC}  Desconocidos  │     %3d      │    %3d%%                     ${BLUE}║${NC}\n" "$UNKNOWN_COUNT" "$UNKNOWN_PERCENT"
    echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
    printf "${BLUE}║${NC}  TOTAL         │     %3d      │   100%%                      ${BLUE}║${NC}\n" "${#TEST_PICS[@]}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    📊 RESUMEN GENERAL                         ║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║${NC} Módulo                 │ Precisión │ Confianza Promedio     ${BLUE}║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
printf "${BLUE}║${NC} %-22s │   %3d%%   │ Real: %3d%% | Fake: %3d%%  ${BLUE}║${NC}\n" "🎭 Anti-Spoofing" "$ANTISPOOFING_OVERALL" "$REAL_CONFIDENCE_AVG" "$FAKE_CONFIDENCE_AVG"
if [ ${#TEST_PICS[@]} -gt 0 ]; then
    printf "${BLUE}║${NC} %-22s │  Docs: %3d%% │ Selfies: %3d%%          ${BLUE}║${NC}\n" "📄 Document Detection" "$DOCUMENT_PERCENT" "$SELFIE_PERCENT"
fi
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

