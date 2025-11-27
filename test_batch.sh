#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DATASET_DIR="/Users/aegp17/Dropbox/Mac/Documents/code/fs-code/anti-spoofing/Dataset/Test"
SAMPLES_PER_CATEGORY=30
API_URL="http://localhost:8000/analyze/deepfake/image"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🧪 Deepfake Detection Test - Batch Mode${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get random real images
REAL_IMAGES=$(find "$DATASET_DIR/Real" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | sort -R | head -$SAMPLES_PER_CATEGORY)

# Get random fake images
FAKE_IMAGES=$(find "$DATASET_DIR/Fake" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | sort -R | head -$SAMPLES_PER_CATEGORY)

# Counters
REAL_CORRECT=0
REAL_WRONG=0
FAKE_CORRECT=0
FAKE_WRONG=0

echo -e "${YELLOW}📊 Testing REAL images (${SAMPLES_PER_CATEGORY} samples):${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

COUNTER=1
for img in $REAL_IMAGES; do
    RESPONSE=$(curl -s -X POST "$API_URL" -F "file=@$img")
    RESULT=$(echo "$RESPONSE" | jq -r '.response' 2>/dev/null)
    CONFIDENCE=$(echo "$RESPONSE" | jq -r '.confidence' 2>/dev/null)
    
    if [ "$RESULT" == "likely_real" ]; then
        echo -e "${GREEN}✓${NC} [$COUNTER/$SAMPLES_PER_CATEGORY] REAL detected correctly (confidence: $CONFIDENCE)"
        ((REAL_CORRECT++))
    else
        echo -e "${RED}✗${NC} [$COUNTER/$SAMPLES_PER_CATEGORY] REAL misclassified as $RESULT (confidence: $CONFIDENCE)"
        ((REAL_WRONG++))
    fi
    ((COUNTER++))
done

echo ""
echo -e "${YELLOW}📊 Testing FAKE images (${SAMPLES_PER_CATEGORY} samples):${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

COUNTER=1
for img in $FAKE_IMAGES; do
    RESPONSE=$(curl -s -X POST "$API_URL" -F "file=@$img")
    RESULT=$(echo "$RESPONSE" | jq -r '.response' 2>/dev/null)
    CONFIDENCE=$(echo "$RESPONSE" | jq -r '.confidence' 2>/dev/null)
    
    if [ "$RESULT" == "likely_deepfake" ]; then
        echo -e "${GREEN}✓${NC} [$COUNTER/$SAMPLES_PER_CATEGORY] FAKE detected correctly (confidence: $CONFIDENCE)"
        ((FAKE_CORRECT++))
    else
        echo -e "${RED}✗${NC} [$COUNTER/$SAMPLES_PER_CATEGORY] FAKE misclassified as $RESULT (confidence: $CONFIDENCE)"
        ((FAKE_WRONG++))
    fi
    ((COUNTER++))
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}📈 RESULTS SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

REAL_ACCURACY=$((REAL_CORRECT * 100 / SAMPLES_PER_CATEGORY))
FAKE_ACCURACY=$((FAKE_CORRECT * 100 / SAMPLES_PER_CATEGORY))
TOTAL_CORRECT=$((REAL_CORRECT + FAKE_CORRECT))
TOTAL_SAMPLES=$((SAMPLES_PER_CATEGORY * 2))
OVERALL_ACCURACY=$((TOTAL_CORRECT * 100 / TOTAL_SAMPLES))

echo -e "🔍 REAL Images:"
echo -e "   Correct:   ${GREEN}${REAL_CORRECT}/${SAMPLES_PER_CATEGORY}${NC} (${GREEN}${REAL_ACCURACY}%${NC})"
echo -e "   Wrong:     ${RED}${REAL_WRONG}/${SAMPLES_PER_CATEGORY}${NC}"
echo ""

echo -e "🔍 FAKE Images:"
echo -e "   Correct:   ${GREEN}${FAKE_CORRECT}/${SAMPLES_PER_CATEGORY}${NC} (${GREEN}${FAKE_ACCURACY}%${NC})"
echo -e "   Wrong:     ${RED}${FAKE_WRONG}/${SAMPLES_PER_CATEGORY}${NC}"
echo ""

echo -e "🎯 OVERALL:"
echo -e "   Total Correct: ${GREEN}${TOTAL_CORRECT}/${TOTAL_SAMPLES}${NC}"
echo -e "   Overall Accuracy: ${GREEN}${OVERALL_ACCURACY}%${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"

