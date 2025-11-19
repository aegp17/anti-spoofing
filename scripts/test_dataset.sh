#!/bin/bash

# Deepfake Detection Testing with Random Dataset Selection
# Prueba el módulo deepfake con imágenes seleccionadas al azar del Dataset

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SERVICE_URL="http://localhost:8000"
DATASET_PATH="/Users/aegp17/Dropbox/Mac/Documents/code/fs-code/anti-spoofing/Dataset"
NUM_SAMPLES=${1:-10}  # Default: 10 random images

# Statistics
TOTAL_TESTED=0
CORRECT=0
INCORRECT=0
ERRORS=0
REAL_CORRECT=0
REAL_TOTAL=0
FAKE_CORRECT=0
FAKE_TOTAL=0

# Helper functions
print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  $1"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_result() {
    local image=$1
    local expected=$2
    local detected=$3
    local confidence=$4
    
    # Check if correct
    if [ "$expected" = "$detected" ]; then
        print_success "CORRECT: $image"
        CORRECT=$((CORRECT + 1))
        if [ "$expected" = "real" ]; then
            REAL_CORRECT=$((REAL_CORRECT + 1))
        else
            FAKE_CORRECT=$((FAKE_CORRECT + 1))
        fi
    else
        print_error "WRONG: Expected $expected, got $detected - $image"
        INCORRECT=$((INCORRECT + 1))
    fi
    
    echo "          Expected: $expected | Detected: $detected | Confidence: $confidence"
}

# Check service
wait_for_service() {
    print_step "Checking service availability..."
    
    for i in {1..5}; do
        if curl -s "$SERVICE_URL/health" > /dev/null 2>&1; then
            print_success "Service is ready"
            return 0
        fi
        sleep 1
    done
    
    print_error "Service not available at $SERVICE_URL"
    exit 1
}

# Get random images
get_random_images() {
    local category=$1
    local count=$2
    local set=$3  # Test, Train, Validation
    
    local path="$DATASET_PATH/$set/$category"
    
    if [ ! -d "$path" ]; then
        print_error "Directory not found: $path"
        return
    fi
    
    # Get random images
    find "$path" -type f -name "*.jpg" | shuf | head -$count
}

# Test image
test_image() {
    local image_path=$1
    local expected_category=$2
    
    if [ ! -f "$image_path" ]; then
        print_warning "File not found: $(basename "$image_path")"
        ERRORS=$((ERRORS + 1))
        return
    fi
    
    TOTAL_TESTED=$((TOTAL_TESTED + 1))
    
    # Count category
    if [ "$expected_category" = "Real" ]; then
        REAL_TOTAL=$((REAL_TOTAL + 1))
    else
        FAKE_TOTAL=$((FAKE_TOTAL + 1))
    fi
    
    # Make request
    RESPONSE=$(curl -s -X POST "$SERVICE_URL/analyze/deepfake/image" \
        -F "file=@$image_path" \
        -H "accept: application/json" 2>/dev/null)
    
    # Extract response
    RESULT=$(echo "$RESPONSE" | grep -o '"response":"[^"]*"' | cut -d'"' -f4 2>/dev/null)
    CONFIDENCE=$(echo "$RESPONSE" | grep -o '"confidence":[0-9.]*' | cut -d':' -f2 2>/dev/null)
    
    if [ -z "$RESULT" ]; then
        print_error "Failed to analyze: $(basename "$image_path")"
        ERRORS=$((ERRORS + 1))
        return
    fi
    
    # Determine detected category
    local detected_category="Unknown"
    if [ "$RESULT" = "likely_real" ]; then
        detected_category="Real"
    elif [ "$RESULT" = "likely_deepfake" ]; then
        detected_category="Fake"
    fi
    
    # Print result
    local expected_lower=$(echo "$expected_category" | tr '[:upper:]' '[:lower:]')
    local detected_lower=$(echo "$detected_category" | tr '[:upper:]' '[:lower:]')
    
    print_result "$(basename "$image_path")" "$expected_lower" "$detected_lower" "$CONFIDENCE"
}

# Main testing
main() {
    print_header "🎭 DEEPFAKE DETECTION - RANDOM DATASET TEST"
    
    # Wait for service
    wait_for_service
    
    echo "Probando con $NUM_SAMPLES imágenes aleatorias del Dataset"
    echo ""
    
    # Calculate split
    HALF_SAMPLES=$((NUM_SAMPLES / 2))
    REAL_SAMPLES=$((NUM_SAMPLES - HALF_SAMPLES))
    FAKE_SAMPLES=$HALF_SAMPLES
    
    # Get random images from Test set
    print_header "Testing Real Images"
    
    REAL_IMAGES=$(get_random_images "Real" "$REAL_SAMPLES" "Test")
    
    for image in $REAL_IMAGES; do
        test_image "$image" "Real"
    done
    
    print_header "Testing Fake Images"
    
    FAKE_IMAGES=$(get_random_images "Fake" "$FAKE_SAMPLES" "Test")
    
    for image in $FAKE_IMAGES; do
        test_image "$image" "Fake"
    done
    
    # Print summary
    print_header "📊 TEST SUMMARY"
    
    echo "Total de pruebas realizadas: $TOTAL_TESTED"
    echo "Correctas: $CORRECT"
    echo "Incorrectas: $INCORRECT"
    echo "Errores: $ERRORS"
    echo ""
    
    if [ $TOTAL_TESTED -gt 0 ]; then
        ACCURACY=$((CORRECT * 100 / TOTAL_TESTED))
        echo -e "${BLUE}═══════════════════════════════════════════${NC}"
        echo -e "${GREEN}Precisión General: $ACCURACY% ($CORRECT/$TOTAL_TESTED)${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════${NC}"
        echo ""
        
        if [ $REAL_TOTAL -gt 0 ]; then
            REAL_ACCURACY=$((REAL_CORRECT * 100 / REAL_TOTAL))
            echo "Real images:  $REAL_ACCURACY% ($REAL_CORRECT/$REAL_TOTAL) correctas"
        fi
        
        if [ $FAKE_TOTAL -gt 0 ]; then
            FAKE_ACCURACY=$((FAKE_CORRECT * 100 / FAKE_TOTAL))
            echo "Fake images:  $FAKE_ACCURACY% ($FAKE_CORRECT/$FAKE_TOTAL) correctas"
        fi
    fi
    
    echo ""
}

# Show dataset info
show_dataset_info() {
    print_header "📚 DATASET STATISTICS"
    
    echo "Test Set:"
    echo "  Real:  $(find "$DATASET_PATH/Test/Real" -type f -name "*.jpg" 2>/dev/null | wc -l) imágenes"
    echo "  Fake:  $(find "$DATASET_PATH/Test/Fake" -type f -name "*.jpg" 2>/dev/null | wc -l) imágenes"
    echo ""
    
    echo "Train Set:"
    echo "  Real:  $(find "$DATASET_PATH/Train/Real" -type f -name "*.jpg" 2>/dev/null | wc -l) imágenes"
    echo "  Fake:  $(find "$DATASET_PATH/Train/Fake" -type f -name "*.jpg" 2>/dev/null | wc -l) imágenes"
    echo ""
    
    echo "Validation Set:"
    echo "  Real:  $(find "$DATASET_PATH/Validation/Real" -type f -name "*.jpg" 2>/dev/null | wc -l) imágenes"
    echo "  Fake:  $(find "$DATASET_PATH/Validation/Fake" -type f -name "*.jpg" 2>/dev/null | wc -l) imágenes"
    echo ""
}

# Run
show_dataset_info
main "$@"

echo ""
print_header "✅ TEST COMPLETE"

