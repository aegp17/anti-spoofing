#!/bin/bash

# Deepfake Detection Random Dataset Testing
# Tests the detector against real dataset with random image selection

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SERVICE_URL="http://localhost:8000"
DATASET_DIR="Dataset/Test"
NUM_SAMPLES=${1:-10}  # Default 10 samples, can be passed as argument

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

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

# Wait for service
wait_for_service() {
    print_step "Waiting for service at $SERVICE_URL..."
    
    for i in {1..30}; do
        if curl -s "$SERVICE_URL/health" > /dev/null 2>&1; then
            print_success "Service is ready"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    
    print_error "Service did not respond in time"
    return 1
}

# Analyze single image
analyze_image() {
    local image_path=$1
    local expected_category=$2
    
    if [ ! -f "$image_path" ]; then
        print_error "File not found: $image_path"
        return 1
    fi
    
    RESPONSE=$(curl -s -X POST "$SERVICE_URL/analyze/deepfake/image" \
        -F "file=@$image_path" \
        -H "accept: application/json")
    
    # Extract fields
    RESULT=$(echo "$RESPONSE" | grep -o '"response":"[^"]*"' | cut -d'"' -f4)
    CONFIDENCE=$(echo "$RESPONSE" | grep -o '"confidence":[0-9.]*' | cut -d':' -f2)
    METHOD=$(echo "$RESPONSE" | grep -o '"method":"[^"]*"' | cut -d'"' -f4)
    
    # Check if result matches expected
    if [ "$expected_category" = "Real" ]; then
        EXPECTED_RESULT="likely_real"
        if [ "$RESULT" = "$EXPECTED_RESULT" ]; then
            MATCH="✓"
            MATCH_COLOR=$GREEN
        else
            MATCH="✗"
            MATCH_COLOR=$RED
        fi
    elif [ "$expected_category" = "Fake" ]; then
        EXPECTED_RESULT="likely_deepfake"
        if [ "$RESULT" = "$EXPECTED_RESULT" ]; then
            MATCH="✓"
            MATCH_COLOR=$GREEN
        else
            MATCH="✗"
            MATCH_COLOR=$RED
        fi
    fi
    
    echo -e "${MATCH_COLOR}${MATCH}${NC} [${expected_category}] ${RESULT} (${CONFIDENCE})"
}

# Main execution
main() {
    print_header "🎭 DEEPFAKE DETECTION - RANDOM DATASET TEST"
    
    # Check dataset
    if [ ! -d "$DATASET_DIR" ]; then
        print_error "Dataset directory not found: $DATASET_DIR"
        exit 1
    fi
    
    # Wait for service
    if ! wait_for_service; then
        print_error "Cannot connect to service"
        exit 1
    fi
    
    # Count total images
    REAL_COUNT=$(find "$DATASET_DIR/Real" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
    FAKE_COUNT=$(find "$DATASET_DIR/Fake" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
    
    print_info "Dataset statistics:"
    echo "  Real images: $REAL_COUNT"
    echo "  Fake images: $FAKE_COUNT"
    echo "  Total: $((REAL_COUNT + FAKE_COUNT))"
    echo ""
    
    # Calculate samples per category
    SAMPLES_PER_CATEGORY=$((NUM_SAMPLES / 2))
    
    print_step "Selecting $NUM_SAMPLES random images ($SAMPLES_PER_CATEGORY per category)..."
    echo ""
    
    # Statistics
    TOTAL_TESTED=0
    CORRECT=0
    REAL_CORRECT=0
    REAL_TESTED=0
    FAKE_CORRECT=0
    FAKE_TESTED=0
    
    # Test Real images
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}TESTING REAL IMAGES${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Get random real images (using sort -R for macOS compatibility)
    REAL_IMAGES=$(find "$DATASET_DIR/Real" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | sort -R | head -$SAMPLES_PER_CATEGORY)
    
    for image in $REAL_IMAGES; do
        TOTAL_TESTED=$((TOTAL_TESTED + 1))
        REAL_TESTED=$((REAL_TESTED + 1))
        
        analyze_image "$image" "Real"
        LAST_RESULT=$?
        
        if [ $LAST_RESULT -eq 0 ]; then
            # Extract confidence to check if it matches
            RESPONSE=$(curl -s -X POST "$SERVICE_URL/analyze/deepfake/image" \
                -F "file=@$image" \
                -H "accept: application/json")
            RESULT=$(echo "$RESPONSE" | grep -o '"response":"[^"]*"' | cut -d'"' -f4)
            
            if [ "$RESULT" = "likely_real" ]; then
                CORRECT=$((CORRECT + 1))
                REAL_CORRECT=$((REAL_CORRECT + 1))
            fi
        fi
    done
    
    echo ""
    
    # Test Fake images
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}TESTING FAKE IMAGES (DEEPFAKES)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Get random fake images (using sort -R for macOS compatibility)
    FAKE_IMAGES=$(find "$DATASET_DIR/Fake" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | sort -R | head -$SAMPLES_PER_CATEGORY)
    
    for image in $FAKE_IMAGES; do
        TOTAL_TESTED=$((TOTAL_TESTED + 1))
        FAKE_TESTED=$((FAKE_TESTED + 1))
        
        analyze_image "$image" "Fake"
        LAST_RESULT=$?
        
        if [ $LAST_RESULT -eq 0 ]; then
            # Extract confidence to check if it matches
            RESPONSE=$(curl -s -X POST "$SERVICE_URL/analyze/deepfake/image" \
                -F "file=@$image" \
                -H "accept: application/json")
            RESULT=$(echo "$RESPONSE" | grep -o '"response":"[^"]*"' | cut -d'"' -f4)
            
            if [ "$RESULT" = "likely_deepfake" ]; then
                CORRECT=$((CORRECT + 1))
                FAKE_CORRECT=$((FAKE_CORRECT + 1))
            fi
        fi
    done
    
    echo ""
    
    # Summary
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}TEST SUMMARY${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    
    ACCURACY=$(awk "BEGIN {printf \"%.1f\", ($CORRECT / $TOTAL_TESTED) * 100}")
    REAL_ACCURACY=$(awk "BEGIN {printf \"%.1f\", ($REAL_CORRECT / $REAL_TESTED) * 100}")
    FAKE_ACCURACY=$(awk "BEGIN {printf \"%.1f\", ($FAKE_CORRECT / $FAKE_TESTED) * 100}")
    
    echo ""
    echo "Overall Results:"
    echo "  Total tested:      $TOTAL_TESTED"
    echo "  Correct:           $CORRECT"
    echo "  Accuracy:          $ACCURACY%"
    echo ""
    echo "Real Images:"
    echo "  Tested:            $REAL_TESTED"
    echo "  Correct:           $REAL_CORRECT"
    echo "  Accuracy:          $REAL_ACCURACY%"
    echo ""
    echo "Fake Images (Deepfakes):"
    echo "  Tested:            $FAKE_TESTED"
    echo "  Correct:           $FAKE_CORRECT"
    echo "  Accuracy:          $FAKE_ACCURACY%"
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Color-coded summary
    if (( $(echo "$ACCURACY >= 80" | bc -l) )); then
        print_success "Excellent accuracy: $ACCURACY%"
    elif (( $(echo "$ACCURACY >= 60" | bc -l) )); then
        print_warning "Good accuracy: $ACCURACY%"
    else
        print_error "Low accuracy: $ACCURACY%"
    fi
    
    echo ""
}

# Run main
main

