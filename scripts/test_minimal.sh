#!/bin/bash

# Minimal Deepfake Testing Script
# Tests the MVP implementation without external dependencies

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_URL="http://localhost:8000"
TEST_DIR="test_pics"

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

# Health check
test_health() {
    print_header "HEALTH CHECK"
    print_step "Testing /health endpoint..."
    
    RESPONSE=$(curl -s "$SERVICE_URL/health")
    echo "Response: $RESPONSE"
    
    if echo "$RESPONSE" | grep -q "healthy"; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
        return 1
    fi
}

# Test image detection
test_image() {
    local image_path=$1
    local label=$2
    
    if [ ! -f "$image_path" ]; then
        print_warning "File not found: $image_path"
        return 1
    fi
    
    echo ""
    print_step "Testing: $label"
    print_step "File: $image_path"
    
    RESPONSE=$(curl -s -X POST "$SERVICE_URL/analyze/deepfake/image" \
        -F "file=@$image_path" \
        -H "accept: application/json")
    
    echo "Response:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    
    # Extract response field
    RESULT=$(echo "$RESPONSE" | grep -o '"response":"[^"]*"' | cut -d'"' -f4)
    CONFIDENCE=$(echo "$RESPONSE" | grep -o '"confidence":[0-9.]*' | cut -d':' -f2)
    
    echo ""
    print_success "Analysis complete"
    echo "  Result: $RESULT"
    echo "  Confidence: $CONFIDENCE"
}

# Test batch images
test_batch() {
    print_header "BATCH IMAGE TESTING"
    
    if [ ! -d "$TEST_DIR" ]; then
        print_warning "Test directory not found: $TEST_DIR"
        echo "  Create it with: mkdir -p $TEST_DIR"
        echo "  Add test images to it"
        return 0
    fi
    
    # Count images
    IMAGE_COUNT=$(find "$TEST_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
    
    if [ $IMAGE_COUNT -eq 0 ]; then
        print_warning "No images found in $TEST_DIR"
        echo "  Add .jpg, .jpeg, or .png files to test"
        return 0
    fi
    
    print_step "Found $IMAGE_COUNT images in $TEST_DIR"
    echo ""
    
    TOTAL=0
    REAL=0
    FAKE=0
    ERRORS=0
    
    for image_file in "$TEST_DIR"/*.{jpg,jpeg,png} 2>/dev/null; do
        if [ -f "$image_file" ]; then
            TOTAL=$((TOTAL + 1))
            
            echo "[$TOTAL] Testing: $(basename "$image_file")"
            
            RESPONSE=$(curl -s -X POST "$SERVICE_URL/analyze/deepfake/image" \
                -F "file=@$image_file" \
                -H "accept: application/json")
            
            RESULT=$(echo "$RESPONSE" | grep -o '"response":"[^"]*"' | cut -d'"' -f4)
            CONFIDENCE=$(echo "$RESPONSE" | grep -o '"confidence":[0-9.]*' | cut -d':' -f2)
            
            if [ -z "$RESULT" ]; then
                print_error "Failed to analyze"
                ERRORS=$((ERRORS + 1))
            elif [ "$RESULT" = "likely_real" ]; then
                print_success "Real (confidence: $CONFIDENCE)"
                REAL=$((REAL + 1))
            elif [ "$RESULT" = "likely_deepfake" ]; then
                print_warning "Deepfake (confidence: $CONFIDENCE)"
                FAKE=$((FAKE + 1))
            else
                print_warning "$RESULT (confidence: $CONFIDENCE)"
            fi
        fi
    done
    
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "BATCH TEST SUMMARY"
    echo "════════════════════════════════════════════════════════════════"
    echo "Total analyzed:  $TOTAL"
    echo "Real images:     $REAL"
    echo "Deepfakes:       $FAKE"
    echo "Errors:          $ERRORS"
    echo "════════════════════════════════════════════════════════════════"
}

# Main execution
main() {
    print_header "🎭 MINIMAL DEEPFAKE DETECTION TEST"
    
    # Wait for service
    if ! wait_for_service; then
        print_error "Cannot connect to service"
        exit 1
    fi
    
    # Health check
    if ! test_health; then
        print_error "Health check failed"
        exit 1
    fi
    
    # Single image tests (if provided as arguments)
    if [ $# -gt 0 ]; then
        for arg in "$@"; do
            test_image "$arg" "$(basename "$arg")"
        done
    else
        print_warning "No specific images provided"
        echo "  Usage: $0 [image1.jpg] [image2.jpg] ..."
    fi
    
    # Batch test
    test_batch
    
    echo ""
    print_header "✅ TEST SUITE COMPLETE"
}

# Run main
main "$@"

