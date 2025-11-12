#!/bin/bash

# Testing script for Anti-Spoofing Docker API
# Usage: bash scripts/test_docker.sh

API_URL="${API_URL:-http://localhost:8000}"
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${COLOR_BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Anti-Spoofing Detector - Docker Testing Suite            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# Test 1: Health Check
echo -e "${COLOR_BLUE}Test 1: Health Check${NC}"
echo "─────────────────────"
RESPONSE=$(curl -s -w "\n%{http_code}" "$API_URL/health")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${COLOR_GREEN}✓${NC} Health check passed (HTTP 200)"
    echo "  Response: $BODY"
else
    echo -e "${COLOR_RED}✗${NC} Health check failed (HTTP $HTTP_CODE)"
    echo "  Make sure container is running: docker-compose up -d"
    exit 1
fi
echo ""

# Test 2: Document Detection (if file exists)
if [ -f "ceduladelantera.jpg" ]; then
    echo -e "${COLOR_BLUE}Test 2: Document Detection${NC}"
    echo "──────────────────────────"
    START=$(date +%s%N)
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/detect" \
        -F "file=@ceduladelantera.jpg")
    END=$(date +%s%N)
    LATENCY=$(( (END - START) / 1000000 ))
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | head -n-1)
    
    if [ "$HTTP_CODE" = "200" ]; then
        RESULT=$(echo "$BODY" | grep -o '"response":"[^"]*"' | cut -d'"' -f4)
        METHOD=$(echo "$BODY" | grep -o '"method":"[^"]*"' | cut -d'"' -f4)
        
        if [ "$RESULT" = "id document detect" ]; then
            echo -e "${COLOR_GREEN}✓${NC} Correctly identified as DOCUMENT"
            echo "  Result: $RESULT"
            echo "  Method: $METHOD"
            echo "  Latency: ${LATENCY}ms"
        else
            echo -e "${COLOR_YELLOW}⚠${NC}  Unexpected result: $RESULT"
            echo "  Expected: id document detect"
            echo "  Response: $BODY"
        fi
    else
        echo -e "${COLOR_RED}✗${NC} Request failed (HTTP $HTTP_CODE)"
        echo "  Response: $BODY"
    fi
    echo ""
else
    echo -e "${COLOR_YELLOW}⚠${NC}  Skipping Test 2: ceduladelantera.jpg not found"
    echo ""
fi

# Test 3: Selfie Detection (if file exists)
if [ -f "perfilfoto.jpeg" ]; then
    echo -e "${COLOR_BLUE}Test 3: Selfie Detection${NC}"
    echo "────────────────────────"
    START=$(date +%s%N)
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/detect" \
        -F "file=@perfilfoto.jpeg")
    END=$(date +%s%N)
    LATENCY=$(( (END - START) / 1000000 ))
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | head -n-1)
    
    if [ "$HTTP_CODE" = "200" ]; then
        RESULT=$(echo "$BODY" | grep -o '"response":"[^"]*"' | cut -d'"' -f4)
        METHOD=$(echo "$BODY" | grep -o '"method":"[^"]*"' | cut -d'"' -f4)
        
        if [ "$RESULT" = "is selfie" ]; then
            echo -e "${COLOR_GREEN}✓${NC} Correctly identified as SELFIE"
            echo "  Result: $RESULT"
            echo "  Method: $METHOD"
            echo "  Latency: ${LATENCY}ms"
        else
            echo -e "${COLOR_YELLOW}⚠${NC}  Unexpected result: $RESULT"
            echo "  Expected: is selfie"
            echo "  Response: $BODY"
        fi
    else
        echo -e "${COLOR_RED}✗${NC} Request failed (HTTP $HTTP_CODE)"
        echo "  Response: $BODY"
    fi
    echo ""
else
    echo -e "${COLOR_YELLOW}⚠${NC}  Skipping Test 3: perfilfoto.jpeg not found"
    echo ""
fi

# Test 4: Batch Processing
if [ -f "ceduladelantera.jpg" ] && [ -f "perfilfoto.jpeg" ]; then
    echo -e "${COLOR_BLUE}Test 4: Batch Processing${NC}"
    echo "───────────────────────"
    START=$(date +%s%N)
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/detect/batch" \
        -F "files=@ceduladelantera.jpg" \
        -F "files=@perfilfoto.jpeg")
    END=$(date +%s%N)
    LATENCY=$(( (END - START) / 1000000 ))
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | head -n-1)
    
    if [ "$HTTP_CODE" = "200" ]; then
        COUNT=$(echo "$BODY" | grep -o '"filename"' | wc -l)
        echo -e "${COLOR_GREEN}✓${NC} Batch processing successful"
        echo "  Files processed: $COUNT"
        echo "  Total latency: ${LATENCY}ms"
        echo "  Response (truncated):"
        echo "$BODY" | head -c 200
        echo "..."
    else
        echo -e "${COLOR_RED}✗${NC} Batch processing failed (HTTP $HTTP_CODE)"
    fi
    echo ""
else
    echo -e "${COLOR_YELLOW}⚠${NC}  Skipping Test 4: Missing image files"
    echo ""
fi

# Test 5: Error Handling - Invalid image
echo -e "${COLOR_BLUE}Test 5: Error Handling (Invalid File)${NC}"
echo "──────────────────────────────────────"
echo "Creating temporary invalid image..."
echo "not an image" > /tmp/invalid.txt

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/detect" \
    -F "file=@/tmp/invalid.txt")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "400" ]; then
    echo -e "${COLOR_GREEN}✓${NC} Correctly rejected invalid file (HTTP 400)"
    echo "  Error message: $(echo $BODY | grep -o '"detail":"[^"]*"')"
else
    echo -e "${COLOR_YELLOW}⚠${NC}  Unexpected response code: $HTTP_CODE"
fi

rm /tmp/invalid.txt
echo ""

# Test 6: API Documentation
echo -e "${COLOR_BLUE}Test 6: API Documentation${NC}"
echo "──────────────────────────"
RESPONSE=$(curl -s -w "\n%{http_code}" "$API_URL/docs")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${COLOR_GREEN}✓${NC} Swagger UI available (HTTP 200)"
    echo "  Access at: $API_URL/docs"
else
    echo -e "${COLOR_YELLOW}⚠${NC}  Swagger UI not available (HTTP $HTTP_CODE)"
fi
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Testing Complete                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${COLOR_GREEN}✓${NC} All tests completed!"
echo ""
echo "📊 Summary:"
echo "  - Health check: PASSED"
if [ -f "ceduladelantera.jpg" ]; then
    echo "  - Document detection: TESTED"
fi
if [ -f "perfilfoto.jpeg" ]; then
    echo "  - Selfie detection: TESTED"
fi
echo "  - Error handling: TESTED"
echo ""
echo "📖 Documentation: $API_URL/docs"
echo ""

