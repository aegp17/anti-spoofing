#!/bin/bash

# Validation script for Anti-Spoofing Docker setup
# Usage: bash scripts/validate_docker.sh

set -e

echo "🔍 Validating Anti-Spoofing Docker Setup"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check Docker installation
echo "1️⃣  Checking Docker installation..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo -e "${GREEN}✓${NC} Docker installed: $DOCKER_VERSION"
else
    echo -e "${RED}✗${NC} Docker not found. Please install Docker Desktop."
    exit 1
fi

echo ""

# Check Docker daemon
echo "2️⃣  Checking Docker daemon..."
if docker ps &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker daemon is running"
else
    echo -e "${RED}✗${NC} Docker daemon is not running. Start Docker Desktop."
    exit 1
fi

echo ""

# Check image existence
echo "3️⃣  Checking image status..."
if docker image ls | grep -q "anti-spoofing"; then
    IMAGE_ID=$(docker image ls | grep "anti-spoofing" | awk '{print $3}' | head -1)
    echo -e "${GREEN}✓${NC} Image exists (ID: ${IMAGE_ID:0:12})"
else
    echo -e "${YELLOW}⚠${NC}  Image not found. Building..."
    docker build -t anti-spoofing:latest .
    echo -e "${GREEN}✓${NC} Image built successfully"
fi

echo ""

# Check for running container
echo "4️⃣  Checking for running container..."
if docker ps | grep -q "anti-spoofing"; then
    CONTAINER_ID=$(docker ps | grep "anti-spoofing" | awk '{print $1}')
    echo -e "${GREEN}✓${NC} Container running (ID: ${CONTAINER_ID:0:12})"
    CONTAINER_RUNNING=true
else
    echo -e "${YELLOW}⚠${NC}  No running container found"
    CONTAINER_RUNNING=false
fi

echo ""

# Check port availability
echo "5️⃣  Checking port 8000..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${GREEN}✓${NC} Port 8000 is in use (service running)"
else
    echo -e "${YELLOW}⚠${NC}  Port 8000 is available"
fi

echo ""

# Health check
if [ "$CONTAINER_RUNNING" = true ]; then
    echo "6️⃣  Performing health check..."
    HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null || echo "fail")
    
    if echo "$HEALTH" | grep -q "healthy"; then
        echo -e "${GREEN}✓${NC} Health check passed"
        echo "  Response: $(echo $HEALTH | jq .)"
    else
        echo -e "${RED}✗${NC} Health check failed"
        echo "  Make sure container is running: docker-compose up -d"
    fi
    echo ""
fi

# File structure check
echo "7️⃣  Checking project structure..."
REQUIRED_FILES=(
    "README.md"
    "main.py"
    "requirements.txt"
    "Dockerfile"
    "docker-compose.yml"
    "src/detector.py"
    "src/image_processor.py"
    "src/heuristic_detector.py"
    "src/ml_classifier.py"
    "config.py"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file (missing)"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All required files present"
else
    echo -e "${RED}✗${NC} Missing $MISSING files"
fi

echo ""
echo "=========================================="
echo "✅ Validation Complete"
echo ""

if [ "$CONTAINER_RUNNING" = true ]; then
    echo "🚀 Next steps:"
    echo "  1. Test endpoint: curl http://localhost:8000/health"
    echo "  2. View API docs: http://localhost:8000/docs"
    echo "  3. Try detection: curl -X POST http://localhost:8000/detect -F 'file=@ceduladelantera.jpg'"
else
    echo "🚀 Next steps:"
    echo "  1. Start container: docker-compose up -d"
    echo "  2. Run validation again: bash scripts/validate_docker.sh"
    echo "  3. Test endpoint: curl http://localhost:8000/health"
fi

echo ""

