#!/bin/bash

# Test script for deepfake detection endpoints
# Usage: ./scripts/test_deepfake.sh [image_path] [video_path]

set -e

SERVICE_URL="http://localhost:8000"
CONTAINER_NAME="anti-spoofing-test"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🎭 DEEPFAKE DETECTION TEST SUITE                            ║"
echo "║  Testing: /analyze/deepfake/image & /analyze/deepfake/video  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if container is running
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "✅ Container '${CONTAINER_NAME}' is running"
else
    echo "⚠️ Container '${CONTAINER_NAME}' is not running"
    echo "Starting container..."
    docker-compose up -d ${CONTAINER_NAME}
    sleep 5
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        echo "✅ Container started successfully"
    else
        echo "❌ Failed to start container"
        exit 1
    fi
fi

# Wait for service to be ready
echo ""
echo "Waiting for service to be available..."
until curl --output /dev/null --silent --head --fail ${SERVICE_URL}/health; do
    printf '.'
    sleep 1
done
echo -e "\n✅ Service is available"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "TEST 1: Health Check"
echo "═══════════════════════════════════════════════════════════════"

HEALTH=$(curl -s ${SERVICE_URL}/health | jq '.')
echo "Response:"
echo "$HEALTH" | jq '.'

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "TEST 2: Deepfake Image Detection (Heuristic)"
echo "═══════════════════════════════════════════════════════════════"

if [ -z "$1" ]; then
    echo "⚠️ No image provided. Skipping image test."
    echo "Usage: $0 [image_path] [video_path]"
    echo "Example: $0 test_pics/selfie.jpg test_pics/video.mp4"
else
    IMAGE_PATH="$1"
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "❌ Image not found: $IMAGE_PATH"
    else
        echo "📸 Testing image: $IMAGE_PATH"
        echo ""
        RESPONSE=$(curl -s -X POST ${SERVICE_URL}/analyze/deepfake/image \
            -F "file=@${IMAGE_PATH}")
        
        echo "Response:"
        echo "$RESPONSE" | jq '.'
        
        # Extract result
        RESULT=$(echo "$RESPONSE" | jq -r '.response')
        CONFIDENCE=$(echo "$RESPONSE" | jq -r '.confidence')
        METHOD=$(echo "$RESPONSE" | jq -r '.method')
        
        echo ""
        echo "Result Summary:"
        echo "  Classification: $RESULT"
        echo "  Confidence:     $CONFIDENCE"
        echo "  Method:         $METHOD"
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "TEST 3: Deepfake Video Detection (Frame Aggregation)"
echo "═══════════════════════════════════════════════════════════════"

if [ -z "$2" ]; then
    echo "⚠️ No video provided. Skipping video test."
    echo "Usage: $0 [image_path] [video_path]"
else
    VIDEO_PATH="$2"
    if [ ! -f "$VIDEO_PATH" ]; then
        echo "❌ Video not found: $VIDEO_PATH"
    else
        echo "🎬 Testing video: $VIDEO_PATH"
        echo "   Parameters: frame_step=15, max_frames=30"
        echo ""
        echo "Processing video... (this may take a moment)"
        echo ""
        
        RESPONSE=$(curl -s -X POST "${SERVICE_URL}/analyze/deepfake/video?frame_step=15&max_frames=30" \
            -F "file=@${VIDEO_PATH}")
        
        echo "Response:"
        echo "$RESPONSE" | jq '.'
        
        # Extract results
        RESULT=$(echo "$RESPONSE" | jq -r '.response')
        CONF_MEAN=$(echo "$RESPONSE" | jq -r '.confidence_mean')
        CONF_MAX=$(echo "$RESPONSE" | jq -r '.confidence_max')
        FRAMES=$(echo "$RESPONSE" | jq -r '.frames_analyzed')
        
        echo ""
        echo "Result Summary:"
        echo "  Classification: $RESULT"
        echo "  Confidence (mean): $CONF_MEAN"
        echo "  Confidence (max):  $CONF_MAX"
        echo "  Frames analyzed:   $FRAMES"
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📊 TEST SUITE COMPLETE"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo "📝 Notes:"
echo "   - Heuristic accuracy: ~60-70% (without ML model)"
echo "   - ML accuracy: ~95-99% (with pre-trained model)"
echo "   - To integrate ML model: Place .pt file in models/"
echo "   - More details: See DEEPFAKE_ARCHITECTURE.md"
echo ""

exit 0

