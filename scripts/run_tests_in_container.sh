#!/bin/bash

# Run deepfake detection tests from inside Docker container
# This ensures cross-platform compatibility

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🐳 RUNNING TESTS INSIDE DOCKER CONTAINER                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Get container ID
CONTAINER_ID=$(docker-compose ps -q detector)

if [ -z "$CONTAINER_ID" ]; then
    echo "❌ Container not running. Starting it..."
    docker-compose up -d
    sleep 5
    CONTAINER_ID=$(docker-compose ps -q detector)
fi

echo "✅ Container ID: $CONTAINER_ID"
echo ""

# Run tests inside container
echo "🧪 Running random dataset test (20 samples)..."
echo ""

docker exec $CONTAINER_ID bash -c '
python3 << "EOF"
import os
import random
import requests
import json
from pathlib import Path

# Configuration
SERVICE_URL = "http://localhost:8000"
DATASET_DIR = "/app/Dataset/Test"
NUM_SAMPLES = 10  # per category

print("=" * 70)
print("🎭 DEEPFAKE DETECTION - RANDOM DATASET TEST")
print("=" * 70)
print()

# Check dataset
if not os.path.exists(DATASET_DIR):
    print(f"❌ Dataset not found at {DATASET_DIR}")
    exit(1)

# Count images
real_dir = f"{DATASET_DIR}/Real"
fake_dir = f"{DATASET_DIR}/Fake"

real_images = list(Path(real_dir).glob("*/*"))
fake_images = list(Path(fake_dir).glob("*/*"))

print(f"📊 Dataset statistics:")
print(f"  Real images: {len(real_images)}")
print(f"  Fake images: {len(fake_images)}")
print(f"  Total: {len(real_images) + len(fake_images)}")
print()

# Select random samples
random.seed(42)
real_samples = random.sample(real_images, min(NUM_SAMPLES, len(real_images)))
fake_samples = random.sample(fake_images, min(NUM_SAMPLES, len(fake_images)))

print(f"🎲 Selected {len(real_samples)} real + {len(fake_samples)} fake images")
print()

# Test function
def test_image(image_path, category):
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{SERVICE_URL}/analyze/deepfake/image",
                files=files,
                timeout=10
            )
            
        if response.status_code == 200:
            data = response.json()
            result = data.get('response', 'unknown')
            confidence = data.get('confidence', 0)
            
            # Check if correct
            if category == "Real":
                expected = "likely_real"
                correct = result == expected
            else:
                expected = "likely_deepfake"
                correct = result == expected
            
            status = "✓" if correct else "✗"
            color_code = "\033[32m" if correct else "\033[31m"
            
            print(f"{color_code}{status}\033[0m [{category}] {result} ({confidence:.2f})")
            return correct
        else:
            print(f"\033[31m✗\033[0m [{category}] Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"\033[31m✗\033[0m [{category}] Exception: {str(e)}")
        return False

# Test Real images
print("\033[36m" + "=" * 70 + "\033[0m")
print("\033[35mTESTING REAL IMAGES\033[0m")
print("\033[36m" + "=" * 70 + "\033[0m")

real_correct = 0
for image_path in real_samples:
    if test_image(str(image_path), "Real"):
        real_correct += 1

print()

# Test Fake images
print("\033[36m" + "=" * 70 + "\033[0m")
print("\033[35mTESTING FAKE IMAGES (DEEPFAKES)\033[0m")
print("\033[36m" + "=" * 70 + "\033[0m")

fake_correct = 0
for image_path in fake_samples:
    if test_image(str(image_path), "Fake"):
        fake_correct += 1

print()

# Summary
total_tested = len(real_samples) + len(fake_samples)
total_correct = real_correct + fake_correct
accuracy = (total_correct / total_tested * 100) if total_tested > 0 else 0
real_accuracy = (real_correct / len(real_samples) * 100) if len(real_samples) > 0 else 0
fake_accuracy = (fake_correct / len(fake_samples) * 100) if len(fake_samples) > 0 else 0

print("\033[36m" + "=" * 70 + "\033[0m")
print("\033[35mTEST SUMMARY\033[0m")
print("\033[36m" + "=" * 70 + "\033[0m")
print()
print("Overall Results:")
print(f"  Total tested:      {total_tested}")
print(f"  Correct:           {total_correct}")
print(f"  Accuracy:          {accuracy:.1f}%")
print()
print("Real Images:")
print(f"  Tested:            {len(real_samples)}")
print(f"  Correct:           {real_correct}")
print(f"  Accuracy:          {real_accuracy:.1f}%")
print()
print("Fake Images (Deepfakes):")
print(f"  Tested:            {len(fake_samples)}")
print(f"  Correct:           {fake_correct}")
print(f"  Accuracy:          {fake_accuracy:.1f}%")
print()
print("\033[36m" + "=" * 70 + "\033[0m")

# Color-coded result
if accuracy >= 80:
    print("\033[32m✓\033[0m Excellent accuracy: {:.1f}%".format(accuracy))
elif accuracy >= 60:
    print("\033[33m⚠\033[0m Good accuracy: {:.1f}%".format(accuracy))
else:
    print("\033[31m✗\033[0m Low accuracy: {:.1f}%".format(accuracy))

EOF
'

echo ""
echo "✅ Test execution completed!"

