#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
MODEL_FILE="Qwen3-8B-Q4_K_M.gguf"
IMAGE_NAME="forge-llama"

echo "=== Forge LLM Setup ==="
echo ""

# Step 1: Download model if not present
mkdir -p "$MODELS_DIR"
if [ -f "$MODELS_DIR/$MODEL_FILE" ]; then
    echo "[OK] Model already downloaded: $MODELS_DIR/$MODEL_FILE"
else
    echo "[..] Downloading $MODEL_FILE (~5 GB) via huggingface-cli..."
    echo "     (install with: uv tool install huggingface-hub[hf_xet])"
    hf download Qwen/Qwen3-8B-GGUF "$MODEL_FILE" --local-dir "$MODELS_DIR"
    echo "[OK] Model downloaded."
fi

echo ""

# Step 2: Build Docker image
echo "[..] Building Docker image '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" -f "$PROJECT_DIR/Dockerfile.llama" "$PROJECT_DIR"
echo "[OK] Image built."

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Run the server with:"
echo ""
echo "  docker run --rm -d \\"
echo "    --name forge-llama \\"
echo "    --device=/dev/kfd --device=/dev/dri \\"
echo "    --group-add video --group-add render \\"
echo "    --security-opt seccomp=unconfined \\"
echo "    -v $MODELS_DIR:/models \\"
echo "    -p 8080:8080 \\"
echo "    $IMAGE_NAME"
echo ""
echo "Server config: 32K context, 4 parallel slots (~8K per slot), 4K max generation."
echo "Adjust -c and -np in Dockerfile.llama if you need different tuning."
echo ""
echo "Then validate with:"
echo ""
echo "  python scripts/validate_model.py"
echo ""
