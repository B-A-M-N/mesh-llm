#!/bin/bash
# D0c Vulkan 4g regression with rebuilt runtime
cd ~/mesh-llm

PKG=/home/bamn/models/Qwen3.5-122B-A10B-Q3_K_XL-layer-package

echo "=== GTT BEFORE ==="
cat /sys/class/drm/card*/device/mem_info_gtt_used
cat /sys/class/drm/card*/device/mem_info_vram_used

echo "=== MATERIALIZE 4g ==="
rm -f /tmp/d0-test-4g.gguf
/tmp/d0_materialize $PKG 4 /tmp/d0-test-4g.gguf 2>&1 | grep OK

echo "=== RUN 4g MEASUREMENT ==="
rm -f /tmp/d0-test-4g.json
MESH_BACKEND=Vulkan0 /tmp/d0b1_memory_envelope /tmp/d0-test-4g.gguf 0 4 /tmp/d0-test-4g.json
echo "EXIT=$?"

echo "=== GTT AFTER ==="
cat /sys/class/drm/card*/device/mem_info_gtt_used
cat /sys/class/drm/card*/device/mem_info_vram_used

echo "=== RESULT JSON ==="
cat /tmp/d0-test-4g.json
