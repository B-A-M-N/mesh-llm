#!/bin/bash
cd ~/mesh-llm

PKG=/home/bamn/models/Qwen3.5-122B-A10B-Q3_K_XL-layer-package
export MESH_BACKEND=Vulkan0

SIZE=26
STAGE=/tmp/d0-stages-vulkan/d0-stage-0_${SIZE}.gguf
RESULT=/tmp/d0-stages-vulkan/d0b1_vulkan_${SIZE}g.json
LOG=/tmp/d0-stages-vulkan/d0b1_vulkan_${SIZE}g.log

mkdir -p /tmp/d0-stages-vulkan
rm -f $STAGE $RESULT

echo "=== [0,$SIZE) MATERIALIZE ==="
/tmp/d0_materialize $PKG $SIZE $STAGE 2>&1 | grep OK

echo "=== GTT BEFORE ==="
cat /sys/class/drm/card*/device/mem_info_gtt_used

echo "=== [0,$SIZE) MEASURE ==="
/tmp/d0b1_memory_envelope $STAGE 0 $SIZE $RESULT > $LOG 2>&1
echo "EXIT=$?"

echo "=== GTT AFTER ==="
cat /sys/class/drm/card*/device/mem_info_gtt_used

echo "=== RESULT ==="
cat $RESULT
