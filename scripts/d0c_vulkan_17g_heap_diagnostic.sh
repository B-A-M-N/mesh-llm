#!/bin/bash
# D0c Vulkan 17g diagnostic — per-heap failure classification
# Uses ggml_vulkan heap budget extension

cd ~/mesh-llm

PKG=/home/bamn/models/Qwen3.5-122B-A10B-Q3_K_XL-layer-package
STAGE_DIR=/home/bamn/d0-stages-vulkan
mkdir -p $STAGE_DIR
export MESH_BACKEND=Vulkan0

echo "=== Vulkan heap budget summary ==="
vulkaninfo 2>/dev/null | grep -A5 -E "heap|MEMORY_HEAP|memoryType|DEVICE_LOCAL|HOST_VISIBLE|budget|usage" | head -50

echo "=== Materialize 17g if needed ==="
if [ ! -f "$STAGE_DIR/d0-stage-0_17.gguf" ]; then
  /tmp/d0_materialize $PKG 17 $STAGE_DIR/d0-stage-0_17.gguf 2>&1 | tail -3
fi

echo "=== Capture Vulkan memory budget DURING load ==="
# Start a background process that polls Vulkan budget every 100ms
(
  while true; do
    echo "=== $(date +%s.%N) ==="
    cat /proc/meminfo | grep -E "MemAvailable|SwapFree"
    # Try to get Vulkan budget via vulkaninfo (may not work mid-load)
    vulkaninfo 2>/dev/null | grep -E "budget|usage" | head -10
    sleep 0.1
  done
) > $STAGE_DIR/vulkan_budget_poll.log 2>&1 &
POLLER_PID=$!

# Run the load
/tmp/d0b1_memory_envelope $STAGE_DIR/d0-stage-0_17.gguf 0 17 $STAGE_DIR/d0b1_vulkan_17g_diag.json > $STAGE_DIR/d0b1_17.log 2>&1
echo "EXIT=$?"

# Stop the poller
kill $POLLER_PID 2>/dev/null
wait $POLLER_PID 2>/dev/null

echo "=== Failure analysis ==="
grep -E "D0B1_OK|D0B1_LOAD_FAIL|radv|VK_ERROR|oom" $STAGE_DIR/d0b1_17.log | head -10

echo "=== Post-load budget ==="
cat /proc/meminfo | grep -E "MemAvailable|SwapFree"
echo "---"
vulkaninfo 2>/dev/null | grep -E "budget|usage|heap" | head -20

echo "=== Budget poll during load ==="
tail -30 $STAGE_DIR/vulkan_budget_poll.log 2>/dev/null || echo "no poll data"

echo "=== Diagnostic complete ==="
