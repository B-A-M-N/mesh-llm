#!/bin/bash
# D0c Vulkan 17g diagnostic — classify the failure precisely
# Runs on worker (bamn@10.0.0.2)

cd ~/mesh-llm

PKG=/home/bamn/models/Qwen3.5-122B-A10B-Q3_K_XL-layer-package
STAGE_DIR=/home/bamn/d0-stages-vulkan
mkdir -p $STAGE_DIR
export MESH_BACKEND=Vulkan0

# Materialize 17g if not present
if [ ! -f "$STAGE_DIR/d0-stage-0_17.gguf" ]; then
  echo "=== Materializing 17g ==="
  /tmp/d0_materialize $PKG 17 $STAGE_DIR/d0-stage-0_17.gguf 2>&1 | tail -3
fi

echo "=== Pre-load memory state ==="
cat /proc/meminfo | grep -E "MemAvailable|SwapTotal|SwapFree"
echo "---"
cat /proc/self/status | grep -E "VmRSS|VmSize|VmSwap"
echo "---"

echo "=== Vulkan memory info (if available) ==="
# Try to get Vulkan memory info from the system
if command -v vulkaninfo &> /dev/null; then
  vulkaninfo --summary 2>/dev/null | grep -A5 "memory" || echo "no vulkaninfo memory summary"
  echo "---"
  vulkaninfo 2>/dev/null | grep -E "heap|memoryType|DEVICE_LOCAL|HOST_VISIBLE|budget" | head -30 || echo "no vulkaninfo heap details"
fi

echo "=== Run strace on 17g load to capture exact failure ==="
# Run with strace to capture mmap/madvise and any OOM kill
strace -f -e trace=memory -o $STAGE_DIR/strace_17g.log \
  /tmp/d0b1_memory_envelope $STAGE_DIR/d0-stage-0_17.gguf 0 17 $STAGE_DIR/d0b1_vulkan_17g_diag.json 2>&1 | grep -E "D0B1_OK|D0B1_LOAD_FAIL|error|VK_|radv|oom" | head -20

echo "=== Post-failure memory state ==="
cat /proc/meminfo | grep -E "MemAvailable|SwapTotal|SwapFree"
echo "---"
dmesg | tail -30 2>/dev/null | grep -iE "oom|killed|radv|vulkan|amdgpu" || echo "no OOM in dmesg"

echo "=== strace failure analysis ==="
grep -E "ENOMEM|SIGKILL|madvise|mmap.*FAILED" $STAGE_DIR/strace_17g.log 2>/dev/null | tail -20 || echo "no memory errors in strace"

echo "=== Diagnostic complete ==="
