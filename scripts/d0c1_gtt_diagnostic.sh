#!/bin/bash
# D0c1: GTT/TTM diagnostic during Vulkan HOST load
# Measure GTT usage as HOST expert residency increases

cd ~/mesh-llm

echo "=== Baseline GTT ==="
cat /sys/class/drm/card*/device/mem_info_gtt_total
cat /sys/class/drm/card*/device/mem_info_gtt_used
cat /sys/class/drm/card*/device/mem_info_vram_total
cat /sys/class/drm/card*/device/mem_info_vram_used

echo "=== Poll GTT during 9g load ==="
# Start background GTT poller
(
  for i in $(seq 1 500); do
    echo "$(date +%s.%N) GTT_USED=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null) VRAM_USED=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null) MEMAVAIL=$(grep MemAvailable /proc/meminfo | awk '{print $2}')"
    sleep 0.1
  done
) > /tmp/gtt_poll_9g.log 2>&1 &
POLLER=$!

# Run 9g load
MESH_BACKEND=Vulkan0 /tmp/d0b1_memory_envelope /home/bamn/d0-stages-vulkan/d0-stage-0_9.gguf 0 9 /tmp/d0b1_vulkan_9g_diag.json 2>&1 | tail -5
kill $POLLER 2>/dev/null

echo "=== GTT during 9g load (sample) ==="
head -20 /tmp/gtt_poll_9g.log
echo "..."
tail -20 /tmp/gtt_poll_9g.log

echo "=== Post-9g GTT ==="
cat /sys/class/drm/card*/device/mem_info_gtt_total
cat /sys/class/drm/card*/device/mem_info_gtt_used
cat /sys/class/drm/card*/device/mem_info_vram_total
cat /sys/class/drm/card*/device/mem_info_vram_used

echo "=== Now attempt 17g with GTT poller ==="
(
  for i in $(seq 1 500); do
    echo "$(date +%s.%N) GTT_USED=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null) VRAM_USED=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null) MEMAVAIL=$(grep MemAvailable /proc/meminfo | awk '{print $2}')"
    sleep 0.1
  done
) > /tmp/gtt_poll_17g.log 2>&1 &
POLLER=$!

MESH_BACKEND=Vulkan0 /tmp/d0b1_memory_envelope /home/bamn/d0-stages-vulkan/d0-stage-0_17.gguf 0 17 /tmp/d0b1_vulkan_17g_diag.json 2>&1 | tail -5
kill $POLLER 2>/dev/null

echo "=== GTT at 17g failure ==="
grep -E "D0B1_LOAD_FAIL|radv|oom" /tmp/d0b1_vulkan_17g_diag*.log 2>/dev/null || true
tail -10 /tmp/gtt_poll_17g.log
