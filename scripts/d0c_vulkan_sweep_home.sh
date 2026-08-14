#!/bin/bash
set -e

cd ~/mesh-llm

# Compile materializer
g++ -O2 -g -o /tmp/d0_materialize .deps/llama.cpp/d0_materialize.cpp \
  -I.deps/llama.cpp/include \
  -I.deps/llama.cpp/ggml/include \
  -Ldist/native-runtimes/meshllm-native-runtime-linux-x86_64-vulkan/lib \
  -l:libllama.so -l:libggml.so -l:libggml-base.so -l:libggml-cpu.so -l:libggml-vulkan.so -l:libmtmd.so \
  -Wl,-rpath,$PWD/dist/native-runtimes/meshllm-native-runtime-linux-x86_64-vulkan/lib \
  -lm -lpthread 2>&1 | tail -5
echo "MATERIALIZE_COMPILED"

# Compile measurement harness
g++ -O2 -g -o /tmp/d0b1_memory_envelope .deps/llama.cpp/d0b1_memory_envelope.cpp \
  -I.deps/llama.cpp/include \
  -I.deps/llama.cpp/ggml/include \
  -Ldist/native-runtimes/meshllm-native-runtime-linux-x86_64-vulkan/lib \
  -l:libllama.so -l:libggml.so -l:libggml-base.so -l:libggml-cpu.so -l:libggml-vulkan.so -l:libmtmd.so \
  -Wl,-rpath,$PWD/dist/native-runtimes/meshllm-native-runtime-linux-x86_64-vulkan/lib \
  -lm -lpthread 2>&1 | tail -5
echo "MEASUREMENT_COMPILED"

# Use /home/bamn (364G free) instead of /tmp (115G full)
STAGE_DIR=/home/bamn/d0-stages-vulkan
mkdir -p $STAGE_DIR
export MESH_BACKEND=Vulkan0
PKG=/home/bamn/models/Qwen3.5-122B-A10B-Q3_K_XL-layer-package

for END in 4 9 17 26; do
  echo "=== D0c Vulkan [0,$END) ==="
  
  # Materialize stage
  /tmp/d0_materialize $PKG $END $STAGE_DIR/d0-stage-0_${END}.gguf 2>&1 | grep "OK\|FAIL\|size"
  
  # Run measurement
  /tmp/d0b1_memory_envelope $STAGE_DIR/d0-stage-0_${END}.gguf 0 $END $STAGE_DIR/d0b1_vulkan_${END}g.json > $STAGE_DIR/d0b1_${END}.log 2>&1
  echo "EXIT=$?"
  grep -E "D0B1_OK|D0B1_LOAD_FAIL" $STAGE_DIR/d0b1_${END}.log | tail -2
  grep "T4 warm" $STAGE_DIR/d0b1_${END}.log || echo "no warm timing"
  
  # Copy results to local
  echo "=== Done [0,$END) ==="
  cp $STAGE_DIR/d0b1_vulkan_${END}g.json /tmp/d0b1_vulkan_${END}g.json 2>/dev/null || true
  
  # Clean up stage immediately
  rm -f $STAGE_DIR/d0-stage-0_${END}.gguf $STAGE_DIR/d0b1_${END}.log
done

echo "=== D0c Vulkan complete ==="
ls -la $STAGE_DIR/d0b1_vulkan_*.json
