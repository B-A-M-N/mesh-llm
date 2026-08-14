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

# Materialize stages (Vulkan backend)
export MESH_BACKEND=Vulkan0
PKG=/home/bamn/models/Qwen3.5-122B-A10B-Q3_K_XL-layer-package
mkdir -p /tmp/d0-stages-vulkan

for END in 4 9 17 26; do
  echo "=== Materializing [0,$END) ==="
  /tmp/d0_materialize $PKG $END /tmp/d0-stages-vulkan/d0-stage-0_${END}.gguf 2>&1 | grep "OK\|FAIL\|size"
done

# Run D0c Vulkan sweep
for END in 4 9 17 26; do
  echo "=== D0c Vulkan [0,$END) ==="
  /tmp/d0b1_memory_envelope /tmp/d0-stages-vulkan/d0-stage-0_${END}.gguf 0 $END /tmp/d0-stages-vulkan/d0b1_vulkan_${END}g.json > /tmp/d0-stages-vulkan/d0b1_${END}.log 2>&1
  echo "EXIT=$?"
  grep -E "D0B1_OK|D0B1_LOAD_FAIL" /tmp/d0-stages-vulkan/d0b1_${END}.log | tail -2
  grep "T4 warm" /tmp/d0-stages-vulkan/d0b1_${END}.log
done

echo "=== D0c Vulkan complete ==="
ls -la /tmp/d0-stages-vulkan/d0b1_vulkan_*.json
