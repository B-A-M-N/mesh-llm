#!/bin/bash
set -e

cd ~/mesh-llm

echo "=== Vulkan HOST fix ==="
# Apply the CPU buffer fix
sed -i 's|ggml_backend_buffer_type_t host_buft =\n.*ggml_backend_dev_host_buffer_type.*\n.*\n.*host_buft = ggml_backend_cpu_buffer_type.*|host_buft = ggml_backend_cpu_buffer_type();|' .deps/llama.cpp/src/skippy.cpp

echo "=== Building Vulkan native runtime ==="
cmake -S .deps/llama.cpp -B .deps/llama-build/vulkan-fix \
  -DGGML_VULKAN=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3

cmake --build .deps/llama-build/vulkan-fix --target llama-app -j$(nproc) 2>&1 | tail -5

echo "=== Packaging ==="
cp -L .deps/llama-build/vulkan-fix/bin/lib*.so* dist/native-runtimes/meshllm-native-runtime-linux-x86_64-vulkan/lib/

echo "=== Done ==="
