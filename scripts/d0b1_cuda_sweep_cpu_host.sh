#!/bin/bash
cd ~/mesh-llm

PKG=/home/joker/models/Qwen3.5-122B-A10B-Q3_K_XL-layer-package

for SIZE in 4 9 17 26; do
  STAGE=/tmp/d0-stages-cuda/d0-stage-0_${SIZE}.gguf
  RESULT=/tmp/d0-stages-cuda/d0b1_cuda_${SIZE}g.json
  LOG=/tmp/d0-stages-cuda/d0b1_cuda_${SIZE}g.log
  
  mkdir -p /tmp/d0-stages-cuda
  rm -f $STAGE $RESULT
  
  echo "=== [0,$SIZE) MATERIALIZE ==="
  /tmp/d0_materialize $PKG $SIZE $STAGE 2>&1 | grep OK
  
  echo "=== [0,$SIZE) MEASURE ==="
  /tmp/d0b1_memory_envelope $STAGE 0 $SIZE $RESULT > $LOG 2>&1
  echo "EXIT=$?"
  
  echo "=== RESULT ==="
  cat $RESULT
  
  echo "=== CLEANUP ==="
  rm -f $STAGE
done

echo "=== CUDA SWEEP COMPLETE ==="
