source /data/gollamago-data/activate.sh
set -o pipefail

(
  set -e
  for dtype in bfloat16; do
    batches="1 8 32 64 128"
    [ "$dtype" != float32 ] || batches="1 8 32 64"

    for batch in $batches; do
      python -m benchmarks.benchmark_operators \
        --models llama3.2-1b \
        --backend tilelang --target maca \
        --dtype "$dtype" --batch-size "$batch" \
        --seq-lens 128 512 2048 4906 \
        --warmup 5 --repeat 50
    done
  done
) 2>&1 | tee /data/gollamago-data/logs/llama1b-large-batch.log