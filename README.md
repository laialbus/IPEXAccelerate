# IPEXAccelerate
Bridge enabling Intel IPEX optimizations with HuggingFace Accelerate for memory-efficient LLM inference on CPU

## 1. Environment Setup (Docker)

The environment is pre-packaged in a public Docker repository.

**Pull the image:**
```bash
docker pull laialbus/ece496:ipex_offload_v3.5
```
**Run the container:**

```bash
# Example mounting /fast-lab-share to /fast-lab-share inside the container
docker run --rm -it --privileged --network=host --ipc=host -v ~/:/home/storage -v /fast-lab-share:/fast-lab-share -v /fast-lab-share/albusl2/docker_tmp:/tmp laialbus/ece496:ipex_offload_v3.5 bash
```

---

# 2. Running the Code

The main entry point is `run.py`.

---

## Key Flags

- `--use-bridge`  
  Activates the IPEX-Accelerate Bridge. Remove this flag to run standard Accelerate benchmarks.

- `--ipex`  
  Enables Intel Extension for PyTorch optimizations.

- `--max-cpu-memory`  
  Limit for model weights in RAM (e.g., `5GiB`).

- `--max-disk-memory`
  Limit for model weights offloaded to disk (e.g., `100GiB`).

- `--offload-folder`  
  Directory for temporary disk offloading.

---

## Command Examples

---

### OPT-1.3B

#### Bridge Run (IPEX with Disk Offloading)

```bash
OMP_NUM_THREADS=24 numactl -m 0 -C 0-23 python run.py --benchmark \
    -m /home/storage/opt-1.3b/ \
    --dtype bfloat16 --ipex --use-bridge \
    --offload-folder /home/storage/albusl2/offload/opt-1.3b \
    --input-tokens 256 --max-new-tokens 8 --batch-size 1 \
    --num-iter 2 --num-warmup 1 --greedy > bridge.out
```

#### Accelerate ONLY Run (Baseline)

```bash
OMP_NUM_THREADS=24 numactl -m 0 -C 0-23 python run.py --benchmark \
    -m /home/storage/opt-1.3b/ \
    --dtype bfloat16 \
    --offload-folder /home/storage/albusl2/offload/opt-1.3b \
    --input-tokens 256 --max-new-tokens 8 --batch-size 1 \
    --num-iter 2 --num-warmup 1 --greedy > accel.out
```

---

### Llama-3-8B

#### Bridge Run

```bash
OMP_NUM_THREADS=24 numactl -m 0 -C 0-23 python run.py --benchmark \
    -m /home/storage/llama-3-8b/ \
    --max-cpu-memory 5GiB --max-disk-memory 100GiB \
    --dtype bfloat16 --ipex --use-bridge \
    --offload-folder /home/storage/albusl2/offload/llama-3-8b \
    --input-tokens 256 --max-new-tokens 8 --batch-size 1 \
    --num-iter 2 --num-warmup 1 --greedy > bridge_llama.out
```

#### Accelerate Only Run

```bash
OMP_NUM_THREADS=24 numactl -m 0 -C 0-23 python run.py --benchmark \
    -m /home/storage/llama-3-8b/ \
    --max-cpu-memory 5GiB --max-disk-memory 100GiB \
    --dtype bfloat16 \
    --offload-folder /home/storage/albusl2/offload/llama-3-8b \
    --input-tokens 256 --max-new-tokens 8 --batch-size 1 \
    --num-iter 2 --num-warmup 1 --greedy > accel_llama.out
```

---

## Massive Scale Runs (Dummy 640GB Model)

Configuration A verifies capability on extremely large models using shared storage for offloading.

---

### Configuration A: 350GB RAM / 350GB Disk (fast-lab-share offload)

```bash
OMP_NUM_THREADS=24 numactl --interleave=all -C 0-23 python run.py --benchmark \
    -m /fast-lab-share/albusl2/dummy_llama_640gb \
    --max-cpu-memory 350GiB --max-disk-memory 350GiB \
    --dtype bfloat16 --ipex --use-bridge \
    --offload-folder /fast-lab-share/albusl2/offload/dummy_640gb \
    --input-tokens 64 --max-new-tokens 4 --batch-size 1 \
    --num-iter 2 --num-warmup 1 --greedy &> ./bridge_640gb.out
```

---

### Configuration B: 550GB RAM / 120GB Disk (Local EMR3 offload)

```bash
OMP_NUM_THREADS=24 numactl --interleave=all -C 0-23 python run.py --benchmark \
    -m /fast-lab-share/albusl2/dummy_llama_640gb \
    --max-cpu-memory 550GiB --max-disk-memory 120GiB \
    --dtype bfloat16 --ipex --use-bridge \
    --offload-folder /home/storage/albusl2/offload/dummy_640gb \
    --input-tokens 64 --max-new-tokens 4 --batch-size 1 \
    --num-iter 2 --num-warmup 1 --greedy &> ./bridge_640gb_local.out
```

---

# 3. Remnant code
BridgeProfiler in run_generation.py can be ignored. It was used to debug the implementation, but the new code no longer uses it.
